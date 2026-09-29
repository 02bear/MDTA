#!/usr/bin/env python3
"""Train/evaluate the pilot on disjoint train/validation/test samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from pilot_model import PairwiseResidueContactPredictor, ResidueContactPredictor
from train_smoke import load_ids, load_pair


def aggregate_metrics(model, pairs, mismatched=False, atom_shuffle=False):
    model.eval()
    labels_all, scores_all, per_sample_ap, changes = [], [], [], []
    with torch.no_grad():
        for i, (_, protein, ligand) in enumerate(pairs):
            used_ligand = pairs[(i + 1) % len(pairs)][2] if mismatched else ligand
            if atom_shuffle:
                used_ligand = used_ligand.clone()
                permutation = torch.arange(
                    used_ligand.x.shape[0] - 1, -1, -1, device=used_ligand.x.device
                )
                used_ligand.x = used_ligand.x[permutation]
            scores = torch.sigmoid(model(protein, used_ligand)).cpu().numpy()
            labels = protein.y.reshape(-1).cpu().numpy()
            labels_all.append(labels)
            scores_all.append(scores)
            if 0 < labels.sum() < labels.size:
                per_sample_ap.append(average_precision_score(labels, scores))
            if mismatched or atom_shuffle:
                true_scores = torch.sigmoid(model(protein, ligand)).cpu().numpy()
                changes.append(float(np.abs(true_scores - scores).mean()))
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    result = {
        "auprc_micro": float(average_precision_score(labels, scores)),
        "auprc_macro": float(np.mean(per_sample_ap)),
        "auroc_micro": float(roc_auc_score(labels, scores)),
        "positive_rate": float(labels.mean()),
        "ap_enrichment": float(average_precision_score(labels, scores) / labels.mean()),
    }
    if mismatched or atom_shuffle:
        result["mean_abs_map_change"] = float(np.mean(changes))
    return result


@torch.no_grad()
def pair_metrics(model, pairs, pair_labels):
    model.eval()
    labels_all, scores_all, macro_ap = [], [], []
    for pdb_id, protein, ligand in pairs:
        record = pair_labels.get(pdb_id)
        if record is None:
            continue
        _, pair_scores = model(protein, ligand, return_pair_scores=True)
        labels = torch.zeros_like(pair_scores)
        for pair in record["pairs"]:
            labels[pair["residue_index"], pair["atom_index"]] = 1.0
        flat_labels = labels.cpu().numpy().reshape(-1)
        flat_scores = torch.sigmoid(pair_scores).cpu().numpy().reshape(-1)
        labels_all.append(flat_labels)
        scores_all.append(flat_scores)
        if flat_labels.sum() > 0:
            macro_ap.append(average_precision_score(flat_labels, flat_scores))
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    return {
        "pair_auprc_micro": float(average_precision_score(labels, scores)),
        "pair_auprc_macro": float(np.mean(macro_ap)),
        "pair_positive_rate": float(labels.mean()),
        "pair_ap_enrichment": float(average_precision_score(labels, scores) / labels.mean()),
    }


def sampled_pair_loss(pair_scores, record, negative_ratio):
    positives = sorted(
        {(pair["residue_index"], pair["atom_index"]) for pair in record["pairs"]}
    )
    if not positives:
        return pair_scores.sum() * 0.0
    positive_flat = torch.tensor(
        [residue * pair_scores.shape[1] + atom for residue, atom in positives],
        device=pair_scores.device,
    )
    mask = torch.ones(pair_scores.numel(), dtype=torch.bool, device=pair_scores.device)
    mask[positive_flat] = False
    negative_flat = torch.nonzero(mask, as_tuple=False).reshape(-1)
    wanted = min(negative_flat.numel(), negative_ratio * positive_flat.numel())
    negative_flat = negative_flat[
        torch.randperm(negative_flat.numel(), device=pair_scores.device)[:wanted]
    ]
    chosen = torch.cat([positive_flat, negative_flat])
    labels = torch.cat(
        [
            pair_scores.new_ones(positive_flat.numel()),
            pair_scores.new_zeros(negative_flat.numel()),
        ]
    )
    logits = pair_scores.reshape(-1)[chosen]
    pos_weight = pair_scores.new_tensor(negative_flat.numel() / positive_flat.numel())
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def train_model(args, train_pairs, val_pairs, protein_only, pair_labels):
    model_class = (
        PairwiseResidueContactPredictor
        if args.architecture == "pairwise"
        else ResidueContactPredictor
    )
    model_kwargs = {"hidden_dim": args.hidden_dim, "protein_only": protein_only}
    if args.architecture == "pairwise":
        model_kwargs.update(
            temperature=args.temperature, atom_dropout=args.atom_dropout
        )
    model = model_class(**model_kwargs).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positives = sum(p.y.sum().item() for _, p, _ in train_pairs)
    negatives = sum(p.y.numel() - p.y.sum().item() for _, p, _ in train_pairs)
    pos_weight = torch.tensor(min(negatives / max(positives, 1), 30), device=args.device)
    best_ap, best_state, history = -1.0, None, []
    rng = random.Random(args.seed + int(protein_only))
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_pairs)))
        rng.shuffle(order)
        loss_sum = 0.0
        for idx in order:
            pdb_id, protein, ligand = train_pairs[idx]
            if args.pair_labels and not protein_only:
                logits, pair_scores = model(protein, ligand, return_pair_scores=True)
            else:
                logits = model(protein, ligand)
            labels = protein.y.reshape(-1).float()
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            if args.pair_labels and not protein_only:
                loss = loss + args.pair_loss_weight * sampled_pair_loss(
                    pair_scores, pair_labels[pdb_id], args.pair_negative_ratio
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()
        val = aggregate_metrics(model, val_pairs)
        if args.pair_labels and not protein_only:
            val.update(pair_metrics(model, val_pairs, pair_labels))
        row = {"epoch": epoch, "loss": loss_sum / len(train_pairs), **val}
        history.append(row)
        print(("protein_only" if protein_only else "conditioned"), row, flush=True)
        if args.selection_metric == "pair_macro" and args.pair_labels and not protein_only:
            selection_value = val["pair_auprc_macro"]
        elif args.selection_metric == "composite" and args.pair_labels and not protein_only:
            selection_value = 0.5 * val["auprc_macro"] + 0.5 * val["pair_auprc_macro"]
        else:
            selection_value = val["auprc_macro"]
        row["selection_value"] = selection_value
        if selection_value > best_ap:
            best_ap = selection_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtbind-root", type=Path, required=True)
    parser.add_argument("--pdbbind-refined", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--architecture", choices=("global", "pairwise"), default="global"
    )
    parser.add_argument("--skip-protein-only", action="store_true")
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--split-key", default="split_ids")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--atom-dropout", type=float, default=0.1)
    parser.add_argument("--pair-labels", type=Path)
    parser.add_argument("--pair-loss-weight", type=float, default=0.5)
    parser.add_argument("--pair-negative-ratio", type=int, default=20)
    parser.add_argument(
        "--selection-metric",
        choices=("residue_macro", "pair_macro", "composite"),
        default="residue_macro",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    site = args.dtbind_root / "Data" / "site"
    if args.split_json:
        split_payload = json.loads(args.split_json.read_text())
        for part in args.split_key.split("."):
            split_payload = split_payload[part]
        split_ids = {name: list(split_payload[name]) for name in ("train", "val", "test")}
    else:
        ids = load_ids(site, args.pdbbind_refined, args.count, args.data_seed)
        n_train, n_val = int(0.8 * len(ids)), int(0.1 * len(ids))
        split_ids = {
            "train": ids[:n_train],
            "val": ids[n_train : n_train + n_val],
            "test": ids[n_train + n_val :],
        }
    pair_labels = {}
    if args.pair_labels:
        payload = json.loads(args.pair_labels.read_text())
        pair_labels = {record["pdb_id"]: record for record in payload["records"]}
        for split in split_ids:
            split_ids[split] = [pdb_id for pdb_id in split_ids[split] if pdb_id in pair_labels]
    pairs = {
        split: [(pdb_id, *load_pair(site, pdb_id, args.device)) for pdb_id in subset]
        for split, subset in split_ids.items()
    }
    if args.pair_labels:
        for split in pairs:
            pairs[split] = [
                item for item in pairs[split]
                if pair_labels[item[0]]["atom_count"] == item[2].x.shape[0]
                and pair_labels[item[0]]["residue_count"] == item[1].x.shape[0]
            ]
            split_ids[split] = [item[0] for item in pairs[split]]
    conditioned, conditioned_history = train_model(
        args, pairs["train"], pairs["val"], protein_only=False, pair_labels=pair_labels
    )
    result = {
        "split_ids": split_ids,
        "conditioned_test": aggregate_metrics(conditioned, pairs["test"]),
        "conditioned_test_mismatched": aggregate_metrics(
            conditioned, pairs["test"], mismatched=True
        ),
        "conditioned_test_atom_shuffle": aggregate_metrics(
            conditioned, pairs["test"], atom_shuffle=True
        ),
        "conditioned_history": conditioned_history,
    }
    protein_only = None
    if not args.skip_protein_only:
        protein_only, protein_only_history = train_model(
            args, pairs["train"], pairs["val"], protein_only=True, pair_labels=pair_labels
        )
        result["protein_only_test"] = aggregate_metrics(protein_only, pairs["test"])
        result["protein_only_history"] = protein_only_history
    if args.pair_labels:
        result["conditioned_pair_test"] = pair_metrics(
            conditioned, pairs["test"], pair_labels
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pilot_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(conditioned.state_dict(), args.output_dir / "conditioned_best.pt")
    if protein_only is not None:
        torch.save(protein_only.state_dict(), args.output_dir / "protein_only_best.pt")
    print("FINAL", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
