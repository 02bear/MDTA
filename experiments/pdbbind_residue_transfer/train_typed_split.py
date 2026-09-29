#!/usr/bin/env python3
"""Fine-tune typed PLIP heads on an existing PDBbind split using rich ligand graphs."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from chemical_masks import INTERACTION_TYPES, pair_candidate_mask, parse_site_sequences
from typed_pair_model import TypedPairwiseContactPredictor, initialize_from_binary


TYPE_INDEX = {name: index for index, name in enumerate(INTERACTION_TYPES)}


def load_sample(site, rich_cache, pdb_id, device):
    protein = torch.load(
        site / "protein_graph" / f"{pdb_id}.pt",
        map_location=device,
        weights_only=False,
    )
    ligand = torch.load(
        rich_cache / f"{pdb_id}_ligand_rich.pt",
        map_location=device,
        weights_only=False,
    )
    return protein, ligand


def label_tensors(record, shape, device):
    typed = torch.zeros((len(INTERACTION_TYPES), *shape), dtype=torch.bool, device=device)
    for pair in record["pairs"]:
        typed[
            TYPE_INDEX[str(pair["type"])],
            int(pair["residue_index"]),
            int(pair["atom_index"]),
        ] = True
    union = typed.any(dim=0)
    residue = union.any(dim=1)
    return residue, union, typed


def sampled_loss(logits, positives, candidates, negative_ratio, zero_positive_negatives):
    candidates = candidates | positives
    flat_logits = logits.reshape(-1)
    flat_positive = positives.reshape(-1)
    flat_candidate = candidates.reshape(-1)
    positive_index = torch.nonzero(flat_positive, as_tuple=False).reshape(-1)
    negative_index = torch.nonzero(flat_candidate & ~flat_positive, as_tuple=False).reshape(-1)
    if positive_index.numel():
        wanted = min(negative_index.numel(), negative_ratio * positive_index.numel())
    else:
        wanted = min(negative_index.numel(), zero_positive_negatives)
    if wanted:
        negative_index = negative_index[
            torch.randperm(negative_index.numel(), device=logits.device)[:wanted]
        ]
    if not positive_index.numel():
        if not negative_index.numel():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            flat_logits[negative_index], torch.zeros_like(flat_logits[negative_index])
        )
    chosen = torch.cat((positive_index, negative_index))
    labels = torch.cat((
        torch.ones(positive_index.numel(), device=logits.device),
        torch.zeros(negative_index.numel(), device=logits.device),
    ))
    pos_weight = logits.new_tensor(max(negative_index.numel() / positive_index.numel(), 1.0))
    return F.binary_cross_entropy_with_logits(flat_logits[chosen], labels, pos_weight=pos_weight)


def safe_ap(labels, scores):
    if labels.sum() == 0 or labels.sum() == labels.size:
        return None
    return float(average_precision_score(labels, scores))


@torch.no_grad()
def evaluate(model, samples):
    model.eval()
    accumulators = {
        name: {"labels": [], "scores": [], "macro": []}
        for name in ("base", "typed_union_raw", "typed_union_masked")
    }
    typed_accumulators = {
        name: {"labels": [], "scores": [], "macro": []}
        for name in INTERACTION_TYPES
    }
    for pdb_id, protein, ligand, record, sequence in samples:
        _, base_logits, typed_logits = model(protein, ligand)
        _, union_labels, typed_labels = label_tensors(
            record, base_logits.shape, base_logits.device
        )
        base_scores = torch.sigmoid(base_logits)
        typed_scores = torch.sigmoid(typed_logits)
        masks = torch.stack([
            pair_candidate_mask(ligand.x, sequence, name)
            for name in INTERACTION_TYPES
        ])
        masked_typed_scores = typed_scores.masked_fill(~masks, 0.0)
        condition_scores = {
            "base": base_scores,
            "typed_union_raw": typed_scores.max(dim=0).values,
            "typed_union_masked": masked_typed_scores.max(dim=0).values,
        }
        flat_union = union_labels.cpu().numpy().reshape(-1)
        for name, scores in condition_scores.items():
            flat_scores = scores.cpu().numpy().reshape(-1)
            accumulators[name]["labels"].append(flat_union)
            accumulators[name]["scores"].append(flat_scores)
            ap = safe_ap(flat_union, flat_scores)
            if ap is not None:
                accumulators[name]["macro"].append(ap)
        for type_index, interaction_type in enumerate(INTERACTION_TYPES):
            labels = typed_labels[type_index].cpu().numpy().reshape(-1)
            scores = masked_typed_scores[type_index].cpu().numpy().reshape(-1)
            typed_accumulators[interaction_type]["labels"].append(labels)
            typed_accumulators[interaction_type]["scores"].append(scores)
            ap = safe_ap(labels, scores)
            if ap is not None:
                typed_accumulators[interaction_type]["macro"].append(ap)

    result = {}
    for name, values in accumulators.items():
        labels = np.concatenate(values["labels"])
        scores = np.concatenate(values["scores"])
        micro = float(average_precision_score(labels, scores))
        result[name] = {
            "auprc_micro": micro,
            "auprc_macro": float(np.mean(values["macro"])),
            "positive_rate": float(labels.mean()),
            "ap_enrichment": float(micro / labels.mean()),
        }
    result["per_type_masked"] = {}
    for name, values in typed_accumulators.items():
        labels = np.concatenate(values["labels"])
        scores = np.concatenate(values["scores"])
        result["per_type_masked"][name] = {
            "positive_count": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "auprc_micro": float(average_precision_score(labels, scores)),
            "auprc_macro": float(np.mean(values["macro"])) if values["macro"] else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtbind-root", type=Path, required=True)
    parser.add_argument("--rich-cache", type=Path, required=True)
    parser.add_argument("--pair-labels", type=Path, required=True)
    parser.add_argument("--split-metrics", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--zero-positive-negatives", type=int, default=8)
    parser.add_argument("--residue-loss-weight", type=float, default=0.5)
    parser.add_argument("--base-pair-loss-weight", type=float, default=0.5)
    parser.add_argument("--typed-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    site = args.dtbind_root / "Data" / "site"
    sequences = parse_site_sequences(site / "site_labels.txt")
    label_payload = json.loads(args.pair_labels.read_text())
    records = {record["pdb_id"]: record for record in label_payload["records"]}
    split_ids = json.loads(args.split_metrics.read_text())["split_ids"]
    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    samples, skipped = {}, defaultdict(list)
    for split in ("train", "val", "test"):
        ids = split_ids[split][: limits[split]] if limits[split] else split_ids[split]
        current = []
        for pdb_id in ids:
            record, sequence = records.get(pdb_id), sequences.get(pdb_id)
            rich_path = args.rich_cache / f"{pdb_id}_ligand_rich.pt"
            if record is None or sequence is None or not rich_path.exists():
                skipped[split].append(pdb_id)
                continue
            protein, ligand = load_sample(site, args.rich_cache, pdb_id, args.device)
            if (
                protein.x.shape[0] != record["residue_count"]
                or ligand.x.shape[0] != record["atom_count"]
                or len(sequence) != record["residue_count"]
            ):
                skipped[split].append(pdb_id)
                continue
            current.append((pdb_id, protein, ligand, record, sequence))
        samples[split] = current

    model = TypedPairwiseContactPredictor(
        hidden_dim=args.hidden_dim, temperature=0.5
    ).to(args.device)
    warm_start = initialize_from_binary(model, args.pretrained)
    model.to(args.device)
    initial_val = evaluate(model, samples["val"])
    initial_test = evaluate(model, samples["test"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed)
    history, best_value, best_state, epochs_without_improvement = [], -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(samples["train"])))
        rng.shuffle(order)
        loss_total = residue_total = base_total = typed_total = 0.0
        for sample_index in order:
            _, protein, ligand, record, sequence = samples["train"][sample_index]
            residue_logits, base_logits, typed_logits = model(protein, ligand)
            residue_labels, union_labels, typed_labels = label_tensors(
                record, base_logits.shape, base_logits.device
            )
            residue_loss = F.binary_cross_entropy_with_logits(
                residue_logits, residue_labels.float(),
                pos_weight=residue_logits.new_tensor(
                    min((~residue_labels).sum().item() / max(residue_labels.sum().item(), 1), 30.0)
                ),
            )
            base_loss = sampled_loss(
                base_logits,
                union_labels,
                torch.ones_like(union_labels),
                args.negative_ratio,
                args.zero_positive_negatives,
            )
            type_losses = []
            for type_index, interaction_type in enumerate(INTERACTION_TYPES):
                candidates = pair_candidate_mask(ligand.x, sequence, interaction_type)
                type_losses.append(sampled_loss(
                    typed_logits[type_index],
                    typed_labels[type_index],
                    candidates,
                    args.negative_ratio,
                    args.zero_positive_negatives,
                ))
            typed_loss = torch.stack(type_losses).mean()
            loss = (
                args.residue_loss_weight * residue_loss
                + args.base_pair_loss_weight * base_loss
                + args.typed_loss_weight * typed_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += loss.item()
            residue_total += residue_loss.item()
            base_total += base_loss.item()
            typed_total += typed_loss.item()

        validation = evaluate(model, samples["val"])
        row = {
            "epoch": epoch,
            "loss": loss_total / len(samples["train"]),
            "residue_loss": residue_total / len(samples["train"]),
            "base_pair_loss": base_total / len(samples["train"]),
            "typed_loss": typed_total / len(samples["train"]),
            "validation": validation,
        }
        history.append(row)
        selection = validation["typed_union_masked"]["auprc_macro"]
        print(json.dumps({
            "epoch": epoch,
            "loss": row["loss"],
            "base_micro": validation["base"]["auprc_micro"],
            "typed_masked_micro": validation["typed_union_masked"]["auprc_micro"],
            "typed_masked_macro": selection,
        }), flush=True)
        if selection > best_value:
            best_value = selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(json.dumps({
                "early_stop_epoch": epoch,
                "best_validation_macro": best_value,
                "patience": args.patience,
            }), flush=True)
            break

    model.load_state_dict(best_state)
    final_test = evaluate(model, samples["test"])
    result = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sample_counts": {key: len(value) for key, value in samples.items()},
        "skipped": dict(skipped),
        "warm_start": warm_start,
        "initial_validation": initial_val,
        "initial_test": initial_test,
        "best_validation_macro": best_value,
        "final_test": final_test,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_dir / "typed_best.pt")
    (args.output_dir / "typed_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("FINAL", json.dumps(final_test, indent=2), flush=True)


if __name__ == "__main__":
    main()
