#!/usr/bin/env python3
"""Overfit a 20-sample pilot and measure ligand-conditioning diagnostics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from pilot_model import ResidueContactPredictor


def load_ids(site_dir: Path, refined_dir: Path, count: int, seed: int):
    refined = {p.parent.name.lower() for p in refined_dir.glob("*/*_protein.pdb")}
    protein = {p.stem.lower() for p in (site_dir / "protein_graph").glob("*.pt")}
    ligand = {
        p.name.lower().removesuffix("_ligand.pt")
        for p in (site_dir / "ligand_graph").glob("*_ligand.pt")
    }
    candidates = []
    for pdb_id in sorted(refined & protein & ligand):
        graph = torch.load(
            site_dir / "protein_graph" / f"{pdb_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        labels = graph.y.reshape(-1)
        if 40 <= labels.numel() <= 800 and 0 < labels.sum().item() < labels.numel():
            candidates.append(pdb_id)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def load_pair(site_dir, pdb_id, device):
    protein = torch.load(
        site_dir / "protein_graph" / f"{pdb_id}.pt",
        map_location=device,
        weights_only=False,
    )
    ligand = torch.load(
        site_dir / "ligand_graph" / f"{pdb_id}_ligand.pt",
        map_location=device,
        weights_only=False,
    )
    return protein, ligand


def metrics(labels, scores):
    labels = np.concatenate(labels)
    scores = np.concatenate(scores)
    return {
        "auprc_micro": float(average_precision_score(labels, scores)),
        "auroc_micro": float(roc_auc_score(labels, scores)),
        "positive_rate": float(labels.mean()),
        "ap_enrichment": float(average_precision_score(labels, scores) / labels.mean()),
    }


@torch.no_grad()
def evaluate(model, pairs, mismatch=False):
    model.eval()
    labels, scores = [], []
    map_changes = []
    for index, (pdb_id, protein, ligand) in enumerate(pairs):
        chosen_ligand = pairs[(index + 1) % len(pairs)][2] if mismatch else ligand
        logits = model(protein, chosen_ligand)
        labels.append(protein.y.reshape(-1).cpu().numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
        if mismatch:
            true_scores = torch.sigmoid(model(protein, ligand))
            map_changes.append((true_scores - torch.sigmoid(logits)).abs().mean().item())
    result = metrics(labels, scores)
    if mismatch:
        result["mean_abs_map_change"] = float(np.mean(map_changes))
    return result


def train_one(args, pairs, protein_only):
    model = ResidueContactPredictor(
        hidden_dim=args.hidden_dim, protein_only=protein_only
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positives = sum(pair[1].y.sum().item() for pair in pairs)
    negatives = sum(pair[1].y.numel() - pair[1].y.sum().item() for pair in pairs)
    pos_weight = torch.tensor(min(negatives / max(positives, 1), 30.0), device=args.device)
    history = []
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(pairs)))
        rng.shuffle(order)
        total_loss = 0.0
        for index in order:
            _, protein, ligand = pairs[index]
            labels = protein.y.reshape(-1).float()
            logits = model(protein, ligand)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            current = evaluate(model, pairs)
            current.update({"epoch": epoch, "loss": total_loss / len(pairs)})
            history.append(current)
            print(("protein_only" if protein_only else "conditioned"), current, flush=True)
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtbind-root", type=Path, required=True)
    parser.add_argument("--pdbbind-refined", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    site_dir = args.dtbind_root / "Data" / "site"
    ids = load_ids(site_dir, args.pdbbind_refined, args.count, args.seed)
    pairs = [(pdb_id, *load_pair(site_dir, pdb_id, args.device)) for pdb_id in ids]
    print("sample_ids", ids, flush=True)
    conditioned, conditioned_history = train_one(args, pairs, protein_only=False)
    protein_only, protein_only_history = train_one(args, pairs, protein_only=True)
    result = {
        "sample_ids": ids,
        "conditioned": evaluate(conditioned, pairs),
        "conditioned_mismatched": evaluate(conditioned, pairs, mismatch=True),
        "protein_only": evaluate(protein_only, pairs),
        "conditioned_history": conditioned_history,
        "protein_only_history": protein_only_history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(conditioned.state_dict(), args.output_dir / "conditioned.pt")
    torch.save(protein_only.state_dict(), args.output_dir / "protein_only.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
