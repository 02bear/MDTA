#!/usr/bin/env python3
"""Post-hoc causal and Top-K diagnostics for a typed rich-ligand checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chemical_masks import INTERACTION_TYPES, pair_candidate_mask, parse_site_sequences
from evaluate_pair_diagnostics import pair_summary, residue_summary, safe_ap, topk_metrics
from train_typed_split import label_tensors, load_sample
from typed_pair_model import TypedPairwiseContactPredictor


def perturb(graph, mode, rng):
    changed = graph.clone()
    if mode == "zero":
        changed.x = torch.zeros_like(changed.x)
    else:
        order = torch.as_tensor(
            rng.permutation(changed.x.shape[0]), dtype=torch.long, device=changed.x.device
        )
        changed.x = changed.x[order]
    return changed


def add_pair_row(rows, labels, scores):
    flat_labels = labels.cpu().numpy().reshape(-1)
    flat_scores = scores.detach().cpu().numpy().reshape(-1)
    rows.append({
        "labels": flat_labels,
        "scores": flat_scores,
        "ap": safe_ap(flat_labels, flat_scores),
        "topk": topk_metrics(flat_labels, flat_scores),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtbind-root", type=Path, required=True)
    parser.add_argument("--rich-cache", type=Path, required=True)
    parser.add_argument("--pair-labels", type=Path, required=True)
    parser.add_argument("--split-metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    site = args.dtbind_root / "Data" / "site"
    sequences = parse_site_sequences(site / "site_labels.txt")
    records = {
        record["pdb_id"]: record
        for record in json.loads(args.pair_labels.read_text())["records"]
    }
    test_ids = json.loads(args.split_metrics.read_text())["split_ids"]["test"]
    samples, skipped = [], []
    for pdb_id in test_ids:
        if pdb_id not in records or pdb_id not in sequences:
            skipped.append(pdb_id)
            continue
        rich_path = args.rich_cache / f"{pdb_id}_ligand_rich.pt"
        if not rich_path.exists():
            skipped.append(pdb_id)
            continue
        protein, ligand = load_sample(site, args.rich_cache, pdb_id, args.device)
        record, sequence = records[pdb_id], sequences[pdb_id]
        if protein.x.shape[0] != len(sequence) or ligand.x.shape[0] != record["atom_count"]:
            skipped.append(pdb_id)
            continue
        samples.append((pdb_id, protein, ligand, record, sequence))

    model = TypedPairwiseContactPredictor(hidden_dim=args.hidden_dim).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    rng = np.random.default_rng(args.seed)
    pair_rows = {
        f"{head}_{condition}": []
        for head in ("base", "typed_masked")
        for condition in ("correct", "atom_feature_shuffle", "atom_features_zero", "residue_feature_shuffle")
    }
    residue_rows = {name: [] for name in (
        "correct", "atom_feature_shuffle", "atom_features_zero",
        "residue_feature_shuffle", "mismatched_ligand",
    )}
    sample_details = []

    with torch.no_grad():
        for index, (pdb_id, protein, ligand, record, sequence) in enumerate(samples):
            original_masks = torch.stack([
                pair_candidate_mask(ligand.x, sequence, name)
                for name in INTERACTION_TYPES
            ])
            residue_labels, union_labels, _ = label_tensors(
                record, (protein.x.shape[0], ligand.x.shape[0]), protein.x.device
            )
            conditions = {
                "correct": (protein, ligand),
                "atom_feature_shuffle": (protein, perturb(ligand, "shuffle", rng)),
                "atom_features_zero": (protein, perturb(ligand, "zero", rng)),
                "residue_feature_shuffle": (perturb(protein, "shuffle", rng), ligand),
            }
            detail = {"pdb_id": pdb_id, "conditions": {}}
            correct_typed = None
            for condition, (used_protein, used_ligand) in conditions.items():
                residue_logits, base_logits, typed_logits = model(used_protein, used_ligand)
                base_scores = torch.sigmoid(base_logits)
                typed_scores = torch.sigmoid(typed_logits).masked_fill(~original_masks, 0.0)
                typed_union = typed_scores.max(dim=0).values
                add_pair_row(pair_rows[f"base_{condition}"], union_labels, base_scores)
                add_pair_row(pair_rows[f"typed_masked_{condition}"], union_labels, typed_union)
                residue_rows[condition].append({
                    "labels": residue_labels.cpu().numpy().reshape(-1),
                    "scores": torch.sigmoid(residue_logits).cpu().numpy().reshape(-1),
                })
                if condition == "correct":
                    correct_typed = typed_union
                detail["conditions"][condition] = {
                    "typed_pair_ap": safe_ap(
                        union_labels.cpu().numpy().reshape(-1),
                        typed_union.cpu().numpy().reshape(-1),
                    ),
                    "mean_abs_typed_map_change": None,
                }
                if condition != "correct":
                    detail["conditions"][condition]["mean_abs_typed_map_change"] = float(
                        (correct_typed - typed_union).abs().mean()
                    )
            mismatch_ligand = samples[(index + 1) % len(samples)][2]
            mismatch_residue_logits, _, _ = model(protein, mismatch_ligand)
            residue_rows["mismatched_ligand"].append({
                "labels": residue_labels.cpu().numpy().reshape(-1),
                "scores": torch.sigmoid(mismatch_residue_logits).cpu().numpy().reshape(-1),
            })
            sample_details.append(detail)

    pair_results = {name: pair_summary(rows) for name, rows in pair_rows.items()}
    for head in ("base", "typed_masked"):
        correct = pair_results[f"{head}_correct"]["auprc_micro"]
        for condition in ("atom_feature_shuffle", "atom_features_zero", "residue_feature_shuffle"):
            key = f"{head}_{condition}"
            pair_results[key]["relative_micro_ap_change_vs_correct"] = float(
                (pair_results[key]["auprc_micro"] - correct) / correct
            )
    result = {
        "evaluated_count": len(samples),
        "skipped": skipped,
        "pair_conditions": pair_results,
        "residue_conditions": {
            name: residue_summary(rows) for name, rows in residue_rows.items()
        },
        "notes": [
            "All perturbed typed predictions use the original ligand's fixed chemistry masks.",
            "Mismatched ligands are evaluated only at residue level because atom indices are not comparable.",
            "Residue labels are the union of PLIP pair labels, not the legacy DTBind site labels.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "typed_diagnostics_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "typed_per_complex.jsonl").open("w", encoding="utf-8") as handle:
        for detail in sample_details:
            handle.write(json.dumps(detail) + "\n")
    print(json.dumps({
        "evaluated_count": len(samples),
        "pair_micro": {name: values["auprc_micro"] for name, values in pair_results.items()},
        "residue_micro": {name: residue_summary(rows)["auprc_micro"] for name, rows in residue_rows.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
