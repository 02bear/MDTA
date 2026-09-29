#!/usr/bin/env python3
"""Detailed diagnostics for a trained PDBbind atom-residue pair predictor.

This script deliberately keeps the existing checkpoint and labels read-only.  A
wrong ligand has no atom-index correspondence with the true ligand, so mismatch
is evaluated at residue level only.  Pair-level causal controls preserve the
matrix shape: atom-feature permutation/ablation and residue-feature permutation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from pilot_model import PairwiseResidueContactPredictor
from train_smoke import load_pair


FIXED_K = (5, 10, 20)


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return None
    return float(average_precision_score(labels, scores))


def distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
    }


def topk_metrics(labels: np.ndarray, scores: np.ndarray):
    labels = labels.astype(bool, copy=False)
    positive_count = int(labels.sum())
    order = np.argsort(-scores, kind="stable")
    result = {}
    for name, requested_k in [("true_count", positive_count), *[(str(k), k) for k in FIXED_K]]:
        k = min(requested_k, labels.size)
        hits = int(labels[order[:k]].sum()) if k else 0
        result[name] = {
            "k": int(k),
            "hits": hits,
            "precision": float(hits / k) if k else None,
            "recall": float(hits / positive_count) if positive_count else None,
            "any_hit": bool(hits),
        }
    return result


def summarize_topk(rows):
    summary = {}
    for key in ["true_count", *map(str, FIXED_K)]:
        values = [row["topk"][key] for row in rows]
        summary[key] = {
            "mean_k": float(np.mean([item["k"] for item in values])),
            "mean_precision": float(np.mean([item["precision"] for item in values if item["precision"] is not None])),
            "mean_recall": float(np.mean([item["recall"] for item in values if item["recall"] is not None])),
            "hits_rate": float(np.mean([item["any_hit"] for item in values])),
        }
    return summary


def pr_summary(labels: np.ndarray, scores: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    operating_points = {}
    for target in (0.10, 0.25, 0.50, 0.75):
        eligible = np.flatnonzero(recall >= target)
        if eligible.size:
            best = eligible[np.argmax(precision[eligible])]
            operating_points[str(target)] = {
                "precision": float(precision[best]),
                "recall": float(recall[best]),
                "threshold": float(thresholds[best]) if best < thresholds.size else None,
            }
    # Compact precision envelope at fixed recall targets; avoid a multi-million-row PR file.
    curve = []
    for target in np.linspace(0.0, 1.0, 101):
        eligible = np.flatnonzero(recall >= target)
        curve.append({
            "recall_target": float(target),
            "best_precision": float(precision[eligible].max()) if eligible.size else None,
        })
    return operating_points, curve


def residue_summary(rows):
    labels = np.concatenate([row["labels"] for row in rows])
    scores = np.concatenate([row["scores"] for row in rows])
    macro = [safe_ap(row["labels"], row["scores"]) for row in rows]
    return {
        "auprc_micro": float(average_precision_score(labels, scores)),
        "auprc_macro_distribution": distribution(macro),
        "auroc_micro": float(roc_auc_score(labels, scores)),
        "positive_rate": float(labels.mean()),
    }


def pair_summary(rows):
    labels = np.concatenate([row["labels"] for row in rows])
    scores = np.concatenate([row["scores"] for row in rows])
    micro_ap = float(average_precision_score(labels, scores))
    operating_points, curve = pr_summary(labels, scores)
    return {
        "auprc_micro": micro_ap,
        "auprc_macro_distribution": distribution([row["ap"] for row in rows]),
        "positive_rate": float(labels.mean()),
        "ap_enrichment": float(micro_ap / labels.mean()),
        "topk": summarize_topk(rows),
        "precision_at_recall": operating_points,
        "pr_envelope": curve,
    }


def build_label_maps(record, shape):
    labels = np.zeros(shape, dtype=np.uint8)
    by_type = defaultdict(set)
    for pair in record["pairs"]:
        index = (int(pair["residue_index"]), int(pair["atom_index"]))
        labels[index] = 1
        by_type[str(pair["type"])].add(index)
    return labels, by_type


def perturb(graph, axis: str, rng: np.random.Generator):
    changed = graph.clone()
    if axis == "zero_atom":
        changed.x = torch.zeros_like(changed.x)
        return changed
    permutation = torch.as_tensor(
        rng.permutation(changed.x.shape[0]), dtype=torch.long, device=changed.x.device
    )
    changed.x = changed.x[permutation]
    return changed


@torch.no_grad()
def score(model, protein, ligand):
    residue_logits, pair_logits = model(protein, ligand, return_pair_scores=True)
    return (
        torch.sigmoid(residue_logits).cpu().numpy(),
        torch.sigmoid(pair_logits).cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtbind-root", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--pair-labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    payload = json.loads(args.metrics_json.read_text())
    test_ids = list(payload["split_ids"]["test"])
    if args.max_samples:
        test_ids = test_ids[: args.max_samples]
    label_payload = json.loads(args.pair_labels.read_text())
    label_records = {record["pdb_id"]: record for record in label_payload["records"]}
    interaction_types = sorted({
        str(pair["type"])
        for record in label_payload["records"]
        for pair in record["pairs"]
    })
    site = args.dtbind_root / "Data" / "site"

    model = PairwiseResidueContactPredictor(hidden_dim=args.hidden_dim).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    loaded = []
    skipped = []
    for pdb_id in test_ids:
        record = label_records.get(pdb_id)
        if record is None:
            skipped.append({"pdb_id": pdb_id, "reason": "missing_pair_labels"})
            continue
        protein, ligand = load_pair(site, pdb_id, args.device)
        if record["residue_count"] != protein.x.shape[0] or record["atom_count"] != ligand.x.shape[0]:
            skipped.append({"pdb_id": pdb_id, "reason": "shape_mismatch"})
            continue
        loaded.append((pdb_id, protein, ligand, record))

    rng = np.random.default_rng(args.seed)
    pair_rows = {name: [] for name in ("correct", "atom_feature_shuffle", "atom_features_zero", "residue_feature_shuffle")}
    residue_rows = {name: [] for name in (*pair_rows.keys(), "mismatched_ligand")}
    type_scores = defaultdict(lambda: {"labels": [], "scores": [], "masked_labels": [], "masked_scores": [], "sample_ap": []})
    sample_output = []

    for index, (pdb_id, protein, ligand, record) in enumerate(loaded):
        pair_labels, by_type = build_label_maps(record, (protein.x.shape[0], ligand.x.shape[0]))
        conditions = {
            "correct": (protein, ligand),
            "atom_feature_shuffle": (protein, perturb(ligand, "atom", rng)),
            "atom_features_zero": (protein, perturb(ligand, "zero_atom", rng)),
            "residue_feature_shuffle": (perturb(protein, "residue", rng), ligand),
        }
        correct_pair_scores = None
        detail = {"pdb_id": pdb_id, "positive_pairs": int(pair_labels.sum()), "conditions": {}}
        for name, (used_protein, used_ligand) in conditions.items():
            residue_scores, pair_scores = score(model, used_protein, used_ligand)
            flat_labels, flat_scores = pair_labels.reshape(-1), pair_scores.reshape(-1)
            ap = safe_ap(flat_labels, flat_scores)
            row = {"labels": flat_labels, "scores": flat_scores, "ap": ap, "topk": topk_metrics(flat_labels, flat_scores)}
            pair_rows[name].append(row)
            residue_labels = protein.y.reshape(-1).cpu().numpy()
            residue_rows[name].append({"labels": residue_labels, "scores": residue_scores})
            if name == "correct":
                correct_pair_scores = pair_scores
            detail["conditions"][name] = {
                "pair_ap": ap,
                "topk": row["topk"],
                "mean_abs_pair_change": None,
            }
            if name != "correct":
                detail["conditions"][name]["mean_abs_pair_change"] = float(np.abs(correct_pair_scores - pair_scores).mean())

        mismatch_ligand = loaded[(index + 1) % len(loaded)][2]
        mismatch_residue_scores, _ = score(model, protein, mismatch_ligand)
        residue_rows["mismatched_ligand"].append({
            "labels": protein.y.reshape(-1).cpu().numpy(),
            "scores": mismatch_residue_scores,
        })

        known_positive = pair_labels.astype(bool)
        for interaction_type in interaction_types:
            indices = by_type.get(interaction_type, set())
            typed = np.zeros_like(pair_labels)
            for pair_index in indices:
                typed[pair_index] = 1
            flat_typed = typed.reshape(-1)
            flat_score = correct_pair_scores.reshape(-1)
            keep = np.logical_or(flat_typed.astype(bool), ~known_positive.reshape(-1))
            type_scores[interaction_type]["labels"].append(flat_typed)
            type_scores[interaction_type]["scores"].append(flat_score)
            type_scores[interaction_type]["masked_labels"].append(flat_typed[keep])
            type_scores[interaction_type]["masked_scores"].append(flat_score[keep])
            type_scores[interaction_type]["sample_ap"].append(safe_ap(flat_typed, flat_score))
        sample_output.append(detail)

    summaries = {name: pair_summary(rows) for name, rows in pair_rows.items()}
    residue_summaries = {name: residue_summary(rows) for name, rows in residue_rows.items()}

    correct_labels = np.concatenate([row["labels"] for row in pair_rows["correct"]])
    correct_scores = np.concatenate([row["scores"] for row in pair_rows["correct"]])
    shuffled_macro = []
    shuffled_labels_parts = []
    shuffle_rng = np.random.default_rng(args.seed + 1)
    for row in pair_rows["correct"]:
        shuffled = shuffle_rng.permutation(row["labels"])
        shuffled_labels_parts.append(shuffled)
        shuffled_macro.append(safe_ap(shuffled, row["scores"]))
    shuffled_labels = np.concatenate(shuffled_labels_parts)
    shuffled_micro = float(average_precision_score(shuffled_labels, correct_scores))

    per_type = {}
    for interaction_type, values in sorted(type_scores.items()):
        labels = np.concatenate(values["labels"])
        scores = np.concatenate(values["scores"])
        masked_labels = np.concatenate(values["masked_labels"])
        masked_scores = np.concatenate(values["masked_scores"])
        per_type[interaction_type] = {
            "positive_count": int(labels.sum()),
            "positive_rate_one_vs_all": float(labels.mean()),
            "auprc_micro_one_vs_all": float(average_precision_score(labels, scores)),
            "auprc_micro_excluding_other_known_types": float(average_precision_score(masked_labels, masked_scores)),
            "sample_ap_distribution_one_vs_all": distribution(values["sample_ap"]),
        }

    for name in pair_rows:
        if name != "correct":
            paired_ap_changes = [
                changed["ap"] - correct["ap"]
                for changed, correct in zip(pair_rows[name], pair_rows["correct"])
                if changed["ap"] is not None and correct["ap"] is not None
            ]
            map_changes = [
                sample["conditions"][name]["mean_abs_pair_change"]
                for sample in sample_output
            ]
            summaries[name]["relative_micro_ap_change_vs_correct"] = float(
                (summaries[name]["auprc_micro"] - summaries["correct"]["auprc_micro"])
                / summaries["correct"]["auprc_micro"]
            )
            summaries[name]["paired_sample_ap_change_distribution"] = distribution(paired_ap_changes)
            summaries[name]["fraction_samples_ap_decreased"] = float(np.mean(np.asarray(paired_ap_changes) < 0))
            summaries[name]["mean_abs_pair_map_change_distribution"] = distribution(map_changes)

    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "metrics_json": str(args.metrics_json),
        "requested_test_count": len(test_ids),
        "evaluated_count": len(loaded),
        "skipped": skipped,
        "pair_conditions": summaries,
        "residue_conditions": residue_summaries,
        "label_permutation_baseline": {
            "auprc_micro": shuffled_micro,
            "auprc_macro_distribution": distribution(shuffled_macro),
            "positive_rate": float(correct_labels.mean()),
        },
        "per_interaction_type": per_type,
        "notes": [
            "Mismatched ligand is residue-only because atom indices do not correspond across ligands.",
            "Atom/residue feature permutations preserve matrix shape but intentionally break feature-to-label alignment.",
            "Per-type one-vs-all uses the current binary pair score; the masked metric excludes contacts annotated only as another type.",
            "Per-type micro metrics include every evaluated complex; per-complex AP distributions omit complexes with no positive of that type.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostics_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "per_complex.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample_output:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({
        "evaluated_count": len(loaded),
        "pair_conditions": {name: {
            "micro": value["auprc_micro"],
            "macro_median": value["auprc_macro_distribution"]["median"],
            "relative_change": value.get("relative_micro_ap_change_vs_correct"),
        } for name, value in summaries.items()},
        "residue_conditions": {name: value["auprc_micro"] for name, value in residue_summaries.items()},
        "label_permutation_micro": shuffled_micro,
        "per_type_micro": {name: value["auprc_micro_one_vs_all"] for name, value in per_type.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
