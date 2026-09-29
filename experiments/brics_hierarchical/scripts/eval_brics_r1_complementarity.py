#!/usr/bin/env python3
"""Fold-1 audit of BRICS topology and the fixed R1 residual transfer.

This script is evaluation-only. It imports the original R1 implementation,
loads existing BRICS checkpoints, and never reads test indices or labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


EXPECTED = {
    "p13d": 0.489465223315432,
    "r1": 0.461357564461127,
    "brics": {42: 0.4746934473514557, 43: 0.48082873225212097, 44: 0.4828588664531708},
    "noedge": 0.4893468916416168,
}


def heading(name: str) -> None:
    print(f"\n[{name}]", flush=True)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).ravel()
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max_abs": float(np.abs(values).max()),
    }


def residual_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).ravel()
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "mae": float(np.abs(values).mean()),
        "mse": float(np.square(values).mean()),
    }


def parse_seed_checkpoint(items: list[str]) -> dict[int, Path]:
    output = {}
    for item in items:
        seed, path = item.split("=", 1)
        output[int(seed)] = Path(path)
    if set(output) != {42, 43, 44}:
        raise ValueError("--brics-checkpoint must specify exactly seeds 42, 43, and 44")
    return output


def set_deterministic(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_grid(global_data, similarity_data, split):
    drug_ids = [str(x) for x in similarity_data["drug_ids"].tolist()]
    protein_ids = [str(x) for x in similarity_data["protein_ids"].tolist()]
    drug_lookup = {x: i for i, x in enumerate(drug_ids)}
    protein_lookup = {x: i for i, x in enumerate(protein_ids)}
    n_drugs, n_proteins = len(drug_ids), len(protein_ids)
    row_index = np.full((n_drugs, n_proteins), -1, dtype=np.int64)
    row_drugs = [str(x) for x in global_data["drug_id"]]
    row_proteins = [str(x) for x in global_data["protein_id"]]
    for row, (drug, protein) in enumerate(zip(row_drugs, row_proteins)):
        row_index[drug_lookup[drug], protein_lookup[protein]] = row
    if np.any(row_index < 0):
        raise RuntimeError("Davis drug/protein grid is incomplete")

    labels = global_data["label"].numpy()[row_index].astype(np.float64)
    p13d = global_data["prediction"].numpy()[row_index].astype(np.float64)
    train_drugs = np.asarray([drug_lookup[str(x)] for x in split["train_drugs"]], dtype=int)
    val_drugs = np.asarray([drug_lookup[str(x)] for x in split["val_drugs"]], dtype=int)
    train_rows_expected = np.sort(row_index[train_drugs].ravel())
    val_rows_expected = np.sort(row_index[val_drugs].ravel())
    train_rows_actual = np.sort(np.asarray(split["train_indices"], dtype=int))
    val_rows_actual = np.sort(np.asarray(split["val_indices"], dtype=int))
    if not np.array_equal(train_rows_expected, train_rows_actual):
        raise RuntimeError("Fold1 train_indices do not equal the train-drug pair grid")
    if not np.array_equal(val_rows_expected, val_rows_actual):
        raise RuntimeError("Fold1 val_indices do not equal the validation-drug pair grid")
    return {
        "drug_ids": drug_ids,
        "protein_ids": protein_ids,
        "row_index": row_index,
        "labels": labels,
        "p13d": p13d,
        "train_drugs": train_drugs,
        "val_drugs": val_drugs,
    }


def load_brics_predictions(graph, global_data, drug_data, rows, condition, seed, checkpoint,
                           project, p13d_checkpoint, device, batch_size):
    set_deterministic(seed)
    store = graph.ChemStore(global_data, drug_data, condition, seed)
    loader = DataLoader(
        graph.base.IndexDataset(rows), batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=store.collate,
    )
    model = graph.BRICSChemGraphP13D(project, p13d_checkpoint).to(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["hierarchy_state"]
    incompat = model.load_state_dict(state, strict=False)
    unexpected = list(incompat.unexpected_keys)
    invalid_missing = [
        key for key in incompat.missing_keys
        if not (key.startswith("drug_fusion.") or key.startswith("decoder."))
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            f"Checkpoint incompatibility for {checkpoint}: unexpected={unexpected}, "
            f"invalid_missing={invalid_missing}"
        )
    result = graph.base.evaluate(model, loader, device)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "indices": result["index"].numpy().astype(int),
        "prediction": result["prediction"].numpy().astype(np.float64),
        "label": result["label"].numpy().astype(np.float64),
        "checkpoint_result": saved.get("result", {}),
        "missing_frozen_keys": list(incompat.missing_keys),
    }


def infer_to_grid(graph, global_data, drug_data, grid, condition, seed, checkpoint,
                  project, p13d_checkpoint, device, batch_size):
    rows = np.concatenate([
        grid["row_index"][grid["train_drugs"]].ravel(),
        grid["row_index"][grid["val_drugs"]].ravel(),
    ]).astype(int)
    inference = load_brics_predictions(
        graph, global_data, drug_data, rows, condition, seed, checkpoint,
        project, p13d_checkpoint, device, batch_size,
    )
    full = np.full(len(global_data["label"]), np.nan, dtype=np.float64)
    full[inference["indices"]] = inference["prediction"]
    prediction_grid = full[grid["row_index"]]
    if np.isnan(prediction_grid[grid["train_drugs"]]).any() or np.isnan(prediction_grid[grid["val_drugs"]]).any():
        raise RuntimeError("BRICS inference did not cover all Fold1 train/validation pairs")
    inference["grid"] = prediction_grid
    return inference


def apply_r1(r1_module, query_drugs, train_drugs, residual_grid, baseline_grid,
             drug_similarity, config):
    prediction, correction, alpha = r1_module.outer_predictions(
        query_drugs, train_drugs, residual_grid, baseline_grid,
        drug_similarity, config, r2_best=None,
    )
    return prediction, correction, alpha


def neighbor_audit(r1_module, query_drugs, train_drugs, similarity, config):
    selected_counts, effective_counts, support_sums = [], [], []
    for query in query_drugs:
        _, weights = r1_module.top_weights(
            similarity[query, train_drugs], config["gamma"], config["k_drug"],
            config["min_drug_similarity"],
        )
        selected_counts.append(len(weights))
        effective_counts.append(int(np.sum(weights > 0)))
        support_sums.append(float(weights.sum()))
    return {
        "selected_neighbor_count": describe(np.asarray(selected_counts)),
        "positive_weight_neighbor_count": describe(np.asarray(effective_counts)),
        "weight_sum": describe(np.asarray(support_sums)),
    }


def neighborhood_residual_variance(r1_module, train_drugs, residual_grid, similarity, config):
    variances, supports = [], []
    train_set = set(train_drugs.tolist())
    for query in train_drugs:
        references = np.asarray(sorted(train_set - {int(query)}), dtype=int)
        _, variance, support = r1_module.drug_kernel_stats(
            np.asarray([query]), references, residual_grid[references], similarity, config,
        )
        if support[0] > 0:
            variances.append(variance[0])
            supports.append(support[0])
    values = np.concatenate(variances) if variances else np.zeros(1)
    return {
        "definition": "leave-self-out among outer-train drugs; same target column; fixed R1 neighbors/kernel",
        "weighted_local_variance": describe(values),
        "supported_train_drugs": len(variances),
        "support": describe(np.asarray(supports) if supports else np.zeros(1)),
    }


def correlation(x, y, rankdata) -> float:
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def complementarity(label, p13d, brics, r1, rankdata):
    err_p = np.abs(label - p13d)
    err_b = np.abs(label - brics)
    err_r = np.abs(label - r1)
    gain_b = err_p - err_b
    gain_r = err_p - err_r
    drug_gain_b = gain_b.mean(axis=1)
    drug_gain_r = gain_r.mean(axis=1)
    q1 = (gain_b > 0) & (gain_r > 0)
    q2 = (gain_b > 0) & (gain_r <= 0)
    q3 = (gain_b <= 0) & (gain_r > 0)
    q4 = (gain_b <= 0) & (gain_r <= 0)
    return {
        "pair_level": {
            "pearson": correlation(gain_b.ravel(), gain_r.ravel(), rankdata),
            "spearman": correlation(rankdata(gain_b.ravel()), rankdata(gain_r.ravel()), rankdata),
        },
        "drug_level_mean_gain": {
            "pearson": correlation(drug_gain_b, drug_gain_r, rankdata),
            "spearman": correlation(rankdata(drug_gain_b), rankdata(drug_gain_r), rankdata),
            "brics": drug_gain_b.tolist(),
            "r1": drug_gain_r.tolist(),
        },
        "quadrants": {
            "Q1_both_improve": {"count": int(q1.sum()), "fraction": float(q1.mean())},
            "Q2_only_brics": {"count": int(q2.sum()), "fraction": float(q2.mean())},
            "Q3_only_r1": {"count": int(q3.sum()), "fraction": float(q3.mean())},
            "Q4_both_fail": {"count": int(q4.sum()), "fraction": float(q4.mean())},
        },
    }, gain_b, gain_r, np.select(
        [q1, q2, q3, q4], ["Q1", "Q2", "Q3", "Q4"], default="UNASSIGNED"
    )


def clustered_bootstrap(label, reference, candidate, drug_ids, n=10000, seed=42):
    label = np.asarray(label)
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    per_drug_delta = np.mean((candidate - label) ** 2 - (reference - label) ** 2, axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_drug_delta), size=(n, len(per_drug_delta)))
    samples = per_drug_delta[draws].mean(axis=1)
    return {
        "definition": "candidate MSE - reference MSE; negative favors candidate",
        "cluster_unit": "held-out drug with all protein pairs",
        "replicates": n,
        "seed": seed,
        "drug_ids": list(drug_ids),
        "per_drug_delta_mse": per_drug_delta.tolist(),
        "mean_delta_mse": float(per_drug_delta.mean()),
        "bootstrap_95_ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "probability_candidate_better": float(np.mean(samples < 0)),
    }


def per_drug_table(drug_ids, label, predictions):
    rows = []
    for i, drug_id in enumerate(drug_ids):
        mse = {name: float(np.mean((pred[i] - label[i]) ** 2)) for name, pred in predictions.items()}
        rows.append({
            "drug_id": drug_id,
            "n_pairs": int(label.shape[1]),
            "p13d_mse": mse["p13d"],
            "brics_mse": mse["brics"],
            "r1_mse": mse["r1"],
            "brics_r1_mse": mse["brics_r1"],
            "noedge_mse": mse["noedge"],
            "noedge_r1_mse": mse["noedge_r1"],
            "gain_brics_vs_p13d": mse["p13d"] - mse["brics"],
            "gain_r1_vs_p13d": mse["p13d"] - mse["r1"],
            "gain_combo_vs_r1": mse["r1"] - mse["brics_r1"],
            "gain_combo_vs_brics": mse["brics"] - mse["brics_r1"],
            "gain_combo_vs_noedge_r1": mse["noedge_r1"] - mse["brics_r1"],
        })
    return pd.DataFrame(rows)


def prediction_table(grid, label, p13d, brics, r1, combo, noedge, noedge_r1,
                     delta_brics, delta_r1, delta_after, gain_b, gain_r, quadrants):
    val_ids = [grid["drug_ids"][i] for i in grid["val_drugs"]]
    n_proteins = len(grid["protein_ids"])
    frame = pd.DataFrame({
        "drug_id": np.repeat(val_ids, n_proteins),
        "protein_id": np.tile(grid["protein_ids"], len(val_ids)),
        "label": label.ravel(),
        "p13d_pred": p13d.ravel(),
        "brics_pred": brics.ravel(),
        "r1_pred": r1.ravel(),
        "brics_r1_pred": combo.ravel(),
        "noedge_pred": noedge.ravel(),
        "noedge_r1_pred": noedge_r1.ravel(),
    })
    for name, pred in {
        "p13d_error": p13d, "brics_error": brics, "r1_error": r1,
        "brics_r1_error": combo, "noedge_error": noedge, "noedge_r1_error": noedge_r1,
    }.items():
        frame[name] = np.abs(label - pred).ravel()
    frame["delta_brics"] = delta_brics.ravel()
    frame["delta_r1"] = delta_r1.ravel()
    frame["delta_r1_after_brics"] = delta_after.ravel()
    frame["gain_brics_vs_p13d"] = gain_b.ravel()
    frame["gain_r1_vs_p13d"] = gain_r.ravel()
    frame["gain_combo_vs_r1"] = (np.abs(label - r1) - np.abs(label - combo)).ravel()
    frame["quadrant"] = quadrants.ravel()
    return frame


def make_report(summary, seed_results, noedge_metrics, output: Path):
    lines = [
        "# BRICS–R1 Complementarity Report (Fold1)", "",
        "## 1. Experiment objective", "",
        "Test whether intrinsic BRICS fragment topology adds stable information beyond the fixed same-target R1 residual transfer.", "",
        "## 2. Fixed configurations", "",
        "- Dataset/split: Davis drug-cold Fold1; outer test data were not accessed.",
        "- R1: original implementation and saved Fold1 configuration; no parameter search in N1.",
        "- Residual mode: in-sample outer-train residual bank.",
        "- BRICS: existing best checkpoints only; no retraining.", "",
        "```json", json.dumps(summary["r1_config_audit"], indent=2), "```", "",
        "## 3. Reproduction audit", "",
        f"- P13D MSE: {summary['reproduction']['p13d']['observed']:.9f}",
        f"- R1 MSE: {summary['reproduction']['r1']['observed']:.9f}",
    ]
    for seed in (42, 43, 44):
        item = summary["reproduction"][f"brics_seed{seed}"]
        lines.append(f"- BRICS seed{seed} MSE: {item['observed']:.9f}")
    lines.extend(["", "## 4. Fold1 results", "", "| Model | MSE | RMSE | MAE | CI | Rm2 |", "|---|---:|---:|---:|---:|---:|"])
    base = seed_results[42]["metrics"]
    for key, label in (("p13d", "P13D"), ("r1", "R1")):
        m = base[key]
        lines.append(f"| {label} | {m['mse']:.6f} | {m['rmse']:.6f} | {m['mae']:.6f} | {m['ci']:.6f} | {m['rm2']:.6f} |")
    for seed in (42, 43, 44):
        for key, label in (("brics", f"BRICS s{seed}"), ("brics_r1", f"BRICS+R1 s{seed}")):
            m = seed_results[seed]["metrics"][key]
            lines.append(f"| {label} | {m['mse']:.6f} | {m['rmse']:.6f} | {m['mae']:.6f} | {m['ci']:.6f} | {m['rm2']:.6f} |")
    for key, label in (("noedge", "NoEdge"), ("noedge_r1", "NoEdge+R1")):
        m = noedge_metrics[key]
        lines.append(f"| {label} | {m['mse']:.6f} | {m['rmse']:.6f} | {m['mae']:.6f} | {m['ci']:.6f} | {m['rm2']:.6f} |")
    lines.extend(["", "## 5. Per-seed results", "", "| Seed | BRICS MSE | BRICS+R1 MSE | R1−combo | Drugs combo<R1 |", "|---:|---:|---:|---:|---:|"])
    for seed in (42, 43, 44):
        x = seed_results[seed]
        lines.append(f"| {seed} | {x['metrics']['brics']['mse']:.6f} | {x['metrics']['brics_r1']['mse']:.6f} | {x['gain_vs_r1']:.6f} | {x['drugs_combo_better_r1']}/7 |")
    lines.extend([
        "", "## 6. Per-drug results", "",
        "Complete values are in each seed's `per_drug_metrics.csv`; no pair-independence assumption was used.", "",
        "## 7. Error complementarity", "",
    ])
    for seed in (42, 43, 44):
        c = seed_results[seed]["complementarity"]["pair_level"]
        lines.append(f"- Seed {seed}: pair-level Pearson={c['pearson']:.4f}, Spearman={c['spearman']:.4f}.")
    lines.extend(["", "## 8. Residual smoothness diagnostics", "", "See each seed's `residual_diagnostics.json`; diagnostics were not used for tuning.", "", "## 9. No-edge causal control", "", f"NoEdge+R1 MSE: {noedge_metrics['noedge_r1']['mse']:.6f}.", "", "## 10. Drug-cluster bootstrap", ""])
    for seed in (42, 43, 44):
        b = seed_results[seed]["bootstrap"]["combo_vs_r1"]
        lines.append(f"- Seed {seed}: ΔMSE={b['mean_delta_mse']:.6f}, 95% CI={b['bootstrap_95_ci']}, P(combo<R1)={b['probability_candidate_better']:.4f}.")
    lines.extend(["", "## 11. Go/No-Go decision", "", f"**{summary['decision']['classification']}**", "", summary["decision"]["statement"], "", "## 12. Scientific interpretation", "", summary["decision"]["interpretation"], "", "Historical-design limitation: the reused BRICS checkpoints were selected by validation early stopping in their original experiment. N1 performs no new tuning or checkpoint selection, but Fold1 remains a pilot rather than an untouched confirmatory estimate.", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--drug-cache", type=Path, required=True)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--p13d-checkpoint", type=Path, required=True)
    parser.add_argument("--r1-results", type=Path, required=True)
    parser.add_argument("--brics-checkpoint", action="append", required=True, help="SEED=/path/best.pt")
    parser.add_argument("--noedge-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--residual-mode", choices=["in_sample", "cross_fitted"], default="in_sample")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.residual_mode != "in_sample":
        raise NotImplementedError("N1 is fixed to --residual-mode in_sample; cross_fitted is reserved for future work")

    scripts = args.project / "experiments/brics_hierarchical/scripts"
    r1_dir = args.project / "experiments/klifs85_interaction"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(r1_dir))
    import train_brics_graph_stage1 as graph
    import run_residual_kernel as r1_module
    from train_klifs_interact import metrics, rankdata

    checkpoints = parse_seed_checkpoint(args.brics_checkpoint)
    for path in [args.global_cache, args.drug_cache, args.similarity, args.split,
                 args.p13d_checkpoint, args.r1_results, args.noedge_checkpoint, *checkpoints.values()]:
        if not path.exists():
            raise FileNotFoundError(path)

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    drug_data = torch.load(args.drug_cache, map_location="cpu", weights_only=False)
    similarity_data = np.load(args.similarity, allow_pickle=True)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    grid = build_grid(global_data, similarity_data, split)
    drug_similarity = similarity_data["drug_similarity"].astype(np.float64)
    r1_saved = json.loads(args.r1_results.read_text(encoding="utf-8"))
    r1_config = r1_saved["inner"]["R1_best"]
    r1_fixed = {key: r1_config[key] for key in (
        "gamma", "k_drug", "min_drug_similarity", "tau", "beta", "clip", "scale"
    )}
    r1_audit = {
        "implementation": str(r1_dir / "run_residual_kernel.py"),
        "saved_config_source": str(args.r1_results),
        "fingerprint": "RDKit Morgan radius=2, fpSize=2048",
        "similarity": "Tanimoto",
        "kernel": "top-k selected similarities raised to gamma, normalized weighted residual mean",
        "similarity_threshold": r1_fixed["min_drug_similarity"],
        "gamma": r1_fixed["gamma"],
        "k_drug": r1_fixed["k_drug"],
        "bandwidth_temperature": "none",
        "gate": {k: r1_fixed[k] for k in ("tau", "beta", "clip", "scale")},
        "fallback": "zero correction (base prediction unchanged) when no positive total weight",
        "residual": "label - base prediction",
        "residual_bank": "outer Fold1 train pairs only; in-sample model predictions",
        "target_rule": "same exact protein column only; no protein similarity",
        "normalization": "kernel weights normalized by their sum; no label/residual standardization",
    }
    heading("R1 CONFIG AUDIT")
    print(json.dumps(r1_audit, indent=2), flush=True)

    train_set = set(grid["train_drugs"].tolist())
    val_set = set(grid["val_drugs"].tolist())
    leakage = {
        "train_val_drug_intersection": len(train_set & val_set),
        "train_drugs": len(train_set),
        "validation_drugs": len(val_set),
        "train_pairs": int(len(grid["train_drugs"]) * len(grid["protein_ids"])),
        "validation_pairs": int(len(grid["val_drugs"]) * len(grid["protein_ids"])),
        "validation_drugs_in_residual_bank": len(val_set & train_set),
        "validation_labels_used_in_residual_bank": False,
        "same_target_enforcement": "residual transfer is column-wise on identical protein_id",
    }
    if leakage["train_val_drug_intersection"] != 0 or leakage["validation_pairs"] != 3094:
        raise RuntimeError(f"Split/leakage audit failed: {leakage}")
    heading("DATA LEAKAGE AUDIT")
    print(json.dumps(leakage, indent=2), flush=True)

    if args.sanity_only:
        smoke_rows = list(split["train_indices"][:8]) + list(split["val_indices"][:8])
        checks = {}
        for seed, checkpoint in checkpoints.items():
            x = load_brics_predictions(
                graph, global_data, drug_data, smoke_rows, "real", seed, checkpoint,
                args.project, args.p13d_checkpoint, args.device, min(args.batch_size, 16),
            )
            checks[f"seed{seed}"] = {"checkpoint": str(checkpoint), "shape": list(x["prediction"].shape), "finite": bool(np.isfinite(x["prediction"]).all())}
        x = load_brics_predictions(
            graph, global_data, drug_data, smoke_rows, "no_fragment_edges", 42,
            args.noedge_checkpoint, args.project, args.p13d_checkpoint, args.device,
            min(args.batch_size, 16),
        )
        checks["noedge"] = {"checkpoint": str(args.noedge_checkpoint), "shape": list(x["prediction"].shape), "finite": bool(np.isfinite(x["prediction"]).all())}
        checks["residual_bank_shape"] = [len(grid["train_drugs"]), len(grid["protein_ids"])]
        checks["neighbor_audit"] = neighbor_audit(r1_module, grid["val_drugs"], grid["train_drugs"], drug_similarity, r1_fixed)
        checks["output_path"] = str(args.output_dir)
        heading("SANITY ONLY")
        print(json.dumps(checks, indent=2), flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_val = grid["labels"][grid["val_drugs"]]
    p13d_val = grid["p13d"][grid["val_drugs"]]
    p13d_metrics = metrics(label_val.ravel(), p13d_val.ravel())
    reproduction = {"p13d": {"expected": EXPECTED["p13d"], "observed": p13d_metrics["mse"], "abs_diff": abs(p13d_metrics["mse"] - EXPECTED["p13d"])}}
    heading("GLOBAL/P13D REPRODUCTION")
    print(json.dumps(reproduction["p13d"], indent=2), flush=True)
    if reproduction["p13d"]["abs_diff"] >= args.tolerance:
        raise RuntimeError("P13D reproduction failure; STOP before R1")

    p13d_residual = grid["labels"] - grid["p13d"]
    r1_val, r1_delta, r1_alpha = apply_r1(
        r1_module, grid["val_drugs"], grid["train_drugs"], p13d_residual,
        grid["p13d"], drug_similarity, r1_fixed,
    )
    r1_metrics = metrics(label_val.ravel(), r1_val.ravel())
    saved_r1 = float(r1_saved["outer_validation"]["R1"]["mse"])
    reproduction["r1"] = {"expected": saved_r1, "observed": r1_metrics["mse"], "abs_diff": abs(r1_metrics["mse"] - saved_r1)}
    heading("R1 REPRODUCTION")
    print(json.dumps(reproduction["r1"], indent=2), flush=True)
    if reproduction["r1"]["abs_diff"] >= args.tolerance:
        raise RuntimeError("R1 reproduction failure; STOP before BRICS+R1")

    neighbor_info = neighbor_audit(
        r1_module, grid["val_drugs"], grid["train_drugs"], drug_similarity, r1_fixed
    )
    heading("RESIDUAL BANK AUDIT")
    print(json.dumps({"shape": [len(grid["train_drugs"]), len(grid["protein_ids"])], "neighbor": neighbor_info, **leakage}, indent=2), flush=True)

    noedge = infer_to_grid(
        graph, global_data, drug_data, grid, "no_fragment_edges", 42,
        args.noedge_checkpoint, args.project, args.p13d_checkpoint, args.device, args.batch_size,
    )
    noedge_grid = noedge["grid"]
    noedge_val = noedge_grid[grid["val_drugs"]]
    noedge_metric = metrics(label_val.ravel(), noedge_val.ravel())
    reproduction["noedge_seed42"] = {"expected": EXPECTED["noedge"], "observed": noedge_metric["mse"], "abs_diff": abs(noedge_metric["mse"] - EXPECTED["noedge"])}
    if reproduction["noedge_seed42"]["abs_diff"] >= args.tolerance:
        raise RuntimeError("No-edge reproduction failure; STOP")
    noedge_residual = grid["labels"] - noedge_grid
    noedge_r1_val, noedge_r1_delta, noedge_r1_alpha = apply_r1(
        r1_module, grid["val_drugs"], grid["train_drugs"], noedge_residual,
        noedge_grid, drug_similarity, r1_fixed,
    )
    noedge_metrics = {
        "noedge": noedge_metric,
        "noedge_r1": metrics(label_val.ravel(), noedge_r1_val.ravel()),
    }
    heading("NO-EDGE CAUSAL CONTROL")
    print(json.dumps(noedge_metrics, indent=2), flush=True)

    val_drug_ids = [grid["drug_ids"][x] for x in grid["val_drugs"]]
    seed_results = {}
    for seed in (42, 43, 44):
        inference = infer_to_grid(
            graph, global_data, drug_data, grid, "real", seed, checkpoints[seed],
            args.project, args.p13d_checkpoint, args.device, args.batch_size,
        )
        brics_grid = inference["grid"]
        brics_val = brics_grid[grid["val_drugs"]]
        brics_metrics = metrics(label_val.ravel(), brics_val.ravel())
        target = EXPECTED["brics"][seed]
        reproduction[f"brics_seed{seed}"] = {"expected": target, "observed": brics_metrics["mse"], "abs_diff": abs(brics_metrics["mse"] - target)}
        heading("BRICS REPRODUCTION")
        print(json.dumps({"seed": seed, **reproduction[f"brics_seed{seed}"]}, indent=2), flush=True)
        if reproduction[f"brics_seed{seed}"]["abs_diff"] >= args.tolerance:
            raise RuntimeError(f"BRICS seed{seed} reproduction failure; STOP before its R1 combination")

        brics_residual = grid["labels"] - brics_grid
        combo_val, combo_delta, combo_alpha = apply_r1(
            r1_module, grid["val_drugs"], grid["train_drugs"], brics_residual,
            brics_grid, drug_similarity, r1_fixed,
        )
        combo_metrics = metrics(label_val.ravel(), combo_val.ravel())
        comp, gain_b, gain_r, quadrants = complementarity(
            label_val, p13d_val, brics_val, r1_val, rankdata
        )
        delta_brics = brics_val - p13d_val
        residual_diag = {
            "p13d_training_residual": residual_stats(p13d_residual[grid["train_drugs"]]),
            "brics_training_residual": residual_stats(brics_residual[grid["train_drugs"]]),
            "p13d_neighborhood": neighborhood_residual_variance(r1_module, grid["train_drugs"], p13d_residual, drug_similarity, r1_fixed),
            "brics_neighborhood": neighborhood_residual_variance(r1_module, grid["train_drugs"], brics_residual, drug_similarity, r1_fixed),
            "corrections": {
                "delta_brics": describe(delta_brics),
                "delta_r1": describe(r1_delta),
                "delta_r1_after_brics": describe(combo_delta),
            },
            "alpha": {"r1": describe(r1_alpha), "r1_after_brics": describe(combo_alpha)},
        }
        preds = {
            "p13d": p13d_val, "brics": brics_val, "r1": r1_val,
            "brics_r1": combo_val, "noedge": noedge_val, "noedge_r1": noedge_r1_val,
        }
        per_drug = per_drug_table(val_drug_ids, label_val, preds)
        boot = {
            "combo_vs_r1": clustered_bootstrap(label_val, r1_val, combo_val, val_drug_ids),
            "combo_vs_brics": clustered_bootstrap(label_val, brics_val, combo_val, val_drug_ids),
            "combo_vs_noedge_r1": clustered_bootstrap(label_val, noedge_r1_val, combo_val, val_drug_ids),
        }
        pair_frame = prediction_table(
            grid, label_val, p13d_val, brics_val, r1_val, combo_val,
            noedge_val, noedge_r1_val, delta_brics, r1_delta, combo_delta,
            gain_b, gain_r, quadrants,
        )
        seed_dir = args.output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        pair_frame.to_csv(seed_dir / "predictions.csv", index=False)
        pair_frame[["drug_id", "protein_id", "label", "gain_brics_vs_p13d", "gain_r1_vs_p13d", "quadrant"]].to_csv(seed_dir / "complementarity_pairs.csv", index=False)
        per_drug.to_csv(seed_dir / "per_drug_metrics.csv", index=False)
        seed_metrics = {"p13d": p13d_metrics, "brics": brics_metrics, "r1": r1_metrics, "brics_r1": combo_metrics, **noedge_metrics}
        write_json(seed_dir / "metrics.json", seed_metrics)
        write_json(seed_dir / "residual_diagnostics.json", residual_diag)
        write_json(seed_dir / "complementarity_analysis.json", comp)
        write_json(args.output_dir / "bootstrap" / f"seed{seed}.json", boot)
        improved = int((per_drug["gain_combo_vs_r1"] > 0).sum())
        worsened = int((per_drug["gain_combo_vs_r1"] < 0).sum())
        seed_results[seed] = {
            "metrics": seed_metrics,
            "gain_vs_r1": r1_metrics["mse"] - combo_metrics["mse"],
            "drugs_combo_better_r1": improved,
            "drugs_combo_worse_r1": worsened,
            "complementarity": comp,
            "residual_diagnostics": residual_diag,
            "bootstrap": boot,
            "checkpoint": str(checkpoints[seed]),
        }
        heading("COMPLEMENTARITY METRICS")
        print(json.dumps({"seed": seed, "metrics": seed_metrics, "gain_vs_r1": seed_results[seed]["gain_vs_r1"], "correlation": comp}, indent=2), flush=True)
        heading("PER-DRUG RESULTS")
        print(per_drug.to_string(index=False), flush=True)
        heading("CLUSTER BOOTSTRAP")
        print(json.dumps({"seed": seed, **boot}, indent=2), flush=True)

    noedge_dir = args.output_dir / "noedge"
    noedge_dir.mkdir(parents=True, exist_ok=True)
    noedge_frame = pd.DataFrame({
        "drug_id": np.repeat(val_drug_ids, len(grid["protein_ids"])),
        "protein_id": np.tile(grid["protein_ids"], len(val_drug_ids)),
        "label": label_val.ravel(),
        "p13d_pred": p13d_val.ravel(),
        "noedge_pred": noedge_val.ravel(),
        "noedge_r1_pred": noedge_r1_val.ravel(),
    })
    noedge_frame.to_csv(noedge_dir / "predictions.csv", index=False)
    write_json(noedge_dir / "metrics.json", noedge_metrics)
    write_json(noedge_dir / "residual_diagnostics.json", {
        "noedge_training_residual": residual_stats(noedge_residual[grid["train_drugs"]]),
        "noedge_neighborhood": neighborhood_residual_variance(r1_module, grid["train_drugs"], noedge_residual, drug_similarity, r1_fixed),
        "delta_noedge": describe(noedge_val - p13d_val),
        "delta_r1_after_noedge": describe(noedge_r1_delta),
        "alpha_after_noedge": describe(noedge_r1_alpha),
    })

    combo_mses = np.asarray([seed_results[s]["metrics"]["brics_r1"]["mse"] for s in (42, 43, 44)])
    better_count = int(np.sum(combo_mses < r1_metrics["mse"]))
    majority_count = int(sum(seed_results[s]["drugs_combo_better_r1"] >= 4 for s in (42, 43, 44)))
    topology_count = int(np.sum(combo_mses < noedge_metrics["noedge_r1"]["mse"]))
    mean_combo = float(combo_mses.mean())
    mean_gain = float(r1_metrics["mse"] - mean_combo)
    criteria = {
        "A_at_least_2_of_3_combo_better_than_r1": better_count >= 2,
        "B_mean_combo_better_than_r1": mean_combo < r1_metrics["mse"],
        "C_mean_absolute_mse_gain_at_least_0p003": mean_gain >= 0.003,
        "D_majority_drugs_in_at_least_2_of_3_seeds": majority_count >= 2,
        "E_topology_combo_better_than_noedge_combo_in_at_least_2_of_3": topology_count >= 2,
    }
    full_go = all(criteria.values())
    if full_go:
        classification = "SUPPORTED"
        statement = "BRICS topology and R1 residual transfer are complementary. RECOMMEND 5-FOLD VALIDATION."
        interpretation = "The fixed combination improves on R1 across seeds and held-out drugs, and also beats the no-edge combination."
    elif better_count == 0 or mean_combo >= r1_metrics["mse"] or topology_count < 2:
        classification = "NOT SUPPORTED"
        statement = "BRICS does not provide stable information beyond R1."
        interpretation = "The fixed residual transfer does not retain a stable topology-specific increment beyond the R1 baseline."
    else:
        classification = "INCONCLUSIVE"
        statement = "Weak/seed-dependent result; do not proceed to five folds."
        interpretation = "Some complementarity is visible, but it fails at least one prespecified stability or effect-size criterion."
    decision = {
        "classification": classification,
        "statement": statement,
        "interpretation": interpretation,
        "criteria": criteria,
        "seeds_combo_better_r1": better_count,
        "seeds_majority_drugs_better": majority_count,
        "seeds_combo_better_noedge_combo": topology_count,
        "r1_mse": r1_metrics["mse"],
        "mean_combo_mse": mean_combo,
        "mean_gain_vs_r1": mean_gain,
    }
    summary = {
        "guardrail": "Fold1 train/validation only; test indices and metrics were not accessed",
        "experiment": "N1 BRICS topology + same-target R1 residual transfer complementarity",
        "residual_mode": args.residual_mode,
        "r1_config_audit": r1_audit,
        "data_leakage_audit": leakage,
        "neighbor_audit": neighbor_info,
        "reproduction": reproduction,
        "seed_results": {str(k): v for k, v in seed_results.items()},
        "noedge": noedge_metrics,
        "decision": decision,
        "paths": {
            "p13d_checkpoint": str(args.p13d_checkpoint),
            "brics_checkpoints": {str(k): str(v) for k, v in checkpoints.items()},
            "noedge_checkpoint": str(args.noedge_checkpoint),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    make_report(summary, seed_results, noedge_metrics, args.output_dir / "BRICS_R1_COMPLEMENTARITY_REPORT.md")
    heading("GO/NO-GO DECISION")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
