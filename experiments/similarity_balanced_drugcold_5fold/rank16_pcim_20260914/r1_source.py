#!/usr/bin/env python3
"""Hierarchical residual-kernel correction for Davis drug-cold prediction.

Hyperparameters are selected only by 5-fold drug-cold CV inside the 47 outer
training drugs.  The seven outer validation drugs are evaluated once after
selection.  Held-out test indices are never materialized or scored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_klifs_interact import metrics


def top_weights(similarities, gamma, k, minimum):
    similarities = np.asarray(similarities, dtype=np.float64)
    order = np.argsort(similarities)[::-1][: min(k, len(similarities))]
    selected = similarities[order]
    keep = selected >= minimum
    order, selected = order[keep], selected[keep]
    weights = selected**gamma
    return order, weights


def drug_kernel_stats(query_drugs, reference_drugs, residual_reference, drug_similarity, config):
    n_query, n_proteins = len(query_drugs), residual_reference.shape[1]
    mean = np.zeros((n_query, n_proteins), dtype=np.float64)
    variance = np.zeros_like(mean)
    support = np.zeros(n_query, dtype=np.float64)
    for row, query in enumerate(query_drugs):
        local, weights = top_weights(
            drug_similarity[query, reference_drugs], config["gamma"], config["k_drug"], config["min_drug_similarity"]
        )
        if not len(local) or weights.sum() <= 1e-12:
            continue
        values = residual_reference[local]
        total = weights.sum()
        mean[row] = (weights[:, None] * values).sum(0) / total
        variance[row] = (weights[:, None] * (values - mean[row]) ** 2).sum(0) / total
        support[row] = total / config["k_drug"]
    return mean, variance, support


def apply_correction(baseline, raw_residual, variance, support, gate_config, disagreement=None):
    effective_variance = variance.copy()
    if disagreement is not None:
        effective_variance = effective_variance + disagreement
    alpha = (
        gate_config["scale"]
        * support[:, None] / (support[:, None] + gate_config["tau"])
        * np.exp(-gate_config["beta"] * effective_variance)
    )
    correction = alpha * np.clip(raw_residual, -gate_config["clip"], gate_config["clip"])
    return baseline + correction, correction, alpha


def protein_smoother(protein_similarity, eta, k, minimum):
    n = protein_similarity.shape[0]
    smoother = np.zeros((n, n), dtype=np.float64)
    supported = np.zeros(n, dtype=bool)
    for protein in range(n):
        similarities = protein_similarity[protein].astype(np.float64).copy()
        similarities[protein] = -1.0
        indices, weights = top_weights(similarities, eta, k, minimum)
        if len(indices) and weights.sum() > 1e-12:
            smoother[protein, indices] = weights / weights.sum()
            supported[protein] = True
    return smoother, supported


def build_inner_folds(train_drugs, seed=42, folds=5):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_drugs)
    return [np.asarray(x, dtype=int) for x in np.array_split(shuffled, folds)]


def oof_base_stats(train_drugs, folds, residual_grid, baseline_grid, label_grid, drug_similarity, drug_config):
    means, variances, supports, baselines, labels, query_ids = [], [], [], [], [], []
    for held_out in folds:
        references = np.asarray([d for d in train_drugs if d not in set(held_out)], dtype=int)
        mean, variance, support = drug_kernel_stats(
            held_out, references, residual_grid[references], drug_similarity, drug_config
        )
        means.append(mean)
        variances.append(variance)
        supports.append(support)
        baselines.append(baseline_grid[held_out])
        labels.append(label_grid[held_out])
        query_ids.extend(held_out.tolist())
    return {
        "mean": np.concatenate(means),
        "variance": np.concatenate(variances),
        "support": np.concatenate(supports),
        "baseline": np.concatenate(baselines),
        "label": np.concatenate(labels),
        "query_drugs": np.asarray(query_ids, dtype=int),
    }


def mse(prediction, label):
    return float(np.mean((prediction - label) ** 2))


def select_r1(train_drugs, folds, residual_grid, baseline_grid, label_grid, drug_similarity):
    baseline_mse = mse(baseline_grid[train_drugs], label_grid[train_drugs])
    rows, best = [], None
    for gamma in (1.0, 2.0, 4.0):
        for k_drug in (4, 8, 16, 32):
            for minimum in (0.0, 0.2, 0.4):
                drug_config = {"gamma": gamma, "k_drug": k_drug, "min_drug_similarity": minimum}
                stats = oof_base_stats(
                    train_drugs, folds, residual_grid, baseline_grid, label_grid, drug_similarity, drug_config
                )
                for tau in (0.05, 0.1, 0.25, 0.5, 1.0):
                    for beta in (0.0, 0.5, 1.0, 2.0):
                        for clip in (0.25, 0.5, 1.0):
                            for scale in (0.25, 0.5, 0.75, 1.0):
                                gate = {"tau": tau, "beta": beta, "clip": clip, "scale": scale}
                                prediction, _, _ = apply_correction(
                                    stats["baseline"], stats["mean"], stats["variance"], stats["support"], gate
                                )
                                score = mse(prediction, stats["label"])
                                row = {**drug_config, **gate, "inner_mse": score, "inner_gain": baseline_mse - score}
                                rows.append(row)
                                if best is None or score < best["inner_mse"]:
                                    best = {**row, "stats": stats}
    return best, pd.DataFrame(rows).sort_values("inner_mse"), baseline_mse


def cross_oof(train_drugs, folds, residual_grid, drug_similarity, drug_config, smoother):
    output = []
    for held_out in folds:
        references = np.asarray([d for d in train_drugs if d not in set(held_out)], dtype=int)
        smoothed_reference = residual_grid[references] @ smoother.T
        mean, _, _ = drug_kernel_stats(held_out, references, smoothed_reference, drug_similarity, drug_config)
        output.append(mean)
    return np.concatenate(output)


def select_r2(train_drugs, folds, residual_grid, protein_similarity, drug_similarity, r1_best):
    stats = r1_best["stats"]
    drug_config = {key: r1_best[key] for key in ("gamma", "k_drug", "min_drug_similarity")}
    gate = {key: r1_best[key] for key in ("tau", "beta", "clip", "scale")}
    rows, best, best_active = [], None, None
    for eta in (1.0, 2.0, 4.0):
        for k_protein in (4, 8, 16, 32):
            for minimum in (0.2, 0.4, 0.6):
                smoother, supported = protein_smoother(protein_similarity, eta, k_protein, minimum)
                cross = cross_oof(train_drugs, folds, residual_grid, drug_similarity, drug_config, smoother)
                # Include the no-KLIFS boundary so model selection can reject
                # cross-protein transfer instead of being forced to use it.
                for same_weight in (0.25, 0.5, 0.75, 0.9, 0.95, 1.0):
                    raw = same_weight * stats["mean"] + (1.0 - same_weight) * cross
                    raw[:, ~supported] = stats["mean"][:, ~supported]
                    disagreement = (1.0 - same_weight) * (stats["mean"] - cross) ** 2
                    disagreement[:, ~supported] = 0.0
                    prediction, _, _ = apply_correction(
                        stats["baseline"], raw, stats["variance"], stats["support"], gate, disagreement
                    )
                    score = mse(prediction, stats["label"])
                    row = {
                        "eta": eta,
                        "k_protein": k_protein,
                        "min_protein_similarity": minimum,
                        "same_weight": same_weight,
                        "inner_mse": score,
                        "inner_gain_vs_r1": r1_best["inner_mse"] - score,
                    }
                    rows.append(row)
                    if best is None or score < best["inner_mse"]:
                        best = {**row, "smoother": smoother, "supported": supported, "cross": cross}
                    if same_weight < 1.0 and (best_active is None or score < best_active["inner_mse"]):
                        best_active = {**row, "smoother": smoother, "supported": supported, "cross": cross}
    return best, best_active, pd.DataFrame(rows).sort_values("inner_mse")


def outer_predictions(query_drugs, reference_drugs, residual_grid, baseline_grid, drug_similarity, r1_best, r2_best=None):
    drug_config = {key: r1_best[key] for key in ("gamma", "k_drug", "min_drug_similarity")}
    gate = {key: r1_best[key] for key in ("tau", "beta", "clip", "scale")}
    same, variance, support = drug_kernel_stats(
        query_drugs, reference_drugs, residual_grid[reference_drugs], drug_similarity, drug_config
    )
    if r2_best is None:
        raw, disagreement = same, None
    else:
        smoothed_reference = residual_grid[reference_drugs] @ r2_best["smoother"].T
        cross, _, _ = drug_kernel_stats(query_drugs, reference_drugs, smoothed_reference, drug_similarity, drug_config)
        weight = r2_best["same_weight"]
        raw = weight * same + (1.0 - weight) * cross
        raw[:, ~r2_best["supported"]] = same[:, ~r2_best["supported"]]
        disagreement = (1.0 - weight) * (same - cross) ** 2
        disagreement[:, ~r2_best["supported"]] = 0.0
    prediction, correction, alpha = apply_correction(
        baseline_grid[query_drugs], raw, variance, support, gate, disagreement
    )
    return prediction, correction, alpha


def bootstrap_by_drug(label, predictions, baseline, n=20000, seed=20260901):
    improvements = ((baseline - label) ** 2 - (predictions - label) ** 2).mean(axis=1)
    rng = np.random.default_rng(seed)
    samples = improvements[rng.integers(0, len(improvements), size=(n, len(improvements)))].mean(axis=1)
    return {
        "drugs_improved": int((improvements > 0).sum()),
        "drugs_worsened": int((improvements < 0).sum()),
        "drug_improvements": improvements.tolist(),
        "macro_improvement": float(improvements.mean()),
        "bootstrap_95_ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "bootstrap_probability_positive": float(np.mean(samples > 0)),
    }


def clean_config(config):
    return {key: value for key, value in config.items() if key not in {"stats", "smoother", "supported", "cross"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffle-controls", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text())
    # Test indices are intentionally never materialized.
    sim = np.load(args.similarity)
    drug_ids = [str(x) for x in sim["drug_ids"].tolist()]
    protein_ids = [str(x) for x in sim["protein_ids"].tolist()]
    drug_lookup = {x: i for i, x in enumerate(drug_ids)}
    protein_lookup = {x: i for i, x in enumerate(protein_ids)}
    n_drugs, n_proteins = len(drug_ids), len(protein_ids)
    row_index = np.full((n_drugs, n_proteins), -1, dtype=int)
    for row, (drug, protein) in enumerate(zip(data["drug_id"], data["protein_id"])):
        row_index[drug_lookup[str(drug)], protein_lookup[str(protein)]] = row
    if (row_index < 0).any():
        raise RuntimeError("Davis pair grid is incomplete")
    labels = data["label"].numpy()[row_index].astype(np.float64)
    baseline = data["prediction"].numpy()[row_index].astype(np.float64)
    residual = labels - baseline
    drug_similarity = sim["drug_similarity"].astype(np.float64)
    protein_similarity = sim["protein_similarity"].astype(np.float64)
    train_drugs = np.asarray([drug_lookup[str(x)] for x in split["train_drugs"]], dtype=int)
    val_drugs = np.asarray([drug_lookup[str(x)] for x in split["val_drugs"]], dtype=int)
    folds = build_inner_folds(train_drugs)

    r1_best, r1_grid, inner_baseline_mse = select_r1(
        train_drugs, folds, residual, baseline, labels, drug_similarity
    )
    r1_grid.head(200).to_csv(args.output_dir / "r1_inner_top200.csv", index=False)
    r2_best, r2_best_active, r2_grid = select_r2(
        train_drugs, folds, residual, protein_similarity, drug_similarity, r1_best
    )
    r2_grid.to_csv(args.output_dir / "r2_inner_grid.csv", index=False)

    r0_prediction = baseline[val_drugs]
    r1_prediction, r1_correction, r1_alpha = outer_predictions(
        val_drugs, train_drugs, residual, baseline, drug_similarity, r1_best
    )
    r2_prediction, r2_correction, r2_alpha = outer_predictions(
        val_drugs, train_drugs, residual, baseline, drug_similarity, r1_best, r2_best
    )
    r2_active_prediction, r2_active_correction, r2_active_alpha = outer_predictions(
        val_drugs, train_drugs, residual, baseline, drug_similarity, r1_best, r2_best_active
    )

    y = labels[val_drugs]
    flat_y = y.ravel()
    results = {
        "guardrail": "outer test indices and metrics were not accessed",
        "inner_drug_cold_folds": [len(x) for x in folds],
        "inner": {
            "R0_mse": inner_baseline_mse,
            "R1_best": clean_config(r1_best),
            "R2_best": clean_config(r2_best),
            "R2_best_active": clean_config(r2_best_active),
        },
        "outer_validation": {
            "R0": metrics(flat_y, r0_prediction.ravel()),
            "R1": metrics(flat_y, r1_prediction.ravel()),
            "R2": metrics(flat_y, r2_prediction.ravel()),
            "R2_active": metrics(flat_y, r2_active_prediction.ravel()),
            "R1_bootstrap": bootstrap_by_drug(y, r1_prediction, r0_prediction),
            "R2_bootstrap": bootstrap_by_drug(y, r2_prediction, r0_prediction),
            "R2_active_bootstrap": bootstrap_by_drug(y, r2_active_prediction, r0_prediction),
            "R1_correction_abs_mean": float(np.abs(r1_correction).mean()),
            "R2_correction_abs_mean": float(np.abs(r2_correction).mean()),
            "R2_active_correction_abs_mean": float(np.abs(r2_active_correction).mean()),
            "R1_alpha_mean": float(r1_alpha.mean()),
            "R2_alpha_mean": float(r2_alpha.mean()),
            "R2_active_alpha_mean": float(r2_active_alpha.mean()),
        },
    }

    rng = np.random.default_rng(20260901)
    controls = []
    # Shuffle controls test the best genuinely KLIFS-active configuration,
    # even when overall selection prefers the no-KLIFS boundary.
    protein_config = clean_config(r2_best_active)
    for control in range(args.shuffle_controls):
        permutation = rng.permutation(n_proteins)
        shuffled_similarity = protein_similarity[permutation][:, permutation]
        smoother, supported = protein_smoother(
            shuffled_similarity,
            protein_config["eta"],
            protein_config["k_protein"],
            protein_config["min_protein_similarity"],
        )
        shuffled_r2 = {**protein_config, "smoother": smoother, "supported": supported}
        prediction, _, _ = outer_predictions(
            val_drugs, train_drugs, residual, baseline, drug_similarity, r1_best, shuffled_r2
        )
        control_metrics = metrics(flat_y, prediction.ravel())
        controls.append({"shuffle": control, **control_metrics})
    control_table = pd.DataFrame(controls)
    control_table.to_csv(args.output_dir / "r3_shuffled_klifs_controls.csv", index=False)
    results["outer_validation"]["R3_shuffled_KLIFS"] = {
        "mse_mean": float(control_table["mse"].mean()),
        "mse_std": float(control_table["mse"].std(ddof=1)),
        "mse_min": float(control_table["mse"].min()),
        "mse_max": float(control_table["mse"].max()),
        "real_R2_active_better_than_fraction_of_shuffles": float(np.mean(results["outer_validation"]["R2_active"]["mse"] < control_table["mse"])),
    }

    prediction_rows = []
    for local_drug, drug in enumerate(val_drugs):
        for protein in range(n_proteins):
            prediction_rows.append({
                "drug_id": drug_ids[drug],
                "protein_id": protein_ids[protein],
                "label": y[local_drug, protein],
                "R0": r0_prediction[local_drug, protein],
                "R1": r1_prediction[local_drug, protein],
                "R2": r2_prediction[local_drug, protein],
                "R2_active": r2_active_prediction[local_drug, protein],
                "R1_correction": r1_correction[local_drug, protein],
                "R2_correction": r2_correction[local_drug, protein],
                "R2_active_correction": r2_active_correction[local_drug, protein],
            })
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "outer_validation_predictions.csv", index=False)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
