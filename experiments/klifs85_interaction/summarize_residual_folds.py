#!/usr/bin/env python3
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


rows = []
all_drug_improvements = []
validation_drugs = []
improvements_by_drug = defaultdict(list)
for filename in sorted(glob.glob("experiments/klifs85_interaction/outputs/residual_kernel_fold*_v2/results.json")):
    path = Path(filename)
    data = json.loads(path.read_text())
    fold = int(path.parent.name.split("fold", 1)[1].split("_", 1)[0])
    split_path = Path(f"data/splits/davis_drug_cold_5fold_seed42/fold_{fold}/split.json")
    split = json.loads(split_path.read_text())
    validation_drugs.extend(str(x) for x in split["val_drugs"])
    inner = data["inner"]
    outer = data["outer_validation"]
    drug_improvements = outer["R1_bootstrap"]["drug_improvements"]
    all_drug_improvements.extend(drug_improvements)
    for drug_id, improvement in zip(split["val_drugs"], drug_improvements):
        improvements_by_drug[str(drug_id)].append(improvement)
    rows.append({
        "fold": fold,
        "inner_r0": inner["R0_mse"],
        "inner_r1": inner["R1_best"]["inner_mse"],
        "r2_weight": inner["R2_best"]["same_weight"],
        "r2_active_weight": inner["R2_best_active"]["same_weight"],
        "r0_mse": outer["R0"]["mse"],
        "r1_mse": outer["R1"]["mse"],
        "r2_active_mse": outer["R2_active"]["mse"],
        "r0_ci": outer["R0"]["ci"],
        "r1_ci": outer["R1"]["ci"],
        "improved_drugs": outer["R1_bootstrap"]["drugs_improved"],
        "worsened_drugs": outer["R1_bootstrap"]["drugs_worsened"],
        "bootstrap_ci": outer["R1_bootstrap"]["bootstrap_95_ci"],
        "real_klifs_vs_shuffle": outer["R3_shuffled_KLIFS"][
            "real_R2_active_better_than_fraction_of_shuffles"
        ],
        "drug_improvements": drug_improvements,
    })

weights = np.asarray([len(row["drug_improvements"]) for row in rows], dtype=np.float64)
r0 = np.asarray([row["r0_mse"] for row in rows])
r1 = np.asarray([row["r1_mse"] for row in rows])
improvements = np.asarray(all_drug_improvements, dtype=np.float64)
rng = np.random.default_rng(20260901)
samples = improvements[
    rng.integers(0, len(improvements), size=(50000, len(improvements)))
].mean(axis=1)
unique_drug_improvements = np.asarray([
    np.mean(values) for values in improvements_by_drug.values()
], dtype=np.float64)
unique_samples = unique_drug_improvements[
    rng.integers(
        0,
        len(unique_drug_improvements),
        size=(50000, len(unique_drug_improvements)),
    )
].mean(axis=1)
aggregate = {
    "folds": len(rows),
    "validation_drug_instances": int(weights.sum()),
    "unique_validation_drugs": len(set(validation_drugs)),
    "folds_r1_mse_improved": int(np.sum(r1 < r0)),
    "drug_instances_improved": int(np.sum(improvements > 0)),
    "drug_instances_worsened": int(np.sum(improvements < 0)),
    "drug_instances_unchanged": int(np.sum(improvements == 0)),
    "pooled_r0_mse": float(np.average(r0, weights=weights)),
    "pooled_r1_mse": float(np.average(r1, weights=weights)),
    "pooled_relative_mse_gain": float(np.average(r0 - r1, weights=weights) / np.average(r0, weights=weights)),
    "drug_macro_mse_improvement": float(improvements.mean()),
    "drug_bootstrap_95_ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    "drug_bootstrap_probability_positive": float(np.mean(samples > 0)),
    "unique_drug_macro_mse_improvement": float(unique_drug_improvements.mean()),
    "unique_drug_bootstrap_95_ci": [
        float(np.quantile(unique_samples, 0.025)),
        float(np.quantile(unique_samples, 0.975)),
    ],
    "unique_drug_bootstrap_probability_positive": float(np.mean(unique_samples > 0)),
    "folds_selecting_active_klifs": int(sum(row["r2_weight"] < 1.0 for row in rows)),
    "folds_active_klifs_better_than_r1_outer": int(sum(row["r2_active_mse"] < row["r1_mse"] for row in rows)),
}
print(json.dumps({"folds": rows, "aggregate": aggregate}, indent=2))
