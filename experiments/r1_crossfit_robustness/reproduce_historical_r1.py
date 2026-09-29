#!/usr/bin/env python3
"""Mandatory N3 gate: reproduce historical P13D and locked R1 on all outer folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from n3_common import R1_CONFIGS, REPRO_TARGETS, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    sys.path.insert(0, str(args.project / "experiments/klifs85_interaction"))
    from experiments.klifs85_interaction import run_residual_kernel as r1
    from experiments.klifs85_interaction.train_klifs_interact import metrics

    similarity_path = args.project / "experiments/klifs85_interaction/data/similarity_audit_fold1/entity_similarities.npz"
    sim = np.load(similarity_path, allow_pickle=True)
    drug_ids = [str(x) for x in sim["drug_ids"].tolist()]
    protein_ids = [str(x) for x in sim["protein_ids"].tolist()]
    drug_lookup = {x: i for i, x in enumerate(drug_ids)}
    protein_lookup = {x: i for i, x in enumerate(protein_ids)}
    drug_similarity = sim["drug_similarity"].astype(np.float64)
    results = {"tolerance": args.tolerance, "folds": {}, "all_passed": True}

    for fold in range(1, 6):
        cache_path = args.project / f"experiments/pdbbind_to_davis_transfer/data/global_predictions/fold{fold}.pt"
        split_path = args.project / f"data/splits/davis_drug_cold_5fold_seed42/fold_{fold}/split.json"
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        row_index = np.full((len(drug_ids), len(protein_ids)), -1, dtype=int)
        for row, (drug, protein) in enumerate(zip(data["drug_id"], data["protein_id"])):
            row_index[drug_lookup[str(drug)], protein_lookup[str(protein)]] = row
        if (row_index < 0).any():
            raise RuntimeError(f"fold {fold}: incomplete Davis grid")
        labels = data["label"].numpy()[row_index].astype(np.float64)
        baseline = data["prediction"].numpy()[row_index].astype(np.float64)
        train = np.asarray([drug_lookup[str(x)] for x in split["train_drugs"]], dtype=int)
        val = np.asarray([drug_lookup[str(x)] for x in split["val_drugs"]], dtype=int)
        if not np.array_equal(np.sort(row_index[train].ravel()), np.sort(split["train_indices"])):
            raise RuntimeError(f"fold {fold}: train row mismatch")
        if not np.array_equal(np.sort(row_index[val].ravel()), np.sort(split["val_indices"])):
            raise RuntimeError(f"fold {fold}: validation row mismatch")
        residual = labels - baseline
        r1_pred, _, _ = r1.outer_predictions(
            val, train, residual, baseline, drug_similarity, R1_CONFIGS[fold]
        )
        p13d_m = metrics(labels[val].ravel(), baseline[val].ravel())
        r1_m = metrics(labels[val].ravel(), r1_pred.ravel())
        p_diff = abs(p13d_m["mse"] - REPRO_TARGETS[fold]["P13D"])
        r_diff = abs(r1_m["mse"] - REPRO_TARGETS[fold]["R1_in"])
        passed = p_diff < args.tolerance and r_diff < args.tolerance
        results["folds"][str(fold)] = {
            "P13D": p13d_m,
            "R1_in": r1_m,
            "P13D_target": REPRO_TARGETS[fold]["P13D"],
            "R1_in_target": REPRO_TARGETS[fold]["R1_in"],
            "P13D_abs_diff": p_diff,
            "R1_in_abs_diff": r_diff,
            "passed": passed,
            "global_cache": str(cache_path),
            "split": str(split_path),
            "R1_locked_config": R1_CONFIGS[fold],
        }
        results["all_passed"] = results["all_passed"] and passed
        print(json.dumps({"fold": fold, **results["folds"][str(fold)]}, indent=2), flush=True)

    write_json(args.output, results)
    print(f"HISTORICAL_REPRODUCTION_ALL_PASSED={results['all_passed']}", flush=True)
    if not results["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
