#!/usr/bin/env python3
"""Independent audit for v2; deliberately does not import the builder."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--similarity-npz", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs_csv, dtype={"drug_id": str, "protein_id": str})
    cache = np.load(args.similarity_npz, allow_pickle=True)
    cache_ids = [str(value) for value in cache["drug_ids"].tolist()]
    similarity = np.asarray(cache["drug_similarity"], dtype=float)
    position = {drug: index for index, drug in enumerate(cache_ids)}
    all_drugs = set(pairs["drug_id"].unique())
    if all_drugs != set(cache_ids):
        raise AssertionError("Drug IDs disagree between pairs and similarity cache")

    root = Path(args.split_root)
    outer_seen, inner_seen, rows = [], [], []
    for fold in range(1, 6):
        split = read_json(root / f"fold_{fold}" / "split.json")
        train_drugs = set(map(str, split["inner_train_drugs"]))
        val_drugs = set(map(str, split["inner_validation_drugs"]))
        development_drugs = set(map(str, split["outer_development_drugs"]))
        test_drugs = set(map(str, split["outer_test_drugs"]))
        assert len(val_drugs) == 7
        assert len(test_drugs) in (13, 14)
        assert train_drugs.isdisjoint(val_drugs)
        assert development_drugs.isdisjoint(test_drugs)
        assert train_drugs | val_drugs == development_drugs
        assert development_drugs | test_drugs == all_drugs

        train_idx = set(map(int, split["train_indices"]))
        val_idx = set(map(int, split["val_indices"]))
        refit_idx = set(map(int, split["refit_indices"]))
        test_idx = set(map(int, split["test_indices"]))
        assert train_idx.isdisjoint(val_idx)
        assert refit_idx.isdisjoint(test_idx)
        assert train_idx | val_idx == refit_idx
        assert refit_idx | test_idx == set(range(len(pairs)))
        assert set(pairs.iloc[sorted(train_idx)]["drug_id"]) == train_drugs
        assert set(pairs.iloc[sorted(val_idx)]["drug_id"]) == val_drugs
        assert set(pairs.iloc[sorted(test_idx)]["drug_id"]) == test_drugs

        fold_values = {}
        for role, query_drugs, reference_drugs in (
            ("inner_validation", val_drugs, train_drugs),
            ("outer_test", test_drugs, development_drugs),
        ):
            query_pos = [position[drug] for drug in query_drugs]
            ref_pos = [position[drug] for drug in reference_drugs]
            top1 = similarity[np.ix_(query_pos, ref_pos)].max(axis=1)
            query_labels = pairs[pairs["drug_id"].isin(query_drugs)]["label"].to_numpy(float)
            reference_mean = float(
                pairs[pairs["drug_id"].isin(reference_drugs)]["label"].mean()
            )
            fold_values[role] = {
                "top1_mean": float(top1.mean()),
                "label_mean": float(query_labels.mean()),
                "label_std": float(query_labels.std()),
                "mean_shift_to_train": float(query_labels.mean() - reference_mean),
                "train_mean_constant_mse": float(
                    np.mean((query_labels - reference_mean) ** 2)
                ),
            }
        rows.append({"fold": fold, **fold_values})
        outer_seen.extend(test_drugs)
        inner_seen.extend(val_drugs)

    outer_counts = Counter(outer_seen)
    inner_counts = Counter(inner_seen)
    assert set(outer_counts) == all_drugs
    assert all(count == 1 for count in outer_counts.values())
    assert len(inner_counts) == 35
    assert all(count == 1 for count in inner_counts.values())

    def value_range(role, name):
        values = np.asarray([row[role][name] for row in rows])
        return float(values.max() - values.min())

    gates = {
        "outer_top1_mean_range_le_0.03": value_range("outer_test", "top1_mean") <= 0.03,
        "outer_label_mean_range_le_0.10": value_range("outer_test", "label_mean") <= 0.10,
        "outer_label_std_range_le_0.12": value_range("outer_test", "label_std") <= 0.12,
        "outer_constant_mse_range_le_0.20": value_range("outer_test", "train_mean_constant_mse") <= 0.20,
        "inner_top1_mean_range_le_0.04": value_range("inner_validation", "top1_mean") <= 0.04,
        "inner_label_mean_range_le_0.12": value_range("inner_validation", "label_mean") <= 0.12,
        "inner_label_std_range_le_0.14": value_range("inner_validation", "label_std") <= 0.14,
        "inner_constant_mse_range_le_0.22": value_range("inner_validation", "train_mean_constant_mse") <= 0.22,
        "inner_validation_unique_35_of_35": len(inner_counts) == 35,
    }
    result = {
        "auditor": "independent implementation; builder not imported",
        "pair_count": int(len(pairs)),
        "drug_count": int(len(all_drugs)),
        "outer_test_unique_68_of_68": len(outer_counts) == 68,
        "inner_validation_unique_35_of_35": len(inner_counts) == 35,
        "folds": rows,
        "ranges": {
            "outer_top1_mean": value_range("outer_test", "top1_mean"),
            "outer_label_mean": value_range("outer_test", "label_mean"),
            "outer_label_std": value_range("outer_test", "label_std"),
            "outer_constant_mse": value_range("outer_test", "train_mean_constant_mse"),
            "inner_top1_mean": value_range("inner_validation", "top1_mean"),
            "inner_label_mean": value_range("inner_validation", "label_mean"),
            "inner_label_std": value_range("inner_validation", "label_std"),
            "inner_constant_mse": value_range("inner_validation", "train_mean_constant_mse"),
        },
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_acceptance_gates_passed"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
