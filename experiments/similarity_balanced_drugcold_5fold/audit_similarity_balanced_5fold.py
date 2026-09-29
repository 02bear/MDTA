#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def top1_profile(query_drugs, reference_drugs, drug_to_pos, similarity):
    q = [drug_to_pos[str(d)] for d in query_drugs]
    r = [drug_to_pos[str(d)] for d in reference_drugs]
    values = similarity[np.ix_(q, r)].max(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def audit_root(root, pairs, drug_to_pos, similarity, is_new):
    all_drugs = set(pairs["drug_id"].astype(str).unique())
    test_seen = []
    rows = []
    for fold in range(1, 6):
        obj = load_json(root / f"fold_{fold}" / "split.json")
        train_idx = set(map(int, obj["train_indices"]))
        val_idx = set(map(int, obj["val_indices"]))
        test_idx = set(map(int, obj["test_indices"]))
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)
        train_drugs = set(pairs.iloc[sorted(train_idx)]["drug_id"].astype(str))
        val_drugs = set(pairs.iloc[sorted(val_idx)]["drug_id"].astype(str))
        test_drugs = set(pairs.iloc[sorted(test_idx)]["drug_id"].astype(str))
        assert train_drugs.isdisjoint(val_drugs)
        assert train_drugs.isdisjoint(test_drugs)
        assert val_drugs.isdisjoint(test_drugs)
        assert len(test_idx) == len(test_drugs) * 442
        if is_new:
            refit_idx = set(map(int, obj["refit_indices"]))
            assert refit_idx == train_idx | val_idx
            assert len(refit_idx) + len(test_idx) == len(pairs)
            assert set(map(str, obj["outer_test_drugs"])) == test_drugs
            reference_drugs = train_drugs | val_drugs
        else:
            reference_drugs = train_drugs | val_drugs
        profile = top1_profile(test_drugs, reference_drugs, drug_to_pos, similarity)
        rows.append({"fold": fold, "test_drugs": len(test_drugs), **profile})
        test_seen.extend(test_drugs)
    counts = {d: test_seen.count(d) for d in all_drugs}
    assert set(test_seen) == all_drugs
    assert all(value == 1 for value in counts.values())
    means = np.asarray([row["mean"] for row in rows], dtype=float)
    medians = np.asarray([row["median"] for row in rows], dtype=float)
    return {
        "folds": rows,
        "all_test_drugs_exactly_once": True,
        "top1_mean_across_fold_sd": float(means.std(ddof=1)),
        "top1_mean_across_fold_range": float(means.max() - means.min()),
        "top1_median_across_fold_sd": float(medians.std(ddof=1)),
        "top1_median_across_fold_range": float(medians.max() - medians.min()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--similarity-npz", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Labels are deliberately excluded from the independent audit read.
    pairs = pd.read_csv(args.pairs_csv, usecols=["drug_id", "protein_id"], dtype=str)
    cache = np.load(args.similarity_npz, allow_pickle=True)
    drug_ids = [str(x) for x in cache["drug_ids"].tolist()]
    similarity = np.asarray(cache["drug_similarity"], dtype=float)
    drug_to_pos = {drug: idx for idx, drug in enumerate(drug_ids)}

    result = {
        "auditor": "independent script; does not import split builder",
        "labels_read": False,
        "pair_count": int(len(pairs)),
        "drug_count": int(pairs["drug_id"].nunique()),
        "new_similarity_balanced": audit_root(
            Path(args.new_root), pairs, drug_to_pos, similarity, True
        ),
        "old_random_seed42": audit_root(
            Path(args.old_root), pairs, drug_to_pos, similarity, False
        ),
    }
    new = result["new_similarity_balanced"]
    old = result["old_random_seed42"]
    result["comparison"] = {
        "top1_mean_sd_reduction_fraction": float(
            1.0 - new["top1_mean_across_fold_sd"] / old["top1_mean_across_fold_sd"]
        ),
        "top1_mean_range_reduction_fraction": float(
            1.0 - new["top1_mean_across_fold_range"] / old["top1_mean_across_fold_range"]
        ),
        "top1_median_sd_reduction_fraction": float(
            1.0 - new["top1_median_across_fold_sd"] / old["top1_median_across_fold_sd"]
        ),
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
