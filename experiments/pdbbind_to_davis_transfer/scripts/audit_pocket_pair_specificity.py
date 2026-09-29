#!/usr/bin/env python3
"""Measure whether pocket-constrained PDBbind scores track Davis affinity."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(values):
    values = [x for x in values if x is not None and np.isfinite(x)]
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "positive_fraction": float(np.mean(np.asarray(values) > 0)) if values else None,
    }


def condition_metrics(frame, score_column):
    pearson = correlation(frame[score_column], frame["label"])
    spearman = correlation(
        frame[score_column].rank(method="average"),
        frame["label"].rank(method="average"),
    )
    within_protein_pearson, within_protein_spearman = [], []
    for _, group in frame.groupby("protein_id"):
        within_protein_pearson.append(correlation(group[score_column], group["label"]))
        within_protein_spearman.append(correlation(
            group[score_column].rank(method="average"),
            group["label"].rank(method="average"),
        ))
    within_drug_pearson, within_drug_spearman = [], []
    for _, group in frame.groupby("drug_id"):
        within_drug_pearson.append(correlation(group[score_column], group["label"]))
        within_drug_spearman.append(correlation(
            group[score_column].rank(method="average"),
            group["label"].rank(method="average"),
        ))
    return {
        "overall_pearson": pearson,
        "overall_spearman": spearman,
        "within_protein_drug_ranking_pearson": summarize(within_protein_pearson),
        "within_protein_drug_ranking_spearman": summarize(within_protein_spearman),
        "within_drug_protein_ranking_pearson": summarize(within_drug_pearson),
        "within_drug_protein_ranking_spearman": summarize(within_drug_spearman),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-cache", type=Path, required=True)
    p.add_argument("--global-cache", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    local = torch.load(args.local_cache, map_location="cpu", weights_only=False)
    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    if local["drug_id"] != global_data["drug_id"] or local["protein_id"] != global_data["protein_id"]:
        raise RuntimeError("cache row identity mismatch")
    frame = pd.DataFrame({
        "drug_id": local["drug_id"], "protein_id": local["protein_id"],
        "label": global_data["label"].numpy(),
    })
    conditions = ("pretrained", "random", "atom_shuffle", "mismatch")
    for condition in conditions:
        frame[f"score_{condition}"] = local[f"score_{condition}"].numpy()
    split = json.loads(args.split.read_text())
    report = {"test_rows_accessed": 0, "partitions": {}}
    for partition, key in (("train", "train_indices"), ("validation", "val_indices")):
        part = frame.iloc[split[key]].copy()
        report["partitions"][partition] = {
            "rows": len(part),
            "drug_count": part["drug_id"].nunique(),
            "protein_count": part["protein_id"].nunique(),
            "conditions": {
                condition: condition_metrics(part, f"score_{condition}")
                for condition in conditions
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
