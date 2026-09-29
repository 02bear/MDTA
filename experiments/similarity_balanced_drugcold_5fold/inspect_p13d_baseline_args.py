#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    args = parser.parse_args()

    root = Path(args.baseline_root)
    records = []
    for fold in range(1, 6):
        path = root / f"fold_{fold}" / "best_model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        records.append({
            "fold": fold,
            "checkpoint": str(path.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "args": checkpoint["args"],
        })

    variable_keys = {"split_json", "output_dir"}
    reference = {
        key: value for key, value in records[0]["args"].items()
        if key not in variable_keys
    }
    mismatches = []
    for record in records[1:]:
        comparable = {
            key: value for key, value in record["args"].items()
            if key not in variable_keys
        }
        if comparable != reference:
            mismatches.append(record["fold"])

    print(json.dumps({
        "all_non_path_args_identical": not mismatches,
        "mismatching_folds": mismatches,
        "reference_non_path_args": reference,
        "fold_records": records,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
