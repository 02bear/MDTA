#!/usr/bin/env python3
"""Technical completeness/leakage gate for one completed N3 outer fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from n3_common import N_PROTEINS, build_n3_split, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    frames, checkpoints = [], {}
    for cf in range(1, 6):
        base = args.output_root / f"fold_{args.fold}/cf_{cf}/stage_b_strict_refit"
        result = json.loads((base / "result.json").read_text(encoding="utf-8"))
        for key in ("INNER_WEIGHTS_REUSED", "HOLDOUT_USED_FOR_TRAINING",
                    "HOLDOUT_USED_FOR_EPOCH_SELECTION",
                    "holdout_labels_parsed_before_checkpoint_freeze",
                    "holdout_labels_parsed_before_final_oof_inference"):
            if result.get(key) is not False:
                raise RuntimeError(f"fold {args.fold} cf {cf}: {key} is not False")
        expected = build_n3_split(args.project, args.fold, cf)
        for key in ("holdout_drugs", "T_j", "epoch_train_drugs", "epoch_val_drugs"):
            if result["split"][key] != expected[key]:
                raise RuntimeError(f"fold {args.fold} cf {cf}: split mismatch {key}")
        checkpoint = Path(result["checkpoint"])
        if sha256_file(checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError(f"fold {args.fold} cf {cf}: checkpoint SHA mismatch")
        frame = pd.read_csv(base / "oof_predictions.csv", dtype={"drug_id": str, "protein_id": str})
        if set(frame.drug_id) != set(expected["holdout_drugs"]):
            raise RuntimeError(f"fold {args.fold} cf {cf}: holdout membership mismatch")
        frames.append(frame)
        checkpoints[str(cf)] = {"E_star": result["E_star"], "checkpoint": str(checkpoint), "sha256": result["checkpoint_sha256"]}
    all_rows = pd.concat(frames, ignore_index=True)
    outer_train = build_n3_split(args.project, args.fold, 1)["outer_train_drugs"]
    numeric = all_rows[["label", "historical_insample_pred", "strict_oof_pred", "residual_insample", "residual_oof"]].to_numpy()
    checks = {
        "expected_pairs": len(outer_train)*N_PROTEINS,
        "observed_pairs": len(all_rows),
        "each_outer_train_drug_exactly_once": bool(set(all_rows.drug_id) == set(outer_train) and all_rows.drug_id.nunique() == len(outer_train)),
        "each_drug_has_442_pairs": bool((all_rows.groupby("drug_id").size() == N_PROTEINS).all()),
        "each_drug_from_one_cf_model": bool((all_rows.groupby("drug_id").cf_fold.nunique() == 1).all()),
        "duplicates": int(all_rows.duplicated(["drug_id", "protein_id"]).sum()),
        "NaN_or_Inf": int((~np.isfinite(numeric)).sum()),
        "all_five_checkpoint_guards_passed": True,
    }
    passed = (
        checks["expected_pairs"] == checks["observed_pairs"]
        and checks["each_outer_train_drug_exactly_once"]
        and checks["each_drug_has_442_pairs"]
        and checks["each_drug_from_one_cf_model"]
        and checks["duplicates"] == 0 and checks["NaN_or_Inf"] == 0
    )
    result = {"fold": args.fold, "passed": passed, "checks": checks, "checkpoints": checkpoints}
    write_json(args.output_root / f"audit/fold_{args.fold}_oof_technical_gate.json", result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
