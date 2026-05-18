# -*- coding: utf-8 -*-
import json
import argparse
from pathlib import Path

import pandas as pd

from datasets.davis_dataset_p13d import DavisDatasetP13D


def build_fixed_split_from_df(df: pd.DataFrame, val_ratio: float = 0.1, seed: int = 42):
    n = len(df)
    if n <= 1:
        return list(range(n)), []

    val_size = max(1, int(round(n * val_ratio)))
    val_size = min(val_size, n - 1)

    sampled = df.sample(frac=1.0, random_state=seed).reset_index()
    val_indices = sampled.loc[: val_size - 1, "index"].tolist()
    train_indices = sampled.loc[val_size:, "index"].tolist()
    return train_indices, val_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", type=str, default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", type=str, default="data/processed/davis/drug_unimol")
    parser.add_argument("--protein_1d_dir", type=str, default="data/processed/davis/protein_1d_esm2")

    parser.add_argument("--old_protein_3d_dir", type=str, default="data/processed/davis/protein_3d_min")
    parser.add_argument("--new_protein_3d_dir", type=str, default="data/processed/davis/protein_3d_gvp")

    parser.add_argument("--split_json_out", type=str, default="data/splits/davis_fixed_split_p13d_gvp.json")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    old_ds = DavisDatasetP13D(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.old_protein_3d_dir,
        use_drug_2d=False,
    )

    new_ds = DavisDatasetP13D(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.new_protein_3d_dir,
        use_drug_2d=False,
    )

    old_df = old_ds.df[["drug_id", "protein_id", "label"]].copy().reset_index(drop=True)
    new_df = new_ds.df[["drug_id", "protein_id", "label"]].copy().reset_index(drop=True)

    equal = old_df.equals(new_df)

    summary = {
        "old_num_pairs": int(len(old_df)),
        "new_num_pairs": int(len(new_df)),
        "is_exactly_equal": bool(equal),
        "old_protein_3d_dir": str(args.old_protein_3d_dir),
        "new_protein_3d_dir": str(args.new_protein_3d_dir),
    }

    if equal:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("[OK] old/new dataset filtering is exactly identical. You can keep using previous fixed split.")
        return

    old_key = old_df.assign(_src="old")
    new_key = new_df.assign(_src="new")
    merged = old_key.merge(new_key, on=["drug_id", "protein_id", "label"], how="outer", indicator=True)

    only_old = merged[merged["_merge"] == "left_only"][ ["drug_id", "protein_id", "label"] ]
    only_new = merged[merged["_merge"] == "right_only"][ ["drug_id", "protein_id", "label"] ]

    diff_dir = Path(args.new_protein_3d_dir)
    only_old_csv = diff_dir / "filter_diff_only_old.csv"
    only_new_csv = diff_dir / "filter_diff_only_new.csv"
    only_old.to_csv(only_old_csv, index=False)
    only_new.to_csv(only_new_csv, index=False)

    train_indices, val_indices = build_fixed_split_from_df(new_df, val_ratio=args.val_ratio, seed=args.seed)

    split_out = Path(args.split_json_out)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    split_obj = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "meta": {
            "source": "new_ds.df",
            "num_total": int(len(new_df)),
            "num_train": int(len(train_indices)),
            "num_val": int(len(val_indices)),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
        },
    }
    with open(split_out, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, indent=2, ensure_ascii=False)

    summary.update(
        {
            "num_only_old": int(len(only_old)),
            "num_only_new": int(len(only_new)),
            "only_old_csv": str(only_old_csv),
            "only_new_csv": str(only_new_csv),
            "new_split_json": str(split_out),
        }
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[WARN] old/new dataset filtering differs. Generated a NEW fixed split from new_ds.df. Do not use the old split.")


if __name__ == "__main__":
    main()
