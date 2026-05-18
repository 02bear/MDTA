# -*- coding: utf-8 -*-
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build fixed train/val split json by shuffled indices")
    parser.add_argument("--pairs_csv", type=str, required=True, help="Path to pairs.csv")
    parser.add_argument("--output_json", type=str, required=True, help="Path to output split json")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio in (0,1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {args.train_ratio}")

    pairs_path = Path(args.pairs_csv)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pairs_path)
    n = len(df)
    if n <= 1:
        raise ValueError(f"Not enough rows in pairs_csv: {n}")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(n * args.train_ratio)
    n_train = max(1, min(n - 1, n_train))

    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()

    split_info = {
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "num_total": int(n),
        "num_train": int(len(train_indices)),
        "num_val": int(len(val_indices)),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Total={n}, Train={len(train_indices)}, Val={len(val_indices)}")


if __name__ == "__main__":
    main()
