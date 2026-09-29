#!/usr/bin/env python3
"""Paired validation-drug analysis for an enabled protected checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_klifs_interact import IndexDataset, KLIFSInteractP13D, PairStore, evaluate, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--ligand-dir", type=Path, required=True)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    feature_data = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    store = PairStore(global_data, feature_data, args.ligand_dir, args.similarity)
    split = json.loads(args.split.read_text())
    val_idx = np.asarray(split["val_indices"], dtype=int)
    # Intentionally never materialize split['test_indices'].
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not checkpoint["enabled"]:
        raise ValueError("Checkpoint is disabled")
    model = KLIFSInteractP13D().to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(
        IndexDataset(len(store.label)), batch_size=128, sampler=val_idx.tolist(),
        collate_fn=store.collate, num_workers=0,
    )
    prediction, label, gate, delta = evaluate(model, loader, args.device)
    baseline = store.prediction[val_idx]
    rows = pd.DataFrame({
        "drug_id": [store.drug_ids[i] for i in val_idx],
        "protein_id": [store.protein_ids[i] for i in val_idx],
        "label": label.numpy(),
        "baseline_prediction": baseline.numpy(),
        "local_prediction": prediction.numpy(),
        "gate": gate.numpy(),
        "delta": delta.numpy(),
    })
    rows["baseline_squared_error"] = (rows["baseline_prediction"] - rows["label"]) ** 2
    rows["local_squared_error"] = (rows["local_prediction"] - rows["label"]) ** 2
    rows["mse_improvement"] = rows["baseline_squared_error"] - rows["local_squared_error"]
    per_drug = rows.groupby("drug_id", as_index=False).agg(
        n=("protein_id", "size"),
        baseline_mse=("baseline_squared_error", "mean"),
        local_mse=("local_squared_error", "mean"),
        mse_improvement=("mse_improvement", "mean"),
        delta_mean=("delta", "mean"),
        delta_std=("delta", "std"),
    )
    rng = np.random.default_rng(20260901)
    improvements = per_drug["mse_improvement"].to_numpy()
    bootstrap = improvements[rng.integers(0, len(improvements), size=(args.bootstrap, len(improvements)))].mean(axis=1)
    summary = {
        "condition": checkpoint["condition"],
        "seed": checkpoint["seed"],
        "best_epoch": checkpoint["best_epoch"],
        "test_rows_accessed": 0,
        "baseline": metrics(label.numpy(), baseline.numpy()),
        "local": metrics(label.numpy(), prediction.numpy()),
        "validation_drugs": len(per_drug),
        "drugs_improved": int((improvements > 0).sum()),
        "drugs_worsened": int((improvements < 0).sum()),
        "drug_macro_mse_improvement": float(improvements.mean()),
        "drug_bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "bootstrap_probability_improvement_positive": float(np.mean(bootstrap > 0)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    per_drug.to_csv(args.output_dir / "per_drug.csv", index=False)
    (args.output_dir / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(per_drug.to_string(index=False))


if __name__ == "__main__":
    main()
