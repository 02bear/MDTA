#!/usr/bin/env python3
"""Train fold1 cached-feature controls without touching Davis test rows."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


class AffineCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, global_prediction, local=None):
        return self.scale * global_prediction + self.offset


class BiasFreeResidual(nn.Module):
    def __init__(self, dim=128, bottleneck=32):
        super().__init__()
        self.first = nn.Linear(dim, bottleneck, bias=False)
        self.second = nn.Linear(bottleneck, 1, bias=False)
        nn.init.zeros_(self.second.weight)

    def forward(self, global_prediction, local):
        delta = self.second(torch.nn.functional.silu(self.first(local))).squeeze(-1)
        return global_prediction + delta


def metrics(prediction, target):
    error = prediction - target
    mse = float(error.square().mean())
    pred_center = prediction - prediction.mean()
    target_center = target - target.mean()
    pearson = float(
        (pred_center * target_center).sum()
        / (pred_center.square().sum().sqrt() * target_center.square().sum().sqrt()).clamp_min(1e-12)
    )
    return {"mse": mse, "rmse": math.sqrt(mse), "pearson": pearson}


def fit(condition, seed, global_prediction, local, target, train_idx, val_idx, args):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if condition == "affine":
        model = AffineCalibration().to(args.device)
        normalized = None
    else:
        model = BiasFreeResidual(local.shape[1]).to(args.device)
        train_local = local[train_idx]
        mean = train_local.mean(0)
        std = train_local.std(0).clamp_min(1e-5)
        normalized = (local - mean) / std
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_mse, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            gp = global_prediction[idx].to(args.device)
            y = target[idx].to(args.device)
            z = normalized[idx].to(args.device) if normalized is not None else None
            loss = torch.nn.functional.mse_loss(model(gp, z), y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            gp = global_prediction[val_idx].to(args.device)
            z = normalized[val_idx].to(args.device) if normalized is not None else None
            pred = model(gp, z).cpu()
            val_mse = float((pred - target[val_idx]).square().mean())
        history.append({"epoch": epoch, "val_mse": val_mse})
        if val_mse < best_mse - 1e-10:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state); model.to(args.device).eval()
    with torch.no_grad():
        gp = global_prediction[val_idx].to(args.device)
        z = normalized[val_idx].to(args.device) if normalized is not None else None
        prediction = model(gp, z).cpu()
    result = metrics(prediction, target[val_idx])
    result.update({
        "condition": condition, "seed": seed, "best_epoch": best_epoch,
        "delta_mean": float((prediction - global_prediction[val_idx]).mean()),
        "delta_std": float((prediction - global_prediction[val_idx]).std()),
        "history": history,
    })
    return result, best_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--global-cache", type=Path, required=True)
    p.add_argument("--local-cache", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    args = p.parse_args()

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    local_data = torch.load(args.local_cache, map_location="cpu", weights_only=False)
    if global_data["drug_id"] != local_data["drug_id"] or global_data["protein_id"] != local_data["protein_id"]:
        raise RuntimeError("global/local cache row identity mismatch")
    split = json.loads(args.split.read_text())
    train_idx = torch.tensor(split["train_indices"], dtype=torch.long)
    val_idx = torch.tensor(split["val_indices"], dtype=torch.long)
    # Deliberately do not materialize or evaluate test_indices.
    global_prediction = global_data["prediction"].float()
    target = global_data["label"].float()
    baseline = metrics(global_prediction[val_idx], target[val_idx])
    baseline.update({
        "delta_mean": 0.0, "delta_std": 0.0,
        "validation_rows": len(val_idx), "test_rows_accessed": 0,
    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"global_frozen": baseline, "runs": []}
    condition_to_key = {
        "affine": None, "random": "random", "pretrained": "pretrained",
        "atom_shuffle": "atom_shuffle", "mismatch": "mismatch",
    }
    for condition, key in condition_to_key.items():
        for seed in args.seeds:
            local = local_data[key].float() if key else local_data["pretrained"].float()
            result, state = fit(
                condition, seed, global_prediction, local, target,
                train_idx, val_idx, args,
            )
            results["runs"].append(result)
            torch.save(state, args.output_dir / f"{condition}_seed{seed}.pt")
            print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)

    grouped = {}
    for condition in condition_to_key:
        rows = [x for x in results["runs"] if x["condition"] == condition]
        grouped[condition] = {
            "mse_mean": float(np.mean([x["mse"] for x in rows])),
            "mse_std": float(np.std([x["mse"] for x in rows], ddof=1)),
            "delta_std_mean": float(np.mean([x["delta_std"] for x in rows])),
            "improving_seeds": sum(x["mse"] < baseline["mse"] for x in rows),
        }
    base = baseline["mse"]
    p0 = grouped["pretrained"]["mse_mean"]
    gain = base - p0
    go_no_go = {
        "relative_validation_mse_improvement": gain / base,
        "at_least_two_percent": gain / base >= 0.02,
        "consistent_at_least_two_of_three": grouped["pretrained"]["improving_seeds"] >= 2,
        "better_than_affine": p0 < grouped["affine"]["mse_mean"],
        "better_than_random": p0 < grouped["random"]["mse_mean"],
        "shuffle_loses_at_least_half_gain": grouped["atom_shuffle"]["mse_mean"] >= base - 0.5 * gain,
        "mismatch_loses_at_least_half_gain": grouped["mismatch"]["mse_mean"] >= base - 0.5 * gain,
        "local_delta_nonconstant": grouped["pretrained"]["delta_std_mean"] > 1e-3,
    }
    go_no_go["pass"] = all(go_no_go.values())
    results["summary"] = grouped
    results["go_no_go"] = go_no_go
    results["protocol"] = {
        "train_rows": len(train_idx), "validation_rows": len(val_idx), "test_rows_accessed": 0,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "max_epochs": args.epochs, "patience": args.patience,
    }
    (args.output_dir / "pilot_results.json").write_text(json.dumps(results, indent=2))
    print("FINAL", json.dumps({"baseline": baseline, "summary": grouped, "go_no_go": go_no_go}, indent=2))


if __name__ == "__main__":
    main()
