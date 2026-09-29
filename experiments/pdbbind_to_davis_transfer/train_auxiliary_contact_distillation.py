#!/usr/bin/env python3
"""Protected PDBbind-contact auxiliary training on frozen Davis fold-1 features.

The original p13d prediction is an immutable skip path.  A student adapter sees
only the original 256-D concatenated Davis representation at inference time.
PDBbind contact features are training-only alignment targets.  Epoch zero is the
exact frozen baseline and remains selectable, so a harmful auxiliary branch is
disabled rather than degrading the main task.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ProtectedAuxiliary(nn.Module):
    def __init__(self, input_dim=256, teacher_dim=128, bottleneck=32):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, teacher_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(teacher_dim),
        )
        self.residual = nn.Sequential(
            nn.Linear(teacher_dim, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, 1, bias=False),
        )
        # The first forward pass is exactly the frozen p13d prediction.
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, pair_feature, frozen_prediction):
        student = self.adapter(pair_feature)
        delta = self.residual(student).squeeze(-1)
        return frozen_prediction + delta, student


def metrics(prediction, target):
    error = prediction - target
    mse = float(error.square().mean())
    pred_center = prediction - prediction.mean()
    target_center = target - target.mean()
    pearson = float(
        (pred_center * target_center).sum()
        / (pred_center.square().sum().sqrt()
           * target_center.square().sum().sqrt()).clamp_min(1e-12)
    )
    return {"mse": mse, "rmse": math.sqrt(mse), "pearson": pearson}


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit(condition, teacher_key, seed, pair_feature, frozen_prediction, teacher,
        target, train_idx, val_idx, baseline_mse, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_train = pair_feature[train_idx]
    x_mean = x_train.mean(0)
    x_std = x_train.std(0).clamp_min(1e-5)
    x = (pair_feature - x_mean) / x_std

    teacher_normalized = None
    if teacher_key is not None:
        t_train = teacher[train_idx]
        t_mean = t_train.mean(0)
        t_std = t_train.std(0).clamp_min(1e-5)
        teacher_normalized = (teacher - t_mean) / t_std

    model = ProtectedAuxiliary(input_dim=x.shape[1], teacher_dim=args.teacher_dim).to(args.device)
    generator = torch.Generator().manual_seed(seed)

    # Optional train-only teacher warm-up.  This separates "teacher not yet
    # distilled" from "distilled teacher does not improve affinity".
    pretrain_history = []
    if teacher_normalized is not None and args.pretrain_epochs > 0:
        pre_optimizer = torch.optim.AdamW(
            model.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        for pre_epoch in range(1, args.pretrain_epochs + 1):
            model.train()
            order = train_idx[torch.randperm(len(train_idx), generator=generator)]
            running = 0.0
            batches = 0
            for start in range(0, len(order), args.batch_size):
                idx = order[start:start + args.batch_size]
                student = model.adapter(x[idx].to(args.device))
                tb = teacher_normalized[idx].to(args.device)
                align_loss = (1.0 - F.cosine_similarity(student, tb, dim=-1)).mean()
                pre_optimizer.zero_grad(set_to_none=True)
                align_loss.backward()
                pre_optimizer.step()
                running += float(align_loss.detach())
                batches += 1
            pretrain_history.append({"epoch": pre_epoch, "train_alignment_loss": running / batches})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Protected selector: epoch zero means no auxiliary branch at all.
    best_mse = baseline_mse
    best_state = None
    best_epoch = 0
    stale = 0
    history = [{"epoch": 0, "val_mse": baseline_mse, "enabled": False}]

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        running_mse = 0.0
        running_align = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            xb = x[idx].to(args.device)
            gp = frozen_prediction[idx].to(args.device)
            y = target[idx].to(args.device)
            pred, student = model(xb, gp)
            affinity_loss = F.mse_loss(pred, y)
            if teacher_normalized is None:
                align_loss = torch.zeros((), device=args.device)
            else:
                tb = teacher_normalized[idx].to(args.device)
                align_loss = (1.0 - F.cosine_similarity(student, tb, dim=-1)).mean()
            loss = affinity_loss + args.align_weight * align_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_mse += float(affinity_loss.detach())
            running_align += float(align_loss.detach())
            batches += 1

        model.eval()
        with torch.no_grad():
            val_prediction, _ = model(
                x[val_idx].to(args.device), frozen_prediction[val_idx].to(args.device)
            )
            val_prediction = val_prediction.cpu()
            val_mse = float((val_prediction - target[val_idx]).square().mean())
        history.append({
            "epoch": epoch,
            "val_mse": val_mse,
            "train_affinity_mse": running_mse / batches,
            "train_alignment_loss": running_align / batches,
            "enabled": True,
        })
        if val_mse < best_mse - args.minimum_improvement:
            best_mse = val_mse
            best_state = cpu_state(model)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    enabled = best_state is not None
    if enabled:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction, _ = model(
                x[val_idx].to(args.device), frozen_prediction[val_idx].to(args.device)
            )
            prediction = prediction.cpu()
    else:
        # Bit-for-bit baseline fallback; no auxiliary computation is needed.
        prediction = frozen_prediction[val_idx].clone()

    result = metrics(prediction, target[val_idx])
    result.update({
        "condition": condition,
        "teacher_key": teacher_key,
        "seed": seed,
        "enabled": enabled,
        "best_epoch": best_epoch,
        "absolute_mse_improvement": baseline_mse - result["mse"],
        "relative_mse_improvement": (baseline_mse - result["mse"]) / baseline_mse,
        "delta_mean": float((prediction - frozen_prediction[val_idx]).mean()),
        "delta_std": float((prediction - frozen_prediction[val_idx]).std()),
        "pretrain_alignment_initial": (
            pretrain_history[0]["train_alignment_loss"] if pretrain_history else None
        ),
        "pretrain_alignment_final": (
            pretrain_history[-1]["train_alignment_loss"] if pretrain_history else None
        ),
        "history": history,
    })
    checkpoint = {
        "enabled": enabled,
        "condition": condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "model_state_dict": best_state,
        "input_mean": x_mean,
        "input_std": x_std,
        "teacher_required_at_inference": False,
        "pretrain_history": pretrain_history,
    }
    return result, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--align-weight", type=float, default=0.05)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--teacher-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    args = parser.parse_args()

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    teacher_data = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
    if global_data["drug_id"] != teacher_data["drug_id"]:
        raise RuntimeError("global/teacher drug row identity mismatch")
    if global_data["protein_id"] != teacher_data["protein_id"]:
        raise RuntimeError("global/teacher protein row identity mismatch")
    pair_feature = global_data["pair_feature"].float()
    frozen_prediction = global_data["prediction"].float()
    target = global_data["label"].float()
    if pair_feature.shape != (len(target), 256):
        raise RuntimeError(f"unexpected pair feature shape: {tuple(pair_feature.shape)}")

    split = json.loads(args.split.read_text())
    train_idx = torch.tensor(split["train_indices"], dtype=torch.long)
    val_idx = torch.tensor(split["val_indices"], dtype=torch.long)
    # Intentionally never materialize split['test_indices'].
    baseline = metrics(frozen_prediction[val_idx], target[val_idx])
    baseline.update({"validation_rows": len(val_idx), "test_rows_accessed": 0})

    conditions = {
        "mse_only": None,
        "random_teacher": "random",
        "pdbbind_teacher": "pretrained",
        "atom_shuffle_teacher": "atom_shuffle",
        "mismatch_teacher": "mismatch",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"global_frozen": baseline, "runs": []}
    for condition, teacher_key in conditions.items():
        teacher = None if teacher_key is None else teacher_data[teacher_key].float()
        if teacher is not None and teacher.shape != (len(target), args.teacher_dim):
            raise RuntimeError(f"unexpected {teacher_key} teacher shape: {tuple(teacher.shape)}")
        for seed in args.seeds:
            result, checkpoint = fit(
                condition, teacher_key, seed, pair_feature, frozen_prediction,
                teacher, target, train_idx, val_idx, baseline["mse"], args,
            )
            results["runs"].append(result)
            torch.save(checkpoint, args.output_dir / f"{condition}_seed{seed}.pt")
            print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)

    grouped = {}
    for condition in conditions:
        rows = [row for row in results["runs"] if row["condition"] == condition]
        grouped[condition] = {
            "mse_mean": float(np.mean([row["mse"] for row in rows])),
            "mse_std": float(np.std([row["mse"] for row in rows], ddof=1)),
            "relative_improvement_mean": float(np.mean([row["relative_mse_improvement"] for row in rows])),
            "enabled_seeds": sum(row["enabled"] for row in rows),
            "improving_seeds": sum(row["mse"] < baseline["mse"] for row in rows),
        }

    pdb = grouped["pdbbind_teacher"]
    controls = [grouped["mse_only"], grouped["random_teacher"], grouped["mismatch_teacher"]]
    go_no_go = {
        "protected_no_seed_worse_than_baseline": all(
            row["mse"] <= baseline["mse"] + 1e-12 for row in results["runs"]
        ),
        "pdbbind_at_least_half_percent_mean_gain": pdb["relative_improvement_mean"] >= 0.005,
        "pdbbind_improves_at_least_two_of_three": pdb["improving_seeds"] >= 2,
        "pdbbind_better_than_mse_random_mismatch_controls": all(
            pdb["mse_mean"] < control["mse_mean"] for control in controls
        ),
        "atom_shuffle_does_not_reproduce_pdbbind_gain": (
            grouped["atom_shuffle_teacher"]["relative_improvement_mean"]
            < 0.5 * pdb["relative_improvement_mean"]
        ),
    }
    go_no_go["pass"] = all(go_no_go.values())
    results["summary"] = grouped
    results["go_no_go"] = go_no_go
    results["protocol"] = {
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "test_rows_accessed": 0,
        "frozen_main_path": True,
        "epoch_zero_exact_baseline_candidate": True,
        "teacher_required_at_inference": False,
        "alignment": "one_minus_cosine",
        "align_weight": args.align_weight,
        "train_only_adapter_pretrain_epochs": args.pretrain_epochs,
        "minimum_validation_mse_improvement": args.minimum_improvement,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "patience": args.patience,
    }
    (args.output_dir / "auxiliary_results.json").write_text(json.dumps(results, indent=2))
    print("FINAL", json.dumps({
        "baseline": baseline,
        "summary": grouped,
        "go_no_go": go_no_go,
    }, indent=2))


if __name__ == "__main__":
    main()
