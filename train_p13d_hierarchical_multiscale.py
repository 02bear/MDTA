from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.collate_p13d_hierarchical import (
    mdta_collate_fn_p13d_hierarchical,
    move_batch_to_device_hierarchical,
)
from datasets.davis_dataset_p13d_hierarchical import DavisDatasetP13DHierarchical
from models.model_p13d_hierarchical_multiscale import HierarchicalMultiScaleDTA


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class FenwickTree:
    def __init__(self, n): self.n, self.tree = n, np.zeros(n + 1, dtype=np.int64)
    def update(self, i):
        while i <= self.n: self.tree[i] += 1; i += i & -i
    def query(self, i):
        value = 0
        while i > 0: value += self.tree[i]; i -= i & -i
        return value


def cindex(y, p):
    ranks = {value: i + 1 for i, value in enumerate(np.unique(p))}
    order = np.argsort(y, kind="mergesort"); y, p = y[order], p[order]
    tree, previous, concordant, comparable, start = FenwickTree(len(ranks)), 0, 0.0, 0.0, 0
    while start < len(y):
        end = start
        while end < len(y) and y[end] == y[start]: end += 1
        for k in range(start, end):
            rank = ranks[p[k]]; less = tree.query(rank - 1); equal = tree.query(rank) - less
            concordant += less + 0.5 * equal; comparable += previous
        for k in range(start, end): tree.update(ranks[p[k]]); previous += 1
        start = end
    return float(concordant / comparable) if comparable else 0.0


def rankdata(values):
    order = np.argsort(values, kind="mergesort"); ranks = np.empty(len(values), dtype=float); i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]: j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0; i = j
    return ranks


def rm2(y, p):
    yc, pc = y - y.mean(), p - p.mean()
    denom = np.sum(yc * yc) * np.sum(pc * pc)
    r2 = float(np.sum(yc * pc) ** 2 / denom) if denom else 0.0
    pdenom = np.sum(p * p); k = np.sum(y * p) / pdenom if pdenom else 0.0
    ydenom = np.sum((y - y.mean()) ** 2)
    r02 = float(1 - np.sum((y - k * p) ** 2) / ydenom) if ydenom else 0.0
    return float(r2 * (1 - math.sqrt(abs(r2 ** 2 - r02 ** 2))))


def metrics(y, p, drug_ids=None):
    error = p - y; mse = float(np.mean(error ** 2))
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else 0.0
    yr, pr = rankdata(y), rankdata(p)
    spear = float(np.corrcoef(yr, pr)[0, 1]) if len(y) > 1 and np.std(yr) and np.std(pr) else 0.0
    denom = np.sum((y - y.mean()) ** 2)
    result = {
        "mse": mse, "rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(error))),
        "ci": cindex(y, p), "rm2": rm2(y, p), "pearson": corr,
        "spearman": spear, "r2": float(1 - np.sum(error ** 2) / denom) if denom else 0.0,
        "bias": float(np.mean(error)),
    }
    if drug_ids is not None:
        per_drug_squared_errors = {}
        for drug_id, squared_error in zip(drug_ids, error ** 2):
            per_drug_squared_errors.setdefault(str(drug_id), []).append(float(squared_error))
        per_drug_mse = np.asarray([
            np.mean(values) for values in per_drug_squared_errors.values()
        ])
        result["drug_macro_mse"] = float(per_drug_mse.mean())
        result["drug_median_mse"] = float(np.median(per_drug_mse))
        result["drug_worst_mse"] = float(per_drug_mse.max())
        result["num_drugs"] = len(per_drug_squared_errors)
    return result


def optimal_residual_gate(prediction_before, proposed_delta, target):
    error = prediction_before.float() - target.float()
    delta = proposed_delta.float()
    return (-(error * delta) / (delta.square() + 1e-6)).clamp(0.0, 1.0).detach()


def masked_mean(value, mask):
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def scale_losses(details, target):
    # Loss/credit arithmetic stays in FP32 under mixed precision.
    pred = details["pred"].float()
    target = target.float()
    base_error = (details["base_pred"].float() - target).pow(2)
    ar_candidate_error = (details["ar_candidate"].float() - target).pow(2)
    after_ar_error = (details["pred_after_ar"].float() - target).pow(2)
    fp_candidate_error = (details["fp_candidate"].float() - target).pow(2)
    full_error = (pred - target).pow(2)
    ar_candidate_benefit = base_error - ar_candidate_error
    fp_candidate_benefit = after_ar_error - fp_candidate_error
    ar_actual_benefit = base_error - after_ar_error
    fp_actual_benefit = after_ar_error - full_error
    optimal_ar_gate = optimal_residual_gate(
        details["base_pred"], details["ar_delta"], target,
    )
    optimal_fp_gate = optimal_residual_gate(
        details["pred_after_ar"], details["fp_delta"], target,
    )
    optimal_gates = torch.cat([optimal_ar_gate, optimal_fp_gate], dim=1)
    predicted_gates = details["scale_gates"].float()
    predicted_gate_logits = details["scale_gate_logits"].float()
    optimal_gate_logits = torch.logit(optimal_gates.clamp(0.02, 0.98))
    informative_gates = torch.cat([
        details["ar_delta"].float().abs() > 1e-3,
        details["fp_delta"].float().abs() > 1e-3,
    ], dim=1)
    per_gate_loss = F.smooth_l1_loss(
        predicted_gate_logits, optimal_gate_logits, beta=1.0, reduction="none",
    )
    gate_loss = masked_mean(per_gate_loss, informative_gates)
    delta_reg = (
        details["ar_delta_raw"].float().pow(2).mean()
        + details["fp_delta_raw"].float().pow(2).mean()
    )
    diagnostics = {
        "gate_ar": details["scale_gates"][:, 0].mean(),
        "gate_fp": details["scale_gates"][:, 1].mean(),
        "benefit_ar": ar_actual_benefit.mean(),
        "benefit_fp": fp_actual_benefit.mean(),
        "candidate_benefit_ar": ar_candidate_benefit.mean(),
        "candidate_benefit_fp": fp_candidate_benefit.mean(),
        "optimal_gate_mae_ar": masked_mean(
            (predicted_gates[:, 0:1] - optimal_ar_gate).abs(),
            informative_gates[:, 0:1],
        ),
        "optimal_gate_mae_fp": masked_mean(
            (predicted_gates[:, 1:2] - optimal_fp_gate).abs(),
            informative_gates[:, 1:2],
        ),
        "optimal_gate_zero_ar": masked_mean(
            (optimal_ar_gate <= 0.05).float(), informative_gates[:, 0:1],
        ),
        "optimal_gate_zero_fp": masked_mean(
            (optimal_fp_gate <= 0.05).float(), informative_gates[:, 1:2],
        ),
        "optimal_gate_informative_ar": informative_gates[:, 0].float().mean(),
        "optimal_gate_informative_fp": informative_gates[:, 1].float().mean(),
        "improve_rate_ar": (ar_actual_benefit > 0).float().mean(),
        "improve_rate_fp": (fp_actual_benefit > 0).float().mean(),
        "abs_delta_ar": details["ar_delta"].abs().mean(),
        "abs_delta_fp": details["fp_delta"].abs().mean(),
        "abs_raw_delta_ar": details["ar_delta_raw"].abs().mean(),
        "abs_raw_delta_fp": details["fp_delta_raw"].abs().mean(),
    }
    return gate_loss, delta_reg, diagnostics


def pairwise_ranking_loss(prediction, target, minimum_gap):
    prediction = prediction.float().view(-1)
    target = target.float().view(-1)
    target_difference = target[:, None] - target[None, :]
    prediction_difference = prediction[:, None] - prediction[None, :]
    valid = torch.triu(target_difference.abs() >= minimum_gap, diagonal=1)
    signed_prediction_difference = (
        target_difference.sign() * prediction_difference
    )
    losses = F.softplus(-signed_prediction_difference)
    valid_weight = valid.to(losses.dtype)
    return (losses * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)


def require_finite(name, tensor, step):
    finite = torch.isfinite(tensor)
    if finite.all():
        return
    bad = int((~finite).sum().item())
    total = tensor.numel()
    finite_values = tensor.detach()[finite]
    value_range = (
        f"[{finite_values.min().item():.6g}, {finite_values.max().item():.6g}]"
        if finite_values.numel() else "no finite values"
    )
    raise FloatingPointError(
        f"NON_FINITE at step {step}: {name} contains {bad}/{total} NaN/Inf; "
        f"finite_range={value_range}"
    )


def run_epoch(model, loader, device, optimizer, args):
    training = optimizer is not None; model.train(training)
    predictions, targets, drug_ids, sums, count = [], [], [], {}, 0
    for step, batch in enumerate(loader, 1):
        batch = move_batch_to_device_hierarchical(batch, device, non_blocking=True)
        target = batch["label"]
        amp_dtype = torch.bfloat16
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=amp_dtype,
            enabled=args.amp and device.type == "cuda",
        ):
            details = model(batch, return_details=True)
            check_finite = (
                args.finite_check_interval > 0
                and (step == 1 or step % args.finite_check_interval == 0)
            )
            if check_finite:
                for name in (
                    "base_pred", "ar_delta_raw", "fp_delta_raw", "ar_delta",
                    "fp_delta", "scale_gate_logits", "scale_gates", "pred",
                ):
                    require_finite(name, details[name], step)
            prediction = details["pred"].float()
            if not training and args.prediction_floor > 0:
                prediction = prediction.clamp_min(args.prediction_floor)
            mse_loss = F.mse_loss(prediction, target.float())
            gate_loss, delta_reg, diagnostics = scale_losses(details, target)
            ranking_loss = pairwise_ranking_loss(
                details["pred"], target, args.ranking_minimum_gap,
            )
            fp_gate_sparsity = details["scale_gates"][:, 1].float().mean()
            loss = (
                mse_loss
                + args.gate_supervision_weight * gate_loss
                + args.delta_l2_weight * delta_reg
                + args.ranking_weight * ranking_loss
                + args.fp_gate_sparsity_weight * fp_gate_sparsity
            )
            if check_finite:
                require_finite("total_loss", loss, step)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True,
                )
                grad_clipped = (grad_norm_tensor > args.grad_clip).float()
                optimizer.step()
            else:
                grad_norm_tensor = loss.new_zeros(())
                grad_clipped = loss.new_zeros(())
        n = target.size(0); count += n
        values = {"objective": loss.detach(), "mse_loss": mse_loss.detach(),
                  "gate_loss": gate_loss.detach(), "delta_l2": delta_reg.detach(),
                  "ranking_loss": ranking_loss.detach(),
                  "fp_gate_sparsity": fp_gate_sparsity.detach(),
                  "grad_clipped_rate": grad_clipped.detach(),
                  "grad_norm": grad_norm_tensor.detach(), **{
                      key: value.detach() for key, value in diagnostics.items()
                  }}
        for key, value in values.items():
            value = value.float()
            sums[key] = sums.get(
                key, torch.zeros((), dtype=torch.float32, device=value.device),
            ) + value * n
        predictions.append(prediction.detach().view(-1))
        targets.append(target.detach().float().view(-1))
        drug_ids.extend(batch["drug_id"])
        if training and args.log_interval > 0 and step % args.log_interval == 0:
            print(f"  STEP {step:05d}/{len(loader):05d} | MSE={mse_loss.item():.6f} | "
                  f"GATE_AR={diagnostics['gate_ar'].item():.3f} | "
                  f"GATE_FP={diagnostics['gate_fp'].item():.3f} | "
                  f"GRAD={grad_norm_tensor.item():.3f}")
    p = torch.cat(predictions).cpu().numpy()
    y = torch.cat(targets).cpu().numpy()
    result = metrics(y, p, drug_ids)
    result.update({key: value.item() / count for key, value in sums.items()})
    return result, y, p


def fmt_metrics(prefix, value):
    return (f"{prefix} MSE={value['mse']:.6f} | RMSE={value['rmse']:.6f} | "
            f"CI={value['ci']:.6f} | RM2={value['rm2']:.6f} | MAE={value['mae']:.6f} | "
            f"R2={value['r2']:.6f} | PEARSON={value['pearson']:.6f} | SPEARMAN={value['spearman']:.6f}")


def save_checkpoint(path, model, optimizer, scheduler, epoch, train, val, args):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train, "val_metrics": val, "args": vars(args)}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", default="data/processed/davis/drug_1d_chemberta2")
    parser.add_argument("--protein_1d_dir", default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_3d_dir", default="data/processed/davis/protein_3d_gvp")
    parser.add_argument("--drug_atom_v2_dir", default="data/processed/davis/drug_atom_features_v2")
    parser.add_argument("--protein_residue_v2_dir", default="data/processed/davis/protein_residue_features_v2")
    parser.add_argument("--split_json", default="data/splits/davis_fixed_split_size2.json")
    parser.add_argument("--output_dir", default="outputs/HierarchicalMultiScale/davis/run1")
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2); parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--interaction_heads", type=int, default=4); parser.add_argument("--region_rounds", type=int, default=2)
    parser.add_argument("--max_fragments", type=int, default=16); parser.add_argument("--max_atoms_per_fragment", type=int, default=32)
    parser.add_argument("--max_pockets", type=int, default=3); parser.add_argument("--max_residues_per_pocket", type=int, default=32)
    parser.add_argument("--gate_supervision_weight", type=float, default=0.5)
    parser.add_argument("--fp_gate_sparsity_weight", type=float, default=1e-2)
    parser.add_argument("--delta_l2_weight", type=float, default=1e-2); parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--ranking_weight", type=float, default=0.01)
    parser.add_argument("--ranking_minimum_gap", type=float, default=0.5)
    parser.add_argument("--prediction_floor", type=float, default=0.0,
                        help="Validation/test-only calibrated lower prediction bound; 0 disables it.")
    parser.add_argument("--delta_max", type=float, default=1.0,
                        help="Maximum absolute AR/FP residual correction.")
    parser.add_argument("--ar_gate_epsilon", type=float, default=0.05)
    parser.add_argument("--fp_gate_epsilon", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=30); parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--selection_drug_median_weight", type=float, default=0.3,
                        help="Weight of median per-drug MSE in validation checkpoint selection.")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--finite_check_interval", type=int, default=200,
                        help="Check detailed forward tensors every N steps; gradients are checked every step.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA autocast.")
    parser.add_argument("--amp_dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--allow_overwrite", action="store_true",
                        help="Allow an existing output directory containing checkpoints to be reused.")
    args = parser.parse_args()
    if not 0.0 <= args.selection_drug_median_weight <= 1.0:
        raise ValueError("--selection_drug_median_weight must be in [0, 1]")
    if args.ranking_weight < 0 or args.gate_supervision_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    set_seed(args.seed)
    output = Path(args.output_dir)
    if (output / "checkpoint_last.pt").exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Output directory already contains a run: {output}. Choose a new --output_dir "
            "or pass --allow_overwrite explicitly."
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    dataset = DavisDatasetP13DHierarchical(
        pairs_csv=args.pairs_csv, drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir, protein_3d_dir=args.protein_3d_dir,
        use_drug_2d=False, use_drug_3d=False,
        drug_atom_v2_dir=args.drug_atom_v2_dir, protein_residue_v2_dir=args.protein_residue_v2_dir,
    )
    split = json.loads(Path(args.split_json).read_text())
    train_set, val_set = Subset(dataset, split["train_indices"]), Subset(dataset, split["val_indices"])
    test_indices = split.get("test_indices", [])
    test_set = Subset(dataset, test_indices) if test_indices else None
    collate = partial(
        mdta_collate_fn_p13d_hierarchical,
        max_fragments=args.max_fragments,
        max_atoms_per_fragment=args.max_atoms_per_fragment,
        max_pockets=args.max_pockets,
        max_residues_per_pocket=args.max_residues_per_pocket,
    )
    loader_options = {
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, **loader_options)
    test_loader = (DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers,
                              collate_fn=collate, pin_memory=True,
                              persistent_workers=args.num_workers > 0) if test_set is not None else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalMultiScaleDTA(hidden_dim=args.hidden_dim, dropout=args.dropout,
                                      interaction_heads=args.interaction_heads, region_rounds=args.region_rounds,
                                      delta_max=args.delta_max,
                                      ar_gate_epsilon=args.ar_gate_epsilon,
                                      fp_gate_epsilon=args.fp_gate_epsilon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    print(f"DEVICE={device} | TRAIN={len(train_set)} | VAL={len(val_set)} | "
          f"TEST={len(test_set) if test_set is not None else 0} | PARAMETERS={sum(p.numel() for p in model.parameters()):,}")
    print("MODEL=Global + vectorized AR + FP propagation + analytic optimal-gate supervision")
    print(f"EVAL_CALIBRATION=prediction_floor={args.prediction_floor:.4f} (validation/test only)")

    history_path = output / "epoch_metrics.jsonl"; history_path.write_text("")
    csv_path = output / "epoch_metrics.csv"
    csv_fields = ["epoch", "seconds", "lr", "selection_score", "best_selection_score",
                  "improved", "best_epoch"] + [
        f"{split_name}_{key}" for split_name in ("train", "val", "best_val")
        for key in ("mse", "rmse", "ci", "rm2", "mae", "r2", "pearson", "spearman",
                    "drug_macro_mse", "drug_median_mse", "drug_worst_mse",
                    "gate_ar", "gate_fp", "benefit_ar", "benefit_fp",
                    "candidate_benefit_ar", "candidate_benefit_fp",
                    "optimal_gate_mae_ar", "optimal_gate_mae_fp",
                    "optimal_gate_zero_ar", "optimal_gate_zero_fp",
                    "optimal_gate_informative_ar", "optimal_gate_informative_fp",
                    "improve_rate_ar", "improve_rate_fp", "grad_clipped_rate",
                    "abs_delta_ar", "abs_delta_fp", "abs_raw_delta_ar", "abs_raw_delta_fp")
    ]
    with csv_path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=csv_fields).writeheader()
    best, best_epoch, best_selection_score, stale = None, -1, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        started = time.time(); train, _, _ = run_epoch(model, train_loader, device, optimizer, args)
        val, val_y, val_p = run_epoch(model, val_loader, device, None, args)
        selection_score = (
            (1.0 - args.selection_drug_median_weight) * val["mse"]
            + args.selection_drug_median_weight * val["drug_median_mse"]
        )
        scheduler.step(selection_score)
        improved = selection_score < best_selection_score - args.min_delta
        if improved:
            best, best_epoch, best_selection_score, stale = (
                dict(val), epoch, selection_score, 0,
            )
            save_checkpoint(output / "checkpoint_best_selection.pt", model, optimizer, scheduler, epoch, train, val, args)
            np.savez(output / "best_val_predictions.npz", target=val_y, prediction=val_p)
            (output / "best_metrics.json").write_text(json.dumps({
                "epoch": epoch, "selection_score": selection_score,
                "selection_drug_median_weight": args.selection_drug_median_weight,
                "train": train, "val": val,
            }, indent=2))
        else: stale += 1
        save_checkpoint(output / "checkpoint_last.pt", model, optimizer, scheduler, epoch, train, val, args)
        row = {"epoch": epoch, "seconds": time.time() - started, "lr": optimizer.param_groups[0]["lr"],
               "selection_score": selection_score, "best_selection_score": best_selection_score,
               "improved": improved, "best_epoch": best_epoch, "train": train, "val": val, "best_val": best}
        with history_path.open("a") as handle: handle.write(json.dumps(row) + "\n")
        flat = {"epoch": epoch, "seconds": row["seconds"], "lr": row["lr"],
                "selection_score": selection_score, "best_selection_score": best_selection_score,
                "improved": improved, "best_epoch": best_epoch}
        for split_name, values in (("train", train), ("val", val), ("best_val", best)):
            for key in ("mse", "rmse", "ci", "rm2", "mae", "r2", "pearson", "spearman",
                        "drug_macro_mse", "drug_median_mse", "drug_worst_mse",
                        "gate_ar", "gate_fp", "benefit_ar", "benefit_fp",
                        "candidate_benefit_ar", "candidate_benefit_fp",
                        "optimal_gate_mae_ar", "optimal_gate_mae_fp",
                        "optimal_gate_zero_ar", "optimal_gate_zero_fp",
                        "optimal_gate_informative_ar", "optimal_gate_informative_fp",
                        "improve_rate_ar", "improve_rate_fp", "grad_clipped_rate",
                        "abs_delta_ar", "abs_delta_fp", "abs_raw_delta_ar", "abs_raw_delta_fp"):
                flat[f"{split_name}_{key}"] = values.get(key, "")
        with csv_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=csv_fields).writerow(flat)
        print(f"\n[EPOCH {epoch:03d}/{args.epochs:03d}] TIME={row['seconds']:.1f}s | LR={row['lr']:.2e} | {'NEW_BEST' if improved else 'no improvement'}")
        print(fmt_metrics("CURRENT TRAIN |", train)); print(fmt_metrics("CURRENT VAL   |", val))
        print(fmt_metrics(f"BEST VAL (epoch {best_epoch:03d}) |", best))
        print(f"SELECTION | CURRENT={selection_score:.6f} BEST={best_selection_score:.6f} | "
              f"VAL_MEDIAN_DRUG_MSE={val['drug_median_mse']:.6f} "
              f"VAL_WORST_DRUG_MSE={val['drug_worst_mse']:.6f} "
              f"WEIGHT={args.selection_drug_median_weight:.2f}")
        print(f"SCALE CURRENT | AR_GATE={val['gate_ar']:.4f} FP_GATE={val['gate_fp']:.4f} | "
              f"AR_BENEFIT={val['benefit_ar']:+.6f} FP_BENEFIT={val['benefit_fp']:+.6f} | "
              f"AR_CAND={val['candidate_benefit_ar']:+.6f} FP_CAND={val['candidate_benefit_fp']:+.6f} | "
              f"AR_DELTA={val['abs_delta_ar']:.6f} FP_DELTA={val['abs_delta_fp']:.6f} | "
              f"AR_RAW={val['abs_raw_delta_ar']:.6f} FP_RAW={val['abs_raw_delta_fp']:.6f} | "
              f"STALE={stale}/{args.patience}")
        print(f"GATE TARGET | AR_MAE={val['optimal_gate_mae_ar']:.4f} "
              f"FP_MAE={val['optimal_gate_mae_fp']:.4f} | "
              f"AR_ZERO={val['optimal_gate_zero_ar']:.3f} FP_ZERO={val['optimal_gate_zero_fp']:.3f} | "
              f"AR_INFO={val['optimal_gate_informative_ar']:.3f} "
              f"FP_INFO={val['optimal_gate_informative_fp']:.3f} | "
              f"AR_IMPROVE={val['improve_rate_ar']:.3f} FP_IMPROVE={val['improve_rate_fp']:.3f} | "
              f"TRAIN_CLIPPED={train['grad_clipped_rate']:.3f}")
        if args.patience > 0 and stale >= args.patience:
            print(f"EARLY_STOP | best_epoch={best_epoch} "
                  f"best_selection={best_selection_score:.6f} best_val_mse={best['mse']:.6f}"); break
    print(fmt_metrics(f"TRAINING FINISHED | BEST VAL (epoch {best_epoch:03d}) |", best))
    final_test = None
    if test_loader is not None:
        checkpoint = torch.load(output / "checkpoint_best_selection.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_test, test_y, test_p = run_epoch(model, test_loader, device, None, args)
        np.savez(output / "final_test_predictions.npz", target=test_y, prediction=test_p)
        (output / "final_test_metrics.json").write_text(json.dumps({
            "selected_by": "minimum weighted overall/median-per-drug validation MSE",
            "best_epoch": best_epoch,
            "selection_score": best_selection_score,
            "selection_drug_median_weight": args.selection_drug_median_weight,
            "test": final_test,
        }, indent=2), encoding="utf-8")
        print(fmt_metrics(f"FINAL TEST (best epoch {best_epoch:03d}) |", final_test))
    (output / "training_summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "best_val": best, "final_test": final_test,
        "best_selection_score": best_selection_score,
        "selection_drug_median_weight": args.selection_drug_median_weight,
        "epochs_completed": epoch, "stopped_early": stale >= args.patience > 0,
        "best_checkpoint": "checkpoint_best_selection.pt",
        "last_checkpoint": "checkpoint_last.pt",
    }, indent=2), encoding="utf-8")
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
