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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.collate_p13d_hierarchical import (
    mdta_collate_fn_p13d_hierarchical,
    move_batch_to_device_hierarchical,
)
from datasets.davis_dataset_p13d_hierarchical import (
    DavisDatasetP13DHierarchical,
)
from models.model_p13d_hurdle_global_ar import HurdleGlobalARDTA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def update(self, i):
        while i <= self.n:
            self.tree[i] += 1
            i += i & -i

    def query(self, i):
        value = 0
        while i > 0:
            value += self.tree[i]
            i -= i & -i
        return value


def cindex(y, p):
    ranks = {
        value: i + 1
        for i, value in enumerate(np.unique(p))
    }
    order = np.argsort(y, kind="mergesort")
    y, p = y[order], p[order]
    tree = FenwickTree(len(ranks))
    previous = 0
    concordant = 0.0
    comparable = 0.0
    start = 0
    while start < len(y):
        end = start
        while end < len(y) and y[end] == y[start]:
            end += 1
        for k in range(start, end):
            rank = ranks[p[k]]
            less = tree.query(rank - 1)
            equal = tree.query(rank) - less
            concordant += less + 0.5 * equal
            comparable += previous
        for k in range(start, end):
            tree.update(ranks[p[k]])
            previous += 1
        start = end
    return float(concordant / comparable) if comparable else 0.0


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while (
            j < len(values)
            and values[order[j]] == values[order[i]]
        ):
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def rm2(y, p):
    yc, pc = y - y.mean(), p - p.mean()
    denom = np.sum(yc * yc) * np.sum(pc * pc)
    r2 = (
        float(np.sum(yc * pc) ** 2 / denom)
        if denom
        else 0.0
    )
    pdenom = np.sum(p * p)
    k = np.sum(y * p) / pdenom if pdenom else 0.0
    ydenom = np.sum((y - y.mean()) ** 2)
    r02 = (
        float(1 - np.sum((y - k * p) ** 2) / ydenom)
        if ydenom
        else 0.0
    )
    return float(
        r2 * (1 - math.sqrt(abs(r2 ** 2 - r02 ** 2))),
    )


def binary_metrics(target, probability, bins=10):
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=float)
    positive = int(target.sum())
    negative = int(len(target) - positive)
    if positive and negative:
        probability_ranks = rankdata(probability)
        rank_sum_positive = probability_ranks[target == 1].sum()
        auroc = (
            rank_sum_positive
            - positive * (positive + 1) / 2.0
        ) / (positive * negative)
    else:
        auroc = 0.0

    order = np.argsort(-probability, kind="mergesort")
    ordered_target = target[order]
    cumulative_positive = np.cumsum(ordered_target)
    precision = cumulative_positive / np.arange(1, len(target) + 1)
    auprc = (
        float(precision[ordered_target == 1].mean())
        if positive
        else 0.0
    )
    prediction = probability >= 0.5
    tp = int(np.sum(prediction & (target == 1)))
    tn = int(np.sum(~prediction & (target == 0)))
    sensitivity = tp / positive if positive else 0.0
    specificity = tn / negative if negative else 0.0

    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        if i == bins - 1:
            mask = (
                (probability >= edges[i])
                & (probability <= edges[i + 1])
            )
        else:
            mask = (
                (probability >= edges[i])
                & (probability < edges[i + 1])
            )
        if mask.any():
            ece += (
                mask.mean()
                * abs(
                    probability[mask].mean()
                    - target[mask].mean()
                )
            )

    return {
        "active_auroc": float(auroc),
        "active_auprc": float(auprc),
        "active_brier": float(
            np.mean((probability - target) ** 2),
        ),
        "active_ece": float(ece),
        "active_accuracy": float(np.mean(prediction == target)),
        "active_balanced_accuracy": float(
            0.5 * (sensitivity + specificity),
        ),
        "active_sensitivity": float(sensitivity),
        "active_specificity": float(specificity),
        "active_fraction": float(target.mean()),
        "predicted_active_fraction": float(prediction.mean()),
        "active_prob_positive": (
            float(probability[target == 1].mean())
            if positive
            else 0.0
        ),
        "active_prob_negative": (
            float(probability[target == 0].mean())
            if negative
            else 0.0
        ),
    }


def regression_metrics(
    y,
    p,
    drug_ids=None,
    active_probability=None,
    conditional_residual=None,
    global_hurdle_prediction=None,
    ar_correction=None,
    affinity_floor=5.0,
    strong_threshold=7.0,
):
    error = p - y
    mse = float(np.mean(error ** 2))
    corr = (
        float(np.corrcoef(y, p)[0, 1])
        if np.std(y) and np.std(p)
        else 0.0
    )
    yr, pr = rankdata(y), rankdata(p)
    spear = (
        float(np.corrcoef(yr, pr)[0, 1])
        if len(y) > 1 and np.std(yr) and np.std(pr)
        else 0.0
    )
    denom = np.sum((y - y.mean()) ** 2)
    result = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "ci": cindex(y, p),
        "rm2": rm2(y, p),
        "pearson": corr,
        "spearman": spear,
        "r2": (
            float(1 - np.sum(error ** 2) / denom)
            if denom
            else 0.0
        ),
        "bias": float(np.mean(error)),
    }

    floor_mask = y <= affinity_floor + 1e-7
    active_mask = y > affinity_floor + 1e-7
    strong_mask = y >= strong_threshold

    def masked_mse(mask):
        return (
            float(np.mean(error[mask] ** 2))
            if mask.any()
            else 0.0
        )

    def masked_bias(mask):
        return (
            float(np.mean(error[mask]))
            if mask.any()
            else 0.0
        )

    result.update(
        {
            "floor_mse": masked_mse(floor_mask),
            "floor_bias": masked_bias(floor_mask),
            "active_mse": masked_mse(active_mask),
            "active_bias": masked_bias(active_mask),
            "strong_mse": masked_mse(strong_mask),
            "strong_bias": masked_bias(strong_mask),
            "strong_count": int(strong_mask.sum()),
        },
    )

    if (
        active_probability is not None
        and conditional_residual is not None
        and global_hurdle_prediction is not None
        and ar_correction is not None
    ):
        active_target = active_mask.astype(np.int64)
        result.update(
            binary_metrics(active_target, active_probability),
        )
        residual_target = np.clip(y - affinity_floor, 0.0, None)
        result["conditional_mse"] = (
            float(
                np.mean(
                    (
                        conditional_residual[active_mask]
                        - residual_target[active_mask]
                    )
                    ** 2,
                ),
            )
            if active_mask.any()
            else 0.0
        )
        global_error = global_hurdle_prediction - y
        result["global_hurdle_mse"] = float(
            np.mean(global_error ** 2),
        )
        result["ar_benefit"] = (
            result["global_hurdle_mse"] - result["mse"]
        )
        result["ar_correction_abs"] = float(
            np.mean(np.abs(ar_correction)),
        )

    if drug_ids is not None:
        squared_by_drug = {}
        target_by_drug = {}
        prediction_by_drug = {}
        for drug_id, target, prediction, squared_error in zip(
            drug_ids,
            y,
            p,
            error ** 2,
        ):
            key = str(drug_id)
            squared_by_drug.setdefault(key, []).append(
                float(squared_error),
            )
            target_by_drug.setdefault(key, []).append(float(target))
            prediction_by_drug.setdefault(key, []).append(
                float(prediction),
            )
        per_drug_mse = np.asarray(
            [
                np.mean(values)
                for values in squared_by_drug.values()
            ],
        )
        per_drug_mean_error = np.asarray(
            [
                np.mean(prediction_by_drug[key])
                - np.mean(target_by_drug[key])
                for key in squared_by_drug
            ],
        )
        result.update(
            {
                "drug_macro_mse": float(per_drug_mse.mean()),
                "drug_median_mse": float(np.median(per_drug_mse)),
                "drug_worst_mse": float(per_drug_mse.max()),
                "drug_mean_mse": float(
                    np.mean(per_drug_mean_error ** 2),
                ),
                "num_drugs": len(squared_by_drug),
            },
        )
    return result


def require_finite(name, tensor, step):
    finite = torch.isfinite(tensor)
    if finite.all():
        return
    bad = int((~finite).sum().item())
    finite_values = tensor.detach()[finite]
    value_range = (
        f"[{finite_values.min().item():.6g}, "
        f"{finite_values.max().item():.6g}]"
        if finite_values.numel()
        else "no finite values"
    )
    raise FloatingPointError(
        f"NON_FINITE at step {step}: {name} contains "
        f"{bad}/{tensor.numel()} NaN/Inf; "
        f"finite_range={value_range}",
    )


def conditional_mse_loss(prediction, target, active_mask):
    if active_mask.any():
        return F.mse_loss(
            prediction[active_mask],
            target[active_mask],
        )
    return prediction.sum() * 0.0


def masked_tensor_mean(values, mask):
    if mask.any():
        return values[mask].mean()
    return values.sum() * 0.0


def run_epoch(model, loader, device, optimizer, args):
    training = optimizer is not None
    model.train(training)
    predictions = []
    global_predictions = []
    targets = []
    active_probabilities = []
    conditional_residuals = []
    ar_corrections = []
    drug_ids = []
    sums = {}
    count = 0

    for step, batch in enumerate(loader, 1):
        batch = move_batch_to_device_hierarchical(
            batch, device, non_blocking=True,
        )
        target = batch["label"].float()
        active_target = (
            target > args.affinity_floor
        ).float()
        active_mask = active_target.bool().view(-1)
        residual_target = (
            target - args.affinity_floor
        ).clamp_min(0.0)

        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.amp and device.type == "cuda",
            ),
        ):
            details = model(batch, return_details=True)
            prediction = details["pred"].float()
            active_logit = details["active_logit"].float()
            active_probability = details[
                "active_probability"
            ].float()
            conditional_residual = details[
                "conditional_residual"
            ].float()

            final_mse_loss = F.mse_loss(prediction, target)
            classification_loss = F.binary_cross_entropy_with_logits(
                active_logit,
                active_target,
            )
            conditional_loss = conditional_mse_loss(
                conditional_residual.view(-1),
                residual_target.view(-1),
                active_mask,
            )
            ar_l2_loss = details[
                "ar_residual_raw"
            ].float().pow(2).mean()
            loss = (
                final_mse_loss
                + args.classification_weight * classification_loss
                + args.conditional_weight * conditional_loss
                + args.ar_l2_weight * ar_l2_loss
            )

            check_finite = (
                args.finite_check_interval > 0
                and (
                    step == 1
                    or step % args.finite_check_interval == 0
                )
            )
            if check_finite:
                for name in (
                    "pred",
                    "global_hurdle_pred",
                    "active_logit",
                    "active_probability",
                    "global_residual_raw",
                    "conditional_residual",
                    "ar_residual_raw",
                    "ar_residual_correction",
                ):
                    require_finite(name, details[name], step)
                require_finite("total_loss", loss, step)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            else:
                grad_norm_tensor = loss.new_zeros(())

        n = target.size(0)
        count += n
        global_mse = F.mse_loss(
            details["global_hurdle_pred"].float(), target,
        )
        ar_benefit = global_mse - final_mse_loss
        prob_positive = masked_tensor_mean(
            active_probability.view(-1), active_mask,
        )
        prob_negative = masked_tensor_mean(
            active_probability.view(-1), ~active_mask,
        )
        values = {
            "objective": loss.detach(),
            "final_mse_loss": final_mse_loss.detach(),
            "classification_loss": classification_loss.detach(),
            "conditional_loss": conditional_loss.detach(),
            "ar_l2_loss": ar_l2_loss.detach(),
            "global_hurdle_mse": global_mse.detach(),
            "ar_benefit": ar_benefit.detach(),
            "active_probability_positive": prob_positive.detach(),
            "active_probability_negative": prob_negative.detach(),
            "ar_correction_abs": details[
                "ar_residual_correction"
            ].float().abs().mean().detach(),
            "grad_norm": grad_norm_tensor.detach(),
        }
        for key, value in values.items():
            value = value.float()
            sums[key] = sums.get(
                key,
                torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=value.device,
                ),
            ) + value * n

        predictions.append(prediction.detach().view(-1))
        global_predictions.append(
            details["global_hurdle_pred"].float().detach().view(-1),
        )
        targets.append(target.detach().view(-1))
        active_probabilities.append(
            active_probability.detach().view(-1),
        )
        conditional_residuals.append(
            conditional_residual.detach().view(-1),
        )
        ar_corrections.append(
            details["ar_residual_correction"]
            .float()
            .detach()
            .view(-1),
        )
        drug_ids.extend(batch["drug_id"])

        if (
            training
            and args.log_interval > 0
            and step % args.log_interval == 0
        ):
            print(
                f"  STEP {step:05d}/{len(loader):05d} | "
                f"MSE={final_mse_loss.item():.6f} | "
                f"CLS={classification_loss.item():.4f} | "
                f"COND={conditional_loss.item():.4f} | "
                f"P+={prob_positive.item():.3f} "
                f"P-={prob_negative.item():.3f} | "
                f"AR_GAIN={ar_benefit.item():+.5f} | "
                f"GRAD={grad_norm_tensor.item():.3f}",
            )

    prediction = torch.cat(predictions).cpu().numpy()
    global_prediction = torch.cat(
        global_predictions,
    ).cpu().numpy()
    target = torch.cat(targets).cpu().numpy()
    active_probability = torch.cat(
        active_probabilities,
    ).cpu().numpy()
    conditional_residual = torch.cat(
        conditional_residuals,
    ).cpu().numpy()
    ar_correction = torch.cat(ar_corrections).cpu().numpy()

    result = regression_metrics(
        target,
        prediction,
        drug_ids=drug_ids,
        active_probability=active_probability,
        conditional_residual=conditional_residual,
        global_hurdle_prediction=global_prediction,
        ar_correction=ar_correction,
        affinity_floor=args.affinity_floor,
        strong_threshold=args.strong_threshold,
    )
    result.update(
        {
            key: value.item() / count
            for key, value in sums.items()
        },
    )
    artifacts = {
        "target": target,
        "prediction": prediction,
        "global_hurdle_prediction": global_prediction,
        "active_probability": active_probability,
        "conditional_residual": conditional_residual,
        "ar_residual_correction": ar_correction,
        "drug_id": np.asarray(drug_ids, dtype=str),
    }
    return result, artifacts


def fmt_metrics(prefix, value):
    return (
        f"{prefix} MSE={value['mse']:.6f} | "
        f"RMSE={value['rmse']:.6f} | "
        f"CI={value['ci']:.6f} | "
        f"RM2={value['rm2']:.6f} | "
        f"MAE={value['mae']:.6f} | "
        f"R2={value['r2']:.6f} | "
        f"PEARSON={value['pearson']:.6f} | "
        f"SPEARMAN={value['spearman']:.6f}"
    )


def fmt_hurdle(prefix, value):
    return (
        f"{prefix} AUROC={value['active_auroc']:.4f} | "
        f"AUPRC={value['active_auprc']:.4f} | "
        f"BRIER={value['active_brier']:.4f} | "
        f"ECE={value['active_ece']:.4f} | "
        f"BACC={value['active_balanced_accuracy']:.4f} | "
        f"P+={value['active_prob_positive']:.4f} | "
        f"P-={value['active_prob_negative']:.4f}"
    )


def fmt_strata(prefix, value):
    return (
        f"{prefix} FLOOR_MSE={value['floor_mse']:.6f} "
        f"(BIAS={value['floor_bias']:+.4f}) | "
        f"ACTIVE_MSE={value['active_mse']:.6f} "
        f"(BIAS={value['active_bias']:+.4f}) | "
        f"STRONG_MSE={value['strong_mse']:.6f} "
        f"(BIAS={value['strong_bias']:+.4f})"
    )


def fmt_components(prefix, value):
    return (
        f"{prefix} GLOBAL_HURDLE_MSE="
        f"{value['global_hurdle_mse']:.6f} | "
        f"AR_BENEFIT={value['ar_benefit']:+.6f} | "
        f"COND_MSE={value['conditional_mse']:.6f} | "
        f"AR_CORR={value['ar_correction_abs']:.6f} | "
        f"DRUG_MEAN_MSE={value['drug_mean_mse']:.6f}"
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train,
    val,
    args,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_metrics": train,
            "val_metrics": val,
            "args": vars(args),
        },
        path,
    )


def save_artifacts(path, artifacts):
    np.savez(path, **artifacts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs_csv",
        default="data/raw/davis/pairs.csv",
    )
    parser.add_argument(
        "--drug_1d_dir",
        default="data/processed/davis/drug_1d_chemberta2",
    )
    parser.add_argument(
        "--protein_1d_dir",
        default="data/processed/davis/protein_1d_esm2",
    )
    parser.add_argument(
        "--protein_3d_dir",
        default="data/processed/davis/protein_3d_gvp",
    )
    parser.add_argument(
        "--drug_atom_v2_dir",
        default="data/processed/davis/drug_atom_features_v2",
    )
    parser.add_argument(
        "--protein_residue_v2_dir",
        default=(
            "data/processed/davis/protein_residue_features_v2"
        ),
    )
    parser.add_argument(
        "--split_json",
        default="data/splits/davis_fixed_split_size2.json",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/HurdleGlobalAR/davis/run1",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--residual_head",
        choices=["global", "global_ar"],
        default="global_ar",
        help="H1 uses global; H2 uses global_ar.",
    )
    parser.add_argument(
        "--affinity_floor",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--strong_threshold",
        type=float,
        default=7.0,
    )
    parser.add_argument(
        "--classification_weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--conditional_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ar_delta_max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ar_l2_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
    )

    parser.add_argument("--max_fragments", type=int, default=12)
    parser.add_argument(
        "--max_atoms_per_fragment",
        type=int,
        default=32,
    )
    parser.add_argument("--max_pockets", type=int, default=3)
    parser.add_argument(
        "--max_residues_per_pocket",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--balanced_selection_weight",
        type=float,
        default=0.3,
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--finite_check_interval",
        type=int,
        default=200,
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    if not 0.0 <= args.balanced_selection_weight <= 1.0:
        raise ValueError(
            "--balanced_selection_weight must be in [0, 1]",
        )
    if args.classification_weight < 0:
        raise ValueError("--classification_weight must be nonnegative")
    if args.conditional_weight < 0:
        raise ValueError("--conditional_weight must be nonnegative")
    if args.ar_l2_weight < 0:
        raise ValueError("--ar_l2_weight must be nonnegative")

    set_seed(args.seed)
    output = Path(args.output_dir)
    if (
        (output / "checkpoint_last.pt").exists()
        and not args.allow_overwrite
    ):
        raise FileExistsError(
            f"Output directory already contains a run: {output}. "
            "Choose a new --output_dir or pass --allow_overwrite.",
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    dataset = DavisDatasetP13DHierarchical(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.protein_3d_dir,
        use_drug_2d=False,
        use_drug_3d=False,
        drug_atom_v2_dir=args.drug_atom_v2_dir,
        protein_residue_v2_dir=args.protein_residue_v2_dir,
    )
    split = json.loads(Path(args.split_json).read_text())
    train_set = Subset(dataset, split["train_indices"])
    val_set = Subset(dataset, split["val_indices"])
    test_indices = split.get("test_indices", [])
    test_set = (
        Subset(dataset, test_indices)
        if test_indices
        else None
    )

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
    train_loader = DataLoader(
        train_set,
        args.batch_size,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_set,
        args.batch_size,
        shuffle=False,
        **loader_options,
    )
    test_loader = (
        DataLoader(
            test_set,
            args.batch_size,
            shuffle=False,
            **loader_options,
        )
        if test_set is not None
        else None
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    model = HurdleGlobalARDTA(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        affinity_floor=args.affinity_floor,
        residual_head=args.residual_head,
        ar_delta_max=args.ar_delta_max,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )

    print(
        f"DEVICE={device} | TRAIN={len(train_set)} | "
        f"VAL={len(val_set)} | "
        f"TEST={len(test_set) if test_set is not None else 0} | "
        f"PARAMETERS={sum(p.numel() for p in model.parameters()):,}",
    )
    print(
        "MODEL=Hurdle-DTA | Global active classifier + "
        f"{args.residual_head} conditional residual | "
        "prediction=floor+P(active)*residual",
    )
    print(
        f"LOSS=final_mse + {args.classification_weight:g}*BCE "
        f"+ {args.conditional_weight:g}*conditional_mse "
        f"+ {args.ar_l2_weight:g}*ar_l2",
    )
    print(
        f"THRESHOLDS=active>{args.affinity_floor:.4f} | "
        f"strong>={args.strong_threshold:.4f}",
    )

    history_path = output / "epoch_metrics.jsonl"
    history_path.write_text("")
    csv_path = output / "epoch_metrics.csv"
    metric_keys = (
        "mse",
        "rmse",
        "ci",
        "rm2",
        "mae",
        "r2",
        "pearson",
        "spearman",
        "drug_macro_mse",
        "drug_median_mse",
        "drug_worst_mse",
        "drug_mean_mse",
        "floor_mse",
        "floor_bias",
        "active_mse",
        "active_bias",
        "strong_mse",
        "strong_bias",
        "active_auroc",
        "active_auprc",
        "active_brier",
        "active_ece",
        "active_balanced_accuracy",
        "active_prob_positive",
        "active_prob_negative",
        "conditional_mse",
        "global_hurdle_mse",
        "ar_benefit",
        "ar_correction_abs",
        "objective",
        "classification_loss",
        "conditional_loss",
    )
    csv_fields = [
        "epoch",
        "seconds",
        "lr",
        "improved_mse",
        "improved_balanced",
        "best_mse_epoch",
        "best_balanced_epoch",
        "balanced_score",
        "best_balanced_score",
    ] + [
        f"{split_name}_{key}"
        for split_name in ("train", "val", "best_val")
        for key in metric_keys
    ]
    with csv_path.open("w", newline="") as handle:
        csv.DictWriter(
            handle, fieldnames=csv_fields,
        ).writeheader()

    best_mse = None
    best_mse_epoch = -1
    best_balanced = None
    best_balanced_epoch = -1
    best_balanced_score = float("inf")
    stale = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train, _ = run_epoch(
            model, train_loader, device, optimizer, args,
        )
        val, val_artifacts = run_epoch(
            model, val_loader, device, None, args,
        )
        scheduler.step(val["mse"])

        balanced_score = (
            (1.0 - args.balanced_selection_weight) * val["mse"]
            + args.balanced_selection_weight
            * val["drug_median_mse"]
        )
        improved_mse = (
            best_mse is None
            or val["mse"] < best_mse["mse"] - args.min_delta
        )
        improved_balanced = (
            balanced_score
            < best_balanced_score - args.min_delta
        )
        if improved_mse:
            best_mse = dict(val)
            best_mse_epoch = epoch
            stale = 0
            save_checkpoint(
                output / "checkpoint_best_val_mse.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                train,
                val,
                args,
            )
            save_artifacts(
                output / "best_val_predictions.npz",
                val_artifacts,
            )
            (output / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": train,
                        "val": val,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            stale += 1

        if improved_balanced:
            best_balanced = dict(val)
            best_balanced_epoch = epoch
            best_balanced_score = balanced_score
            save_checkpoint(
                output / "checkpoint_best_balanced.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                train,
                val,
                args,
            )
            save_artifacts(
                output / "best_balanced_val_predictions.npz",
                val_artifacts,
            )
            (
                output / "best_balanced_metrics.json"
            ).write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "balanced_score": balanced_score,
                        "balanced_selection_weight": (
                            args.balanced_selection_weight
                        ),
                        "train": train,
                        "val": val,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        save_checkpoint(
            output / "checkpoint_last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            train,
            val,
            args,
        )
        row = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "improved_mse": improved_mse,
            "improved_balanced": improved_balanced,
            "best_mse_epoch": best_mse_epoch,
            "best_balanced_epoch": best_balanced_epoch,
            "balanced_score": balanced_score,
            "best_balanced_score": best_balanced_score,
            "train": train,
            "val": val,
            "best_val": best_mse,
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

        flat = {
            "epoch": epoch,
            "seconds": row["seconds"],
            "lr": row["lr"],
            "improved_mse": improved_mse,
            "improved_balanced": improved_balanced,
            "best_mse_epoch": best_mse_epoch,
            "best_balanced_epoch": best_balanced_epoch,
            "balanced_score": balanced_score,
            "best_balanced_score": best_balanced_score,
        }
        for split_name, values in (
            ("train", train),
            ("val", val),
            ("best_val", best_mse),
        ):
            for key in metric_keys:
                flat[f"{split_name}_{key}"] = values.get(
                    key, "",
                )
        with csv_path.open("a", newline="") as handle:
            csv.DictWriter(
                handle, fieldnames=csv_fields,
            ).writerow(flat)

        status = " ".join(
            filter(
                None,
                [
                    (
                        "NEW_BEST_MSE"
                        if improved_mse
                        else ""
                    ),
                    (
                        "NEW_BEST_BALANCED"
                        if improved_balanced
                        else ""
                    ),
                ],
            ),
        ) or "no improvement"
        print(
            f"\n[EPOCH {epoch:03d}/{args.epochs:03d}] "
            f"TIME={row['seconds']:.1f}s | "
            f"LR={row['lr']:.2e} | {status}",
        )
        print(fmt_metrics("CURRENT TRAIN |", train))
        print(fmt_metrics("CURRENT VAL   |", val))
        print(
            fmt_metrics(
                f"BEST MSE VAL (epoch {best_mse_epoch:03d}) |",
                best_mse,
            ),
        )
        print(fmt_hurdle("HURDLE TRAIN |", train))
        print(fmt_hurdle("HURDLE VAL   |", val))
        print(fmt_strata("STRATA VAL    |", val))
        print(fmt_components("COMPONENT VAL |", val))
        print(
            f"SELECTION | BALANCED={balanced_score:.6f} "
            f"BEST={best_balanced_score:.6f} "
            f"(epoch {best_balanced_epoch:03d}) | "
            f"VAL_MEDIAN_DRUG_MSE="
            f"{val['drug_median_mse']:.6f} | "
            f"STALE={stale}/{args.patience}",
        )

        if args.patience > 0 and stale >= args.patience:
            print(
                f"EARLY_STOP | best_mse_epoch={best_mse_epoch} "
                f"best_val_mse={best_mse['mse']:.6f}",
            )
            break

    print(
        fmt_metrics(
            "TRAINING FINISHED | "
            f"BEST MSE VAL (epoch {best_mse_epoch:03d}) |",
            best_mse,
        ),
    )
    final_tests = {}
    if test_loader is not None:
        selections = (
            (
                "minimum_validation_mse",
                "checkpoint_best_val_mse.pt",
                best_mse_epoch,
            ),
            (
                "balanced_validation",
                "checkpoint_best_balanced.pt",
                best_balanced_epoch,
            ),
        )
        for (
            selection_name,
            checkpoint_name,
            selected_epoch,
        ) in selections:
            checkpoint = torch.load(
                output / checkpoint_name,
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(
                checkpoint["model_state_dict"],
            )
            test_metrics, test_artifacts = run_epoch(
                model,
                test_loader,
                device,
                None,
                args,
            )
            final_tests[selection_name] = {
                "epoch": selected_epoch,
                "checkpoint": checkpoint_name,
                "metrics": test_metrics,
            }
            save_artifacts(
                output
                / (
                    "final_test_predictions_"
                    f"{selection_name}.npz"
                ),
                test_artifacts,
            )
            print(
                fmt_metrics(
                    f"FINAL TEST ({selection_name}, "
                    f"epoch {selected_epoch:03d}) |",
                    test_metrics,
                ),
            )
            print(
                fmt_hurdle(
                    f"FINAL HURDLE ({selection_name}) |",
                    test_metrics,
                ),
            )
            print(
                fmt_strata(
                    f"FINAL STRATA ({selection_name}) |",
                    test_metrics,
                ),
            )
            print(
                fmt_components(
                    f"FINAL COMPONENT ({selection_name}) |",
                    test_metrics,
                ),
            )

        (output / "final_test_metrics.json").write_text(
            json.dumps(
                {
                    "test_was_not_used_for_checkpoint_selection": (
                        True
                    ),
                    "selections": final_tests,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "model": "HurdleGlobalARDTA",
                "residual_head": args.residual_head,
                "best_mse_epoch": best_mse_epoch,
                "best_mse_val": best_mse,
                "best_balanced_epoch": best_balanced_epoch,
                "best_balanced_score": best_balanced_score,
                "best_balanced_val": best_balanced,
                "final_tests": final_tests,
                "epochs_completed": epoch,
                "stopped_early": (
                    stale >= args.patience > 0
                ),
                "best_mse_checkpoint": (
                    "checkpoint_best_val_mse.pt"
                ),
                "best_balanced_checkpoint": (
                    "checkpoint_best_balanced.pt"
                ),
                "last_checkpoint": "checkpoint_last.pt",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
