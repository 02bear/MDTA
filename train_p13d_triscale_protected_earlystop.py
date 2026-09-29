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

from datasets.collate_p13d_triscale import (
    mdta_collate_fn_p13d_triscale,
    move_batch_to_device_triscale,
)
from datasets.davis_dataset_p13d_triscale import DavisDatasetP13DTriScale
from models.model_p13d_triscale_protected import (
    BRANCH_NAMES,
    MyModelMDTAP13DProtectedTriScale,
    parse_active_scales,
)
from train_p13d_earlystop import compute_regression_metrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(dataset, indices, args, shuffle=False):
    collate = partial(
        mdta_collate_fn_p13d_triscale,
        max_fragments=args.max_fragments,
        max_atoms_per_fragment=args.max_atoms_per_fragment,
        max_pockets=args.max_pockets,
        max_residues_per_pocket=args.max_residues_per_pocket,
    )
    worker_options = {}
    if args.num_workers > 0:
        worker_options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        **worker_options,
    )


def _autocast_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported amp dtype: {name}")


def masked_mse(pred, target, mask):
    """Masked FP32 MSE with a differentiable zero for an empty mask."""
    squared_error = (pred.float() - target.float()).square()
    mask_float = mask.to(squared_error.dtype)
    return (squared_error * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def compute_losses(output, label, active_scales, weights):
    label = label.reshape(-1)
    global_loss = F.mse_loss(output["global_pred"].float(), label.float())
    has_local = bool({"fp", "ar"} & active_scales)
    final_loss = (
        F.mse_loss(output["pred"].float(), label.float())
        if has_local
        else output["global_pred"].sum() * 0.0
    )
    fp_loss = masked_mse(
        output["fp_pred"], label, output["fp_valid_mask"]
    )
    ar_loss = masked_mse(
        output["ar_pred"], label, output["ar_valid_mask"]
    )
    joint_loss = masked_mse(
        output["joint_pred"], label, output["joint_valid_mask"]
    )
    total = (
        weights["final"] * final_loss
        + weights["global"] * global_loss
        + weights["fp"] * fp_loss
        + weights["ar"] * ar_loss
        + weights["joint"] * joint_loss
    )
    return {
        "total": total,
        "final": final_loss,
        "global": global_loss,
        "fp": fp_loss,
        "ar": ar_loss,
        "joint": joint_loss,
    }


def split_parameter_groups(model):
    global_params = [parameter for parameter in model.global_model.parameters()]
    fusion_params = [model.fusion_logits]
    excluded = {id(parameter) for parameter in global_params + fusion_params}
    local_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    groups = {
        "global": global_params,
        "local": local_params,
        "fusion": fusion_params,
    }
    all_ids = [id(parameter) for values in groups.values() for parameter in values]
    trainable_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("a trainable parameter appears in more than one group")
    if set(all_ids) != set(trainable_ids):
        raise RuntimeError("parameter groups do not cover every trainable parameter")
    return groups


def build_optimizer(model, args):
    groups = split_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["global"], "lr": args.global_lr},
            {"params": groups["local"], "lr": args.local_lr},
            {"params": groups["fusion"], "lr": args.fusion_lr},
        ],
        weight_decay=args.weight_decay,
    )
    counts = {
        name: sum(parameter.numel() for parameter in values)
        for name, values in groups.items()
    }
    counts["total"] = sum(counts.values())
    return optimizer, counts


def _safe_regression_metrics(prediction, label):
    prediction = prediction.float().reshape(-1)
    label = label.float().reshape(-1)
    count = int(label.numel())
    if count == 0:
        return {
            "count": 0,
            "mse": None,
            "rmse": None,
            "mae": None,
            "ci": None,
            "rm2": None,
        }
    mse = float(torch.mean((prediction - label).square()))
    mae = float(torch.mean(torch.abs(prediction - label)))
    if count < 2:
        return {
            "count": count,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": mae,
            "ci": None,
            "rm2": None,
        }
    try:
        metrics = compute_regression_metrics(prediction, label)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        metrics = {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": mae,
            "ci": None,
            "rm2": None,
        }
    metrics["count"] = count
    return metrics


def _prediction_distribution(prediction, label):
    prediction_np = prediction.float().reshape(-1).numpy().astype(np.float64)
    label_np = label.float().reshape(-1).numpy().astype(np.float64)
    if label_np.size > 1 and float(np.var(label_np)) > 0.0:
        slope = float(np.polyfit(label_np, prediction_np, 1)[0])
    else:
        slope = None
    return {
        "label_mean": float(label_np.mean()),
        "label_std": float(label_np.std()),
        "pred_mean": float(prediction_np.mean()),
        "pred_std": float(prediction_np.std()),
        "pred_slope": slope,
    }


def _tail_metrics(prediction, label):
    result = {}
    prediction = prediction.float().reshape(-1)
    label = label.float().reshape(-1)
    for threshold in (7, 8):
        mask = label > threshold
        count = int(mask.sum())
        if count:
            mse = float(torch.mean((prediction[mask] - label[mask]).square()))
            rmse = math.sqrt(mse)
        else:
            mse = rmse = None
        result[f"label_gt_{threshold}"] = {
            "count": count,
            "mse": mse,
            "rmse": rmse,
        }
    return result


def _format_metric(value):
    return "NA" if value is None else f"{value:.6f}"


def run_epoch(
    model,
    data_loader,
    device,
    optimizer=None,
    collect_rows=False,
    compute_full_metrics=True,
    log_interval=100,
    *,
    epoch=None,
    amp=False,
    amp_dtype="bf16",
    loss_weights=None,
    grad_clip_norm=5.0,
):
    """Run an epoch without per-sample CUDA-to-host synchronization."""
    if loss_weights is None:
        loss_weights = {
            "final": 1.0,
            "global": 1.0,
            "fp": 0.3,
            "ar": 0.3,
            "joint": 0.5,
        }
    is_train = optimizer is not None
    model.train(is_train)
    total_batches = len(data_loader)
    start_time = time.perf_counter()
    total_samples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loss_sums = {
        name: torch.zeros((), device=device, dtype=torch.float32)
        for name in ("total", "final", "global", "fp", "ar", "joint")
    }
    loss_counts = {name: torch.zeros_like(loss_sums["total"]) for name in loss_sums}
    stat_sums = {
        name: torch.zeros_like(loss_sums["total"])
        for name in (
            "fragments_per_sample",
            "pockets_per_sample",
            "fp_pairs_per_sample",
            "ar_regions_per_sample",
            "local_missing_rate",
        )
    }
    fusion_weight_sum = torch.zeros(4, device=device, dtype=torch.float32)
    valid_counts = {
        name: torch.zeros((), device=device, dtype=torch.float32)
        for name in ("fp", "ar", "joint")
    }

    keep_predictions = compute_full_metrics or collect_rows
    prediction_batches = (
        {name: [] for name in ("final", "global", "fp", "ar", "joint")}
        if keep_predictions
        else None
    )
    label_batches = [] if keep_predictions else None
    mask_batches = (
        {name: [] for name in ("fp", "ar", "joint")}
        if keep_predictions
        else None
    )
    fusion_batches = [] if keep_predictions else None
    drug_ids = [] if collect_rows else None
    protein_ids = [] if collect_rows else None
    autocast_enabled = bool(amp and device.type in {"cuda", "cpu"})
    non_blocking = device.type == "cuda"

    grad_context = torch.enable_grad if is_train else torch.no_grad
    with grad_context():
        for batch_index, batch in enumerate(data_loader, start=1):
            batch = move_batch_to_device_triscale(
                batch, device, non_blocking=non_blocking
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=_autocast_dtype(amp_dtype),
                enabled=autocast_enabled,
            ):
                output = model(batch)
            label = batch["label"].reshape(-1)
            losses = compute_losses(
                output, label, model.active_scales, loss_weights
            )
            if is_train:
                losses["total"].backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_norm
                    )
                optimizer.step()

            batch_size = label.shape[0]
            total_samples += batch_size
            branch_batch_counts = {
                name: output[f"{name}_valid_mask"].sum().to(torch.float32)
                for name in ("fp", "ar", "joint")
            }
            for name in ("total", "final", "global"):
                loss_sums[name] = loss_sums[name] + losses[name].detach() * batch_size
                loss_counts[name] = loss_counts[name] + batch_size
            for name in ("fp", "ar", "joint"):
                loss_sums[name] = (
                    loss_sums[name]
                    + losses[name].detach() * branch_batch_counts[name]
                )
                loss_counts[name] = loss_counts[name] + branch_batch_counts[name]
                valid_counts[name] = valid_counts[name] + branch_batch_counts[name]
            stats = output["stats"]
            for name in stat_sums:
                stat_sums[name] = (
                    stat_sums[name] + stats[name].float() * batch_size
                )
            fusion_weight_sum = (
                fusion_weight_sum
                + output["fusion_weights"].detach().float().sum(dim=0)
            )

            if keep_predictions:
                prediction_batches["final"].append(output["pred"].detach())
                for name in ("global", "fp", "ar", "joint"):
                    prediction_batches[name].append(
                        output[f"{name}_pred"].detach()
                    )
                label_batches.append(label.detach())
                for name in mask_batches:
                    mask_batches[name].append(
                        output[f"{name}_valid_mask"].detach()
                    )
                fusion_batches.append(output["fusion_weights"].detach())
            if collect_rows:
                drug_ids.extend(batch["drug_id"])
                protein_ids.extend(batch["protein_id"])

            if log_interval > 0 and (
                batch_index % log_interval == 0 or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - start_time
                running_loss = (
                    loss_sums["total"] / loss_counts["total"].clamp_min(1)
                ).item()
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / 2**20
                    reserved = torch.cuda.memory_reserved(device) / 2**20
                else:
                    allocated = reserved = 0.0
                print(
                    f"PROGRESS epoch={epoch if epoch is not None else '-'} "
                    f"batch={batch_index}/{total_batches} "
                    f"running_loss={running_loss:.6f} "
                    f"elapsed_seconds={elapsed:.1f} "
                    f"samples_per_second={total_samples / max(elapsed, 1e-9):.2f} "
                    f"cuda_memory_allocated={allocated:.1f}MiB "
                    f"cuda_memory_reserved={reserved:.1f}MiB",
                    flush=True,
                )

    aggregate_values = []
    for name in ("total", "final", "global", "fp", "ar", "joint"):
        aggregate_values.extend([loss_sums[name], loss_counts[name]])
    aggregate_values.extend(stat_sums.values())
    aggregate_values.extend([fusion_weight_sum, *valid_counts.values()])
    aggregate = torch.cat(
        [value.reshape(-1) for value in aggregate_values]
    ).detach().cpu()
    cursor = 0
    losses_epoch = {}
    for name in ("total", "final", "global", "fp", "ar", "joint"):
        value_sum = float(aggregate[cursor])
        value_count = float(aggregate[cursor + 1])
        losses_epoch[name] = value_sum / max(value_count, 1.0)
        cursor += 2
    stats_epoch = {}
    for name in stat_sums:
        stats_epoch[name] = float(aggregate[cursor]) / max(total_samples, 1)
        cursor += 1
    fusion_weights_epoch = aggregate[cursor : cursor + 4] / max(total_samples, 1)
    cursor += 4
    valid_counts_epoch = {}
    for name in valid_counts:
        valid_counts_epoch[name] = int(aggregate[cursor])
        cursor += 1

    elapsed = time.perf_counter() - start_time
    metrics = {
        "losses": losses_epoch,
        "fusion_weights": {
            name: float(fusion_weights_epoch[index])
            for index, name in enumerate(BRANCH_NAMES)
        },
        "local_validity_stats": {
            **stats_epoch,
            "fp_valid_count": valid_counts_epoch["fp"],
            "ar_valid_count": valid_counts_epoch["ar"],
            "joint_valid_count": valid_counts_epoch["joint"],
        },
        "performance": {
            "elapsed_seconds": elapsed,
            "samples_per_second": total_samples / max(elapsed, 1e-9),
            "seconds_per_batch": elapsed / max(total_batches, 1),
            "peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else 0.0
            ),
        },
    }
    rows = None

    if keep_predictions:
        predictions = {
            name: torch.cat(values, dim=0).float().cpu().reshape(-1)
            for name, values in prediction_batches.items()
        }
        labels = torch.cat(label_batches, dim=0).float().cpu().reshape(-1)
        masks = {
            name: torch.cat(values, dim=0).bool().cpu().reshape(-1)
            for name, values in mask_batches.items()
        }
        fusion = torch.cat(fusion_batches, dim=0).float().cpu()
        if compute_full_metrics:
            metrics["final"] = _safe_regression_metrics(
                predictions["final"], labels
            )
            metrics["global"] = _safe_regression_metrics(
                predictions["global"], labels
            )
            for name in ("fp", "ar", "joint"):
                metrics[name] = _safe_regression_metrics(
                    predictions[name][masks[name]], labels[masks[name]]
                )
            metrics["prediction_distribution"] = _prediction_distribution(
                predictions["final"], labels
            )
            metrics["tail_metrics"] = _tail_metrics(
                predictions["final"], labels
            )

        if collect_rows:
            rows = []
            for index, (drug_id, protein_id) in enumerate(
                zip(drug_ids, protein_ids)
            ):
                rows.append(
                    {
                        "drug_id": drug_id,
                        "protein_id": protein_id,
                        "label": float(labels[index]),
                        "global_pred": float(predictions["global"][index]),
                        "fp_pred": float(predictions["fp"][index]),
                        "ar_pred": float(predictions["ar"][index]),
                        "joint_pred": float(predictions["joint"][index]),
                        "final_pred": float(predictions["final"][index]),
                        "w_global": float(fusion[index, 0]),
                        "w_fp": float(fusion[index, 1]),
                        "w_ar": float(fusion[index, 2]),
                        "w_joint": float(fusion[index, 3]),
                        "fp_valid": bool(masks["fp"][index]),
                        "ar_valid": bool(masks["ar"][index]),
                        "joint_valid": bool(masks["joint"][index]),
                    }
                )
    return metrics, rows


def build_parser():
    parser = argparse.ArgumentParser()
    string_defaults = [
        ("pairs_csv", "data/raw/davis/pairs.csv"),
        ("drug_1d_dir", "data/processed/davis/drug_1d_chemberta2"),
        ("drug_2d_dir", "data/processed/davis/drug_2d"),
        ("drug_3d_dir", "data/processed/davis/drug_3d"),
        ("protein_1d_dir", "data/processed/davis/protein_1d_esm2"),
        ("protein_3d_dir", "data/processed/davis/protein_3d_gvp"),
        ("split_json", "data/splits/davis_fixed_split_size2.json"),
        ("output_dir", "outputs/davis_p13d_triscale_protected"),
        (
            "drug_fragment_cache",
            "data/processed/davis/multiscale/drug_brics_fragments.pt",
        ),
        (
            "protein_pocket_cache",
            "data/processed/davis/multiscale/protein_pockets_p2rank_top3_v2.pt",
        ),
        ("active_scales", "global,fp,ar"),
        ("cross_scale_mode", "bottom_up"),
    ]
    for name, default in string_defaults:
        parser.add_argument(f"--{name}", default=default)
    integer_defaults = [
        ("seed", 42),
        ("batch_size", 2),
        ("num_workers", 0),
        ("epochs", 30),
        ("early_stop_patience", 60),
        ("drug_1d_in_dim", 768),
        ("drug_3d_node_in_dim", 10),
        ("hidden_dim", 128),
        ("max_fragments", 12),
        ("max_atoms_per_fragment", 24),
        ("max_pockets", 3),
        ("max_residues_per_pocket", 32),
        ("local_hidden_dim", 128),
        ("log_interval", 100),
        ("ar_region_chunk_size", 32),
    ]
    for name, default in integer_defaults:
        parser.add_argument(f"--{name}", type=int, default=default)
    float_defaults = [
        ("lr", 3e-4),
        ("weight_decay", 1e-5),
        ("early_stop_min_delta", 1e-4),
        ("dropout", 0.1),
        ("local_dropout", 0.1),
        ("grad_clip_norm", 5.0),
        ("final_loss_weight", 1.0),
        ("global_loss_weight", 1.0),
        ("fp_loss_weight", 0.3),
        ("ar_loss_weight", 0.3),
        ("joint_loss_weight", 0.5),
    ]
    for name, default in float_defaults:
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--global_lr", type=float, default=None)
    parser.add_argument("--local_lr", type=float, default=None)
    parser.add_argument("--fusion_lr", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--top_k_fp_for_ar", type=int, default=None)
    parser.add_argument("--joint_recompute_fp_score", action="store_true")
    return parser


def _print_epoch(epoch, train_metrics, val_metrics, model):
    train_losses = train_metrics["losses"]
    final = val_metrics["final"]
    global_metrics = val_metrics["global"]
    weights = val_metrics["fusion_weights"]
    validity = val_metrics["local_validity_stats"]
    distribution = val_metrics["prediction_distribution"]
    tails = val_metrics["tail_metrics"]
    print(
        f"EPOCH {epoch} "
        f"TRAIN_TOTAL_LOSS={train_losses['total']:.6f} "
        f"TRAIN_FINAL_LOSS={train_losses['final']:.6f} "
        f"TRAIN_GLOBAL_LOSS={train_losses['global']:.6f} "
        f"TRAIN_FP_LOSS={train_losses['fp']:.6f} "
        f"TRAIN_AR_LOSS={train_losses['ar']:.6f} "
        f"TRAIN_JOINT_LOSS={train_losses['joint']:.6f} "
        f"VAL_FINAL_MSE={_format_metric(final['mse'])} "
        f"VAL_FINAL_RMSE={_format_metric(final['rmse'])} "
        f"VAL_FINAL_CI={_format_metric(final['ci'])} "
        f"VAL_FINAL_RM2={_format_metric(final['rm2'])} "
        f"VAL_GLOBAL_MSE={_format_metric(global_metrics['mse'])} "
        f"VAL_GLOBAL_RMSE={_format_metric(global_metrics['rmse'])} "
        f"VAL_GLOBAL_CI={_format_metric(global_metrics['ci'])} "
        f"VAL_GLOBAL_RM2={_format_metric(global_metrics['rm2'])} "
        f"VAL_FP_MSE={_format_metric(val_metrics['fp']['mse'])} "
        f"VAL_AR_MSE={_format_metric(val_metrics['ar']['mse'])} "
        f"VAL_JOINT_MSE={_format_metric(val_metrics['joint']['mse'])} "
        f"FP_VALID_COUNT={val_metrics['fp']['count']} "
        f"AR_VALID_COUNT={val_metrics['ar']['count']} "
        f"JOINT_VALID_COUNT={val_metrics['joint']['count']} "
        f"W_GLOBAL={weights['global']:.6f} "
        f"W_FP={weights['fp']:.6f} W_AR={weights['ar']:.6f} "
        f"W_JOINT={weights['joint']:.6f} "
        f"AR_TO_FP_GATE={float(torch.sigmoid(model.ar_to_fp_gate)):.6f} "
        f"FP_TO_GLOBAL_GATE={float(torch.sigmoid(model.fp_to_global_gate)):.6f} "
        f"LOCAL_MISSING_RATE={validity['local_missing_rate']:.6f} "
        f"AVG_FRAGMENTS={validity['fragments_per_sample']:.3f} "
        f"AVG_POCKETS={validity['pockets_per_sample']:.3f} "
        f"AVG_FP_PAIRS={validity['fp_pairs_per_sample']:.3f} "
        f"AVG_AR_REGIONS={validity['ar_regions_per_sample']:.3f} "
        f"LABEL_MEAN={distribution['label_mean']:.6f} "
        f"LABEL_STD={distribution['label_std']:.6f} "
        f"PRED_MEAN={distribution['pred_mean']:.6f} "
        f"PRED_STD={distribution['pred_std']:.6f} "
        f"PRED_SLOPE={_format_metric(distribution['pred_slope'])} "
        f"MSE_LABEL_GT_7={_format_metric(tails['label_gt_7']['mse'])} "
        f"RMSE_LABEL_GT_7={_format_metric(tails['label_gt_7']['rmse'])} "
        f"COUNT_LABEL_GT_7={tails['label_gt_7']['count']} "
        f"MSE_LABEL_GT_8={_format_metric(tails['label_gt_8']['mse'])} "
        f"RMSE_LABEL_GT_8={_format_metric(tails['label_gt_8']['rmse'])} "
        f"COUNT_LABEL_GT_8={tails['label_gt_8']['count']}",
        flush=True,
    )


def main():
    args = build_parser().parse_args()
    scales = parse_active_scales(args.active_scales)
    if scales != {"global"} and (
        not Path(args.drug_fragment_cache).exists()
        or not Path(args.protein_pocket_cache).exists()
    ):
        raise SystemExit("local scales require both offline cache files")
    args.global_lr = args.lr if args.global_lr is None else args.global_lr
    args.local_lr = args.lr if args.local_lr is None else args.local_lr
    args.fusion_lr = args.lr if args.fusion_lr is None else args.fusion_lr
    if args.amp:
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    dataset = DavisDatasetP13DTriScale(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.protein_3d_dir,
        drug_2d_dir=args.drug_2d_dir,
        use_drug_2d=False,
        drug_3d_dir=args.drug_3d_dir,
        use_drug_3d=True,
        drug_fragment_cache=args.drug_fragment_cache,
        protein_pocket_cache=args.protein_pocket_cache,
    )
    with open(args.split_json, encoding="utf-8") as handle:
        split = json.load(handle)
    train_loader = loader(dataset, split["train_indices"], args, True)
    val_loader = loader(dataset, split["val_indices"], args)
    test_loader = loader(
        dataset, split.get("test_indices", split["val_indices"]), args
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModelMDTAP13DProtectedTriScale(
        drug_1d_in_dim=args.drug_1d_in_dim,
        drug_3d_node_in_dim=args.drug_3d_node_in_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        active_scales=scales,
        cross_scale_mode=args.cross_scale_mode,
        local_hidden_dim=args.local_hidden_dim,
        local_dropout=args.local_dropout,
        max_fragments=args.max_fragments,
        max_atoms_per_fragment=args.max_atoms_per_fragment,
        max_pockets=args.max_pockets,
        max_residues_per_pocket=args.max_residues_per_pocket,
        ar_region_chunk_size=args.ar_region_chunk_size,
        top_k_fp_for_ar=args.top_k_fp_for_ar,
        joint_recompute_fp_score=args.joint_recompute_fp_score,
    ).to(device)
    optimizer, parameter_counts = build_optimizer(model, args)
    loss_weights = {
        "final": args.final_loss_weight,
        "global": args.global_loss_weight,
        "fp": args.fp_loss_weight,
        "ar": args.ar_loss_weight,
        "joint": args.joint_loss_weight,
    }
    print("MODEL ProtectedTriScale from-scratch")
    print("ACTIVE_SCALES", sorted(scales))
    print("CROSS_SCALE_MODE", args.cross_scale_mode)
    print("JOINT_RECOMPUTE_FP_SCORE", args.joint_recompute_fp_score)
    print("TOP_K_FP_FOR_AR", args.top_k_fp_for_ar)
    print("AR_REGION_CHUNK_SIZE", args.ar_region_chunk_size)
    print("AMP", args.amp)
    print("AMP_DTYPE", args.amp_dtype)
    for name, count in parameter_counts.items():
        print(f"{name.upper()}_PARAMS", count)

    best_final = float("inf")
    best_global = float("inf")
    wait = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, train_rows = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            collect_rows=False,
            compute_full_metrics=False,
            log_interval=args.log_interval,
            epoch=epoch,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            loss_weights=loss_weights,
            grad_clip_norm=args.grad_clip_norm,
        )
        assert train_rows is None
        val_metrics, val_rows = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            collect_rows=False,
            compute_full_metrics=True,
            log_interval=args.log_interval,
            epoch=epoch,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            loss_weights=loss_weights,
            grad_clip_norm=args.grad_clip_norm,
        )
        assert val_rows is None
        _print_epoch(epoch, train_metrics, val_metrics, model)

        final_rmse = val_metrics["final"]["rmse"]
        global_rmse = val_metrics["global"]["rmse"]
        if final_rmse < best_final - args.early_stop_min_delta:
            best_final = final_rmse
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                },
                output_dir / "best_model.pt",
            )
            with open(
                output_dir / "best_metrics.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(val_metrics, handle, indent=2)
        else:
            wait += 1
        if global_rmse < best_global - args.early_stop_min_delta:
            best_global = global_rmse
            torch.save(
                {
                    "global_model_state_dict": model.global_model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                },
                output_dir / "best_global_model.pt",
            )
            with open(
                output_dir / "best_global_metrics.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(val_metrics["global"], handle, indent=2)
        if args.early_stop_patience > 0 and wait >= args.early_stop_patience:
            break

    checkpoint = torch.load(
        output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, rows = run_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        collect_rows=True,
        compute_full_metrics=True,
        log_interval=args.log_interval,
        epoch="test",
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        loss_weights=loss_weights,
        grad_clip_norm=args.grad_clip_norm,
    )
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    with open(
        output_dir / "test_predictions.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
