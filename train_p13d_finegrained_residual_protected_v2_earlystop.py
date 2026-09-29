# -*- coding: utf-8 -*-
"""Train and evaluate Protected Residual E2 v2 on Davis drug-cold splits."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.collate_p13d import mdta_collate_fn_p13d, move_batch_to_device
from datasets.davis_dataset_p13d import DavisDatasetP13D
from models.model_p13d_finegrained_residual_protected_v2 import (
    GLOBAL_MODULE_NAMES,
    LOCAL_MODULE_NAMES,
    MyModelMDTAP13DFineGrainedProtectedV2,
)
from train_p13d_finegrained_residual_earlystop import (
    compute_regression_metrics,
)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def save_json(path: str | Path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False)


def load_split(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    for key in ("train_indices", "val_indices", "test_indices"):
        if key not in split or not isinstance(split[key], list):
            raise KeyError(f"split file is missing list field {key!r}")
    sets = {key: set(map(int, split[key])) for key in split if key.endswith("_indices")}
    if sets["train_indices"] & sets["val_indices"]:
        raise RuntimeError("train and validation indices overlap")
    if sets["train_indices"] & sets["test_indices"]:
        raise RuntimeError("train and test indices overlap")
    if sets["val_indices"] & sets["test_indices"]:
        raise RuntimeError("validation and test indices overlap")
    return split


def build_dataset(args):
    return DavisDatasetP13D(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.protein_3d_dir,
        drug_2d_dir=args.drug_2d_dir,
        use_drug_2d=False,
        drug_3d_dir=args.drug_3d_dir,
        use_drug_3d=True,
    )


def verify_drug_cold_split(dataset, split):
    result = {}
    drug_sets = {}
    for name, key in (
        ("train", "train_indices"),
        ("val", "val_indices"),
        ("test", "test_indices"),
    ):
        indices = list(map(int, split[key]))
        invalid = [i for i in indices if i < 0 or i >= len(dataset)]
        if invalid:
            raise ValueError(f"{name} has invalid indices: {invalid[:10]}")
        drugs = set(dataset.df.iloc[indices]["drug_id"].astype(str))
        drug_sets[name] = drugs
        result[f"{name}_pairs"] = len(indices)
        result[f"{name}_drugs"] = len(drugs)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = drug_sets[left] & drug_sets[right]
        if overlap:
            raise RuntimeError(f"{left}/{right} drug overlap: {sorted(overlap)}")
    result["drug_disjoint_check"] = True
    return result


def make_loader(dataset, indices, args, shuffle: bool):
    kwargs = {
        "dataset": Subset(dataset, list(map(int, indices))),
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "collate_fn": mdta_collate_fn_p13d,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def build_model(args, device):
    return MyModelMDTAP13DFineGrainedProtectedV2(
        drug_1d_in_dim=args.drug_1d_in_dim,
        drug_3d_node_in_dim=args.drug_3d_node_in_dim,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        task="regression",
        pocket_top_k=args.pocket_top_k,
        interaction_heads=args.interaction_heads,
        delta_limit=args.delta_limit,
        include_base_pred_in_delta_head=args.include_base_pred_in_delta_head,
    ).to(device)


def split_parameter_groups(model):
    groups = {
        "global": [
            parameter
            for name in GLOBAL_MODULE_NAMES
            for parameter in getattr(model, name).parameters()
            if parameter.requires_grad
        ],
        "local": [
            parameter
            for name in LOCAL_MODULE_NAMES
            for parameter in getattr(model, name).parameters()
            if parameter.requires_grad
        ],
    }
    grouped_ids = [id(p) for values in groups.values() for p in values]
    trainable_ids = [id(p) for p in model.parameters() if p.requires_grad]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("a trainable parameter appears in multiple groups")
    if set(grouped_ids) != set(trainable_ids):
        missing = set(trainable_ids) - set(grouped_ids)
        extra = set(grouped_ids) - set(trainable_ids)
        raise RuntimeError(
            f"parameter groups are incomplete: missing={len(missing)} extra={len(extra)}"
        )
    return groups


def build_optimizer(model, args):
    groups = split_parameter_groups(model)
    global_init_mode = getattr(args, "global_init_mode", "scratch")
    if global_init_mode == "baseline_checkpoint":
        global_lr = args.global_lr_after_init
        optimizer_class = torch.optim.AdamW
    else:
        global_lr = args.global_lr
        optimizer_name = getattr(args, "scratch_optimizer", "adam")
        optimizer_class = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }[optimizer_name]
    optimizer = optimizer_class(
        [
            {"params": groups["global"], "lr": global_lr, "name": "global"},
            {"params": groups["local"], "lr": args.local_lr, "name": "local"},
        ],
        weight_decay=args.weight_decay,
    )
    counts = {
        name: sum(parameter.numel() for parameter in values)
        for name, values in groups.items()
    }
    counts["total"] = sum(counts.values())
    counts["optimizer"] = optimizer_class.__name__
    counts["global_lr"] = global_lr
    counts["local_lr"] = args.local_lr
    return optimizer, counts


def _split_signature(path):
    split = load_split(path)
    return {
        key: list(map(int, split[key]))
        for key in ("train_indices", "val_indices")
    }


def load_global_checkpoint(model, checkpoint_path, expected_split_json):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict) or not checkpoint_args.get("split_json"):
        raise ValueError("global checkpoint does not record its split_json")
    if _split_signature(checkpoint_args["split_json"]) != _split_signature(
        expected_split_json
    ):
        raise ValueError("global checkpoint train/validation split does not match")
    selection_split = checkpoint_args.get("selection_split")
    if selection_split is not None and selection_split != "val":
        raise ValueError(
            f"global checkpoint was selected on {selection_split!r}, not validation"
        )
    if "test_metrics" in checkpoint:
        raise ValueError(
            "global checkpoint contains test_metrics and cannot prove validation-only selection"
        )
    checkpoint_val_metrics = checkpoint.get("val_metrics")
    if not isinstance(checkpoint_val_metrics, dict) or "mse" not in checkpoint_val_metrics:
        raise ValueError("global checkpoint does not contain validation MSE")

    source = checkpoint.get("model_state_dict")
    if not isinstance(source, dict):
        raise KeyError("global checkpoint is missing model_state_dict")
    destination = model.state_dict()
    prefixes = tuple(f"{name}." for name in GLOBAL_MODULE_NAMES)
    expected_keys = [key for key in destination if key.startswith(prefixes)]
    compatible = {}
    missing = []
    shape_mismatch = []
    for key in expected_keys:
        if key not in source:
            missing.append(key)
        elif source[key].shape != destination[key].shape:
            shape_mismatch.append(
                (key, tuple(source[key].shape), tuple(destination[key].shape))
            )
        else:
            compatible[key] = source[key]
    unused = [key for key in source if key not in compatible]
    expected_numel = sum(destination[key].numel() for key in expected_keys)
    matched_numel = sum(destination[key].numel() for key in compatible)
    coverage = matched_numel / max(expected_numel, 1)
    print("GLOBAL_CHECKPOINT", checkpoint_path)
    print("GLOBAL_MATCHED_TENSORS", len(compatible))
    print("GLOBAL_MISSING_KEYS", missing)
    print("GLOBAL_SHAPE_MISMATCH", shape_mismatch)
    print("GLOBAL_UNUSED_KEYS", unused)
    print("GLOBAL_PARAMETER_COVERAGE", f"{coverage:.6f}")
    if missing or shape_mismatch or coverage < 0.999999:
        raise RuntimeError("global checkpoint does not fully cover the baseline branch")
    destination.update(compatible)
    model.load_state_dict(destination, strict=True)
    return {
        "matched_tensors": len(compatible),
        "missing_keys": missing,
        "shape_mismatch": shape_mismatch,
        "unused_keys": unused,
        "coverage": coverage,
        "checkpoint_val_metrics": checkpoint_val_metrics,
    }


def compare_global_config(args, reference_checkpoint_path):
    checkpoint = torch.load(
        reference_checkpoint_path, map_location="cpu", weights_only=False
    )
    reference = checkpoint.get("args")
    if not isinstance(reference, dict):
        raise ValueError("baseline reference checkpoint does not contain args")

    def reference_value(*names):
        for name in names:
            if name in reference:
                return reference[name]
        return None

    checks = {
        "batch_size": (args.batch_size, reference_value("batch_size")),
        "lr/global_lr": (args.global_lr, reference_value("global_lr", "lr")),
        "weight_decay": (args.weight_decay, reference_value("weight_decay")),
        "dropout": (args.dropout, reference_value("dropout")),
        "hidden_dim": (args.hidden_dim, reference_value("hidden_dim")),
        "seed": (args.seed, reference_value("seed")),
        "drug_1d_in_dim": (
            args.drug_1d_in_dim,
            reference_value("drug_1d_in_dim"),
        ),
        "drug_3d_node_in_dim": (
            args.drug_3d_node_in_dim,
            reference_value("drug_3d_node_in_dim"),
        ),
        "protein_1d_in_dim": (1280, reference_value("protein_1d_in_dim")),
        "protein_3d_node_s_dim": (
            6,
            reference_value("protein_3d_node_s_dim"),
        ),
        "protein_3d_node_v_dim": (
            3,
            reference_value("protein_3d_node_v_dim"),
        ),
        "early_stop_patience": (
            args.early_stop_patience,
            reference_value("early_stop_patience"),
        ),
        "early_stop_min_delta": (
            args.early_stop_min_delta,
            reference_value("early_stop_min_delta"),
        ),
        "optimizer": (
            args.scratch_optimizer,
            reference_value("optimizer", "optimizer_name"),
        ),
    }
    report = {}
    warnings = []
    for name, (current, baseline) in checks.items():
        match = None if baseline is None else current == baseline
        report[name] = {
            "current": current,
            "baseline": baseline,
            "match": match,
        }
        if match is not True:
            reason = "not recorded by baseline" if baseline is None else "mismatch"
            warnings.append(f"{name}: {reason} (current={current}, baseline={baseline})")

    baseline_split_path = reference_value("split_json")
    split_match = None
    if baseline_split_path is not None:
        split_match = _split_signature(args.split_json) == _split_signature(
            baseline_split_path
        )
    report["split_json"] = {
        "current": args.split_json,
        "baseline": baseline_split_path,
        "match": split_match,
    }
    if split_match is not True:
        reason = "not recorded by baseline" if baseline_split_path is None else "mismatch"
        warnings.append(f"split_json: {reason}")
    result = {
        "reference_checkpoint": str(reference_checkpoint_path),
        "checks": report,
        "all_recorded_fields_match": all(
            item["match"] is not False for item in report.values()
        ),
        "warnings": warnings,
    }
    print("GLOBAL_CONFIG_MATCH_REPORT", json.dumps(result, sort_keys=True))
    for warning in warnings:
        print("WARNING GLOBAL_CONFIG_MISMATCH", warning, flush=True)
    return result


def compute_protected_losses(output, label, args, disable_local=False):
    label = label.reshape_as(output["base_pred"]).float()
    base = output["base_pred"].float()
    delta = output["local_delta"].float()
    final = output["pred"].float()
    residual_target = label - base.detach()
    valid_mask = output["local_valid_mask"].reshape(-1).bool()
    delta_flat = delta.reshape(-1)
    residual_target_flat = residual_target.reshape(-1)

    base_loss = F.mse_loss(base, label)
    local_loss_disabled = disable_local or not valid_mask.any()
    if local_loss_disabled:
        residual_loss = delta.sum() * 0.0
        delta_penalty = delta.sum() * 0.0
    else:
        residual_loss = F.smooth_l1_loss(
            delta_flat[valid_mask],
            residual_target_flat[valid_mask],
            beta=args.residual_huber_beta,
        )
        delta_penalty = delta_flat[valid_mask].square().mean()
    final_loss = (
        F.mse_loss(final, label)
        if args.final_loss_weight > 0
        else base_loss * 0.0
    )
    total_loss = (
        base_loss
        + args.residual_loss_weight * residual_loss
        + args.delta_penalty_weight * delta_penalty
        + args.final_loss_weight * final_loss
    )
    if local_loss_disabled:
        total_loss = base_loss
    return {
        "total": total_loss,
        "base": base_loss,
        "residual": residual_loss,
        "delta_penalty": delta_penalty,
        "final": final_loss,
        "valid_mask": valid_mask,
        "residual_target": residual_target,
    }


def _safe_metrics(prediction, label):
    prediction = prediction.float().reshape(-1)
    label = label.float().reshape(-1)
    if label.numel() == 0:
        return {name: None for name in ("mse", "rmse", "mae", "ci", "rm2")}
    if label.numel() == 1:
        error = prediction - label
        mse = float(error.square().mean())
        return {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": float(error.abs().mean()),
            "ci": None,
            "rm2": None,
        }
    return compute_regression_metrics(prediction, label)


def _slope(label, prediction):
    label = np.asarray(label, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if label.size < 2 or float(np.var(label)) == 0.0:
        return None
    return float(np.polyfit(label, prediction, 1)[0])


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def per_drug_metrics(rows):
    records = []
    for drug_id, group in rows.groupby("drug_id", sort=True):
        label = torch.tensor(group["label"].to_numpy(np.float32))
        base = torch.tensor(group["base_pred"].to_numpy(np.float32))
        final = torch.tensor(group["final_pred"].to_numpy(np.float32))
        delta = group["local_delta"].to_numpy(np.float64)
        base_metrics = _safe_metrics(base, label)
        final_metrics = _safe_metrics(final, label)
        record = {
            "drug_id": str(drug_id),
            "sample_count": int(len(group)),
            "label_mean": float(label.mean()),
            "label_std": float(label.std(unbiased=False)),
        }
        for name, value in base_metrics.items():
            record[f"base_{name}"] = value
        for name, value in final_metrics.items():
            record[f"final_{name}"] = value
        record.update(
            {
                "delta_mean": float(delta.mean()),
                "delta_std": float(delta.std()),
                "delta_abs_mean": float(np.abs(delta).mean()),
                "delta_max_abs": float(np.abs(delta).max()),
                "mse_improvement": base_metrics["mse"] - final_metrics["mse"],
                "rmse_improvement": base_metrics["rmse"] - final_metrics["rmse"],
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def summarize_per_drug(table):
    degradation = table["final_rmse"] - table["base_rmse"]
    return {
        "mean_drug_base_rmse": float(table["base_rmse"].mean()),
        "mean_drug_final_rmse": float(table["final_rmse"].mean()),
        "median_drug_base_rmse": float(table["base_rmse"].median()),
        "median_drug_final_rmse": float(table["final_rmse"].median()),
        "worst_drug_base_rmse": float(table["base_rmse"].max()),
        "worst_drug_final_rmse": float(table["final_rmse"].max()),
        "worst_drug_degradation": float(degradation.max()),
        "fraction_drugs_improved": float((table["rmse_improvement"] > 0).mean()),
    }


def _label_bin_metrics(rows):
    label = rows["label"].to_numpy(np.float64)
    definitions = [
        ("label_eq_5", label == 5),
        ("label_gt_5_le_6", (label > 5) & (label <= 6)),
        ("label_gt_6_le_7", (label > 6) & (label <= 7)),
        ("label_gt_7_le_8", (label > 7) & (label <= 8)),
        ("label_gt_8", label > 8),
    ]
    result = {}
    for name, mask in definitions:
        part = rows.loc[mask]
        if part.empty:
            result[name] = {
                "sample_count": 0,
                "base_mse": None,
                "final_mse": None,
                "delta_mean": None,
                "delta_abs_mean": None,
            }
            continue
        result[name] = {
            "sample_count": int(len(part)),
            "base_mse": float(np.mean((part.base_pred - part.label) ** 2)),
            "final_mse": float(np.mean((part.final_pred - part.label) ** 2)),
            "delta_mean": float(part.local_delta.mean()),
            "delta_abs_mean": float(part.local_delta.abs().mean()),
        }
    return result


def _epoch_diagnostics(
    rows,
    losses,
    delta_limit,
    local_valid_count,
    local_valid_rate,
    performance,
):
    label = rows.label.to_numpy(np.float64)
    base = rows.base_pred.to_numpy(np.float64)
    final = rows.final_pred.to_numpy(np.float64)
    delta = rows.local_delta.to_numpy(np.float64)
    true_residual = label - base
    abs_delta = np.abs(delta)
    per_drug = per_drug_metrics(rows)
    return {
        "losses": losses,
        "base": _safe_metrics(torch.tensor(base), torch.tensor(label)),
        "final": _safe_metrics(torch.tensor(final), torch.tensor(label)),
        "prediction_distribution": {
            "label_mean": float(label.mean()),
            "label_std": float(label.std()),
            "base_pred_mean": float(base.mean()),
            "base_pred_std": float(base.std()),
            "final_pred_mean": float(final.mean()),
            "final_pred_std": float(final.std()),
            "base_pred_slope": _slope(label, base),
            "final_pred_slope": _slope(label, final),
        },
        "delta_distribution": {
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "delta_abs_mean": float(abs_delta.mean()),
            "delta_p50": float(np.quantile(abs_delta, 0.50)),
            "delta_p90": float(np.quantile(abs_delta, 0.90)),
            "delta_p95": float(np.quantile(abs_delta, 0.95)),
            "delta_p99": float(np.quantile(abs_delta, 0.99)),
            "delta_min": float(delta.min()),
            "delta_max": float(delta.max()),
            "bound_saturation_rate": float(
                (abs_delta >= 0.95 * delta_limit).mean()
            ),
        },
        "residual_quality": {
            "true_residual_mean": float(true_residual.mean()),
            "true_residual_std": float(true_residual.std()),
            "corr_delta_true_residual": _correlation(delta, true_residual),
            "delta_residual_mse": float(np.mean((delta - true_residual) ** 2)),
        },
        "label_bins": _label_bin_metrics(rows),
        "per_drug_summary": summarize_per_drug(per_drug),
        "local_valid_count": int(local_valid_count),
        "local_valid_rate": local_valid_rate,
        "performance": performance,
    }, per_drug


def _autocast_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported amp dtype: {name}")


def run_epoch(
    model,
    loader,
    device,
    args,
    optimizer=None,
    epoch=None,
    disable_local=False,
):
    is_train = optimizer is not None
    model.train(is_train)
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss_names = ("total", "base", "residual", "delta_penalty", "final")
    loss_sums = {
        name: torch.zeros((), device=device, dtype=torch.float32)
        for name in loss_names
    }
    loss_counts = {
        name: torch.zeros((), device=device, dtype=torch.float32)
        for name in loss_names
    }
    prediction_batches = {name: [] for name in ("label", "base", "delta", "final", "valid")}
    drug_ids, protein_ids = [], []
    sample_count = 0
    total_batches = len(loader)
    max_batches = args.max_train_batches if is_train else args.max_eval_batches
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            drug_ids.extend(map(str, batch["drug_id"]))
            protein_ids.extend(map(str, batch["protein_id"]))
            batch = move_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=_autocast_dtype(args.amp_dtype),
                enabled=bool(args.amp and device.type in {"cuda", "cpu"}),
            ):
                output = model(
                    batch,
                    return_debug=True,
                    disable_local=disable_local,
                )
            label = batch["label"].reshape_as(output["base_pred"]).float()
            base = output["base_pred"].float()
            delta = output["local_delta"].float()
            final = output["pred"].float()
            batch_losses = compute_protected_losses(
                output, label, args, disable_local=disable_local
            )
            total_loss = batch_losses["total"]
            if is_train:
                total_loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm
                    )
                optimizer.step()

            current_batch = label.shape[0]
            sample_count += current_batch
            batch_count = torch.as_tensor(
                current_batch, device=device, dtype=torch.float32
            )
            valid_count = batch_losses["valid_mask"].sum().to(torch.float32)
            for name in ("total", "base", "final"):
                loss_sums[name] = (
                    loss_sums[name] + batch_losses[name].detach() * batch_count
                )
                loss_counts[name] = loss_counts[name] + batch_count
            for name in ("residual", "delta_penalty"):
                loss_sums[name] = (
                    loss_sums[name] + batch_losses[name].detach() * valid_count
                )
                loss_counts[name] = loss_counts[name] + valid_count
            prediction_batches["label"].append(label.detach().reshape(-1))
            prediction_batches["base"].append(base.detach().reshape(-1))
            prediction_batches["delta"].append(delta.detach().reshape(-1))
            prediction_batches["final"].append(final.detach().reshape(-1))
            prediction_batches["valid"].append(
                output["local_valid_mask"].detach().reshape(-1)
            )

            if args.log_interval > 0 and (
                batch_index % args.log_interval == 0 or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - start
                running = (loss_sums["total"] / max(sample_count, 1)).item()
                allocated = (
                    torch.cuda.memory_allocated(device) / 2**20
                    if device.type == "cuda"
                    else 0.0
                )
                reserved = (
                    torch.cuda.memory_reserved(device) / 2**20
                    if device.type == "cuda"
                    else 0.0
                )
                print(
                    f"PROGRESS epoch={epoch} batch={batch_index}/{total_batches} "
                    f"running_loss={running:.6f} elapsed_seconds={elapsed:.1f} "
                    f"samples_per_second={sample_count/max(elapsed,1e-9):.2f} "
                    f"cuda_memory_allocated={allocated:.1f}MiB "
                    f"cuda_memory_reserved={reserved:.1f}MiB",
                    flush=True,
                )

    aggregate = torch.stack(
        [
            torch.stack([loss_sums[name], loss_counts[name]])
            for name in loss_names
        ]
    ).cpu()
    losses = {
        name: float(aggregate[index, 0]) / max(float(aggregate[index, 1]), 1.0)
        for index, name in enumerate(loss_names)
    }
    tensors = {
        name: torch.cat(values).float().cpu()
        for name, values in prediction_batches.items()
    }
    rows = pd.DataFrame(
        {
            "drug_id": drug_ids,
            "protein_id": protein_ids,
            "label": tensors["label"].numpy(),
            "base_pred": tensors["base"].numpy(),
            "local_delta": tensors["delta"].numpy(),
            "final_pred": tensors["final"].numpy(),
            "local_valid": tensors["valid"].bool().numpy(),
        }
    )
    elapsed = time.perf_counter() - start
    performance = {
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count / max(elapsed, 1e-9),
        "seconds_per_batch": elapsed / max(total_batches, 1),
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else 0.0
        ),
    }
    local_valid_count = int(tensors["valid"].sum())
    metrics, drug_table = _epoch_diagnostics(
        rows,
        losses,
        model.delta_limit,
        local_valid_count,
        float(tensors["valid"].mean()),
        performance,
    )
    return metrics, rows, drug_table


def save_checkpoint(path, model, optimizer, epoch, args, train_metrics, val_metrics):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def _pool_record(epoch, path, val_metrics):
    return {
        "epoch": int(epoch),
        "path": str(path),
        "overall_mse": float(val_metrics["final"]["mse"]),
        "worst_drug_degradation": float(
            val_metrics["per_drug_summary"]["worst_drug_degradation"]
        ),
        "mean_drug_final_rmse": float(
            val_metrics["per_drug_summary"]["mean_drug_final_rmse"]
        ),
    }


def update_checkpoint_pool(
    records,
    pool_size,
    checkpoint_dir,
    model,
    optimizer,
    epoch,
    args,
    train_metrics,
    val_metrics,
):
    path = checkpoint_dir / f"candidate_epoch_{epoch:04d}.pt"
    record = _pool_record(epoch, path, val_metrics)
    prospective = sorted(
        records + [record], key=lambda item: item["overall_mse"]
    )[:pool_size]
    if any(item["epoch"] == epoch for item in prospective):
        save_checkpoint(
            path, model, optimizer, epoch, args, train_metrics, val_metrics
        )
    keep_epochs = {item["epoch"] for item in prospective}
    for old in records:
        if old["epoch"] not in keep_epochs:
            old_path = Path(old["path"])
            if old_path.exists():
                old_path.unlink()
    return prospective


def select_checkpoints(records, tolerance, top_k, minimum_gap):
    if not records:
        raise RuntimeError("checkpoint pool is empty")
    best_mse = min(item["overall_mse"] for item in records)
    near_best = [
        item
        for item in records
        if item["overall_mse"] <= best_mse * (1.0 + tolerance)
    ]
    robust = min(
        near_best,
        key=lambda item: (
            item["worst_drug_degradation"],
            item["mean_drug_final_rmse"],
            item["overall_mse"],
        ),
    )
    ensemble = []
    for item in sorted(records, key=lambda value: value["overall_mse"]):
        if all(abs(item["epoch"] - chosen["epoch"]) >= minimum_gap for chosen in ensemble):
            ensemble.append(item)
        if len(ensemble) >= top_k:
            break
    return robust, ensemble, near_best


def update_spaced_ensemble_pool(
    records,
    top_k,
    checkpoint_dir,
    model,
    optimizer,
    epoch,
    args,
    train_metrics,
    val_metrics,
):
    """Keep validation-best snapshots sampled at the required epoch gap."""
    if epoch % args.checkpoint_min_epoch_gap != 0:
        return records
    path = checkpoint_dir / f"ensemble_epoch_{epoch:04d}.pt"
    record = _pool_record(epoch, path, val_metrics)
    prospective = sorted(
        records + [record], key=lambda item: item["overall_mse"]
    )[:top_k]
    if any(item["epoch"] == epoch for item in prospective):
        save_checkpoint(
            path, model, optimizer, epoch, args, train_metrics, val_metrics
        )
    keep_epochs = {item["epoch"] for item in prospective}
    for old in records:
        if old["epoch"] not in keep_epochs:
            old_path = Path(old["path"])
            if old_path.exists():
                old_path.unlink()
    return prospective


def load_model_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = SimpleNamespace(**checkpoint["args"])
    model = build_model(saved_args, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def evaluate_initial_base(
    model,
    val_loader,
    device,
    args,
    checkpoint_val_metrics,
    output_dir,
):
    full_eval_args = SimpleNamespace(**vars(args))
    full_eval_args.max_eval_batches = None
    metrics, _, _ = run_epoch(
        model,
        val_loader,
        device,
        full_eval_args,
        optimizer=None,
        epoch=0,
        disable_local=True,
    )
    checkpoint_mse = float(checkpoint_val_metrics["mse"])
    reloaded_mse = float(metrics["base"]["mse"])
    mse_difference = abs(reloaded_mse - checkpoint_mse)
    result = {
        "reloaded_base_metrics": metrics["base"],
        "checkpoint_validation_metrics": checkpoint_val_metrics,
        "mse_absolute_difference": mse_difference,
        "tolerance": args.baseline_metric_tolerance,
        "within_tolerance": mse_difference <= args.baseline_metric_tolerance,
    }
    save_json(Path(output_dir) / "initial_base_validation_metrics.json", result)
    print("INITIAL_BASE_MSE", f"{metrics['base']['mse']:.10f}")
    print("INITIAL_BASE_RMSE", f"{metrics['base']['rmse']:.10f}")
    print("INITIAL_BASE_CI", metrics["base"]["ci"])
    print("INITIAL_BASE_RM2", metrics["base"]["rm2"])
    print("CHECKPOINT_BASE_MSE", f"{checkpoint_mse:.10f}")
    print("INITIAL_BASE_MSE_DIFFERENCE", f"{mse_difference:.10g}")
    if mse_difference > args.baseline_metric_tolerance:
        raise RuntimeError(
            "reloaded baseline validation MSE differs from checkpoint: "
            f"difference={mse_difference:.10g} "
            f"tolerance={args.baseline_metric_tolerance:.10g}"
        )
    return result


def print_epoch(epoch, train_metrics, val_metrics, warmup):
    print(
        f"EPOCH {epoch:03d} warmup={int(warmup)} "
        f"TRAIN_TOTAL={train_metrics['losses']['total']:.6f} "
        f"TRAIN_BASE_LOSS={train_metrics['losses']['base']:.6f} "
        f"TRAIN_RESIDUAL_LOSS={train_metrics['losses']['residual']:.6f} "
        f"TRAIN_DELTA_PENALTY={train_metrics['losses']['delta_penalty']:.6f} "
        f"TRAIN_LOCAL_VALID_COUNT={train_metrics['local_valid_count']} "
        f"TRAIN_LOCAL_VALID_RATE={train_metrics['local_valid_rate']:.6f} "
        f"TRAIN_BASE_MSE={train_metrics['base']['mse']:.6f} "
        f"TRAIN_FINAL_MSE={train_metrics['final']['mse']:.6f} "
        f"VAL_TOTAL={val_metrics['losses']['total']:.6f} "
        f"VAL_BASE_LOSS={val_metrics['losses']['base']:.6f} "
        f"VAL_RESIDUAL_LOSS={val_metrics['losses']['residual']:.6f} "
        f"VAL_DELTA_PENALTY={val_metrics['losses']['delta_penalty']:.6f} "
        f"VAL_LOCAL_VALID_COUNT={val_metrics['local_valid_count']} "
        f"VAL_LOCAL_VALID_RATE={val_metrics['local_valid_rate']:.6f} "
        f"VAL_BASE_MSE={val_metrics['base']['mse']:.6f} "
        f"VAL_FINAL_MSE={val_metrics['final']['mse']:.6f} "
        f"VAL_DELTA_MEAN={val_metrics['delta_distribution']['delta_mean']:.6f} "
        f"VAL_DELTA_P95={val_metrics['delta_distribution']['delta_p95']:.6f} "
        f"VAL_BOUND_SATURATION={val_metrics['delta_distribution']['bound_saturation_rate']:.6f} "
        f"VAL_CORR_DELTA_RESIDUAL={val_metrics['residual_quality']['corr_delta_true_residual']} "
        f"VAL_WORST_DRUG_DEGRADATION={val_metrics['per_drug_summary']['worst_drug_degradation']:.6f} "
        f"VAL_FRACTION_DRUGS_IMPROVED={val_metrics['per_drug_summary']['fraction_drugs_improved']:.6f}",
        flush=True,
    )


def train_main(args, device):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    save_json(output_dir / "run_config.json", vars(args))

    dataset = build_dataset(args)
    split = load_split(args.split_json)
    split_stats = verify_drug_cold_split(dataset, split)
    print("SPLIT_STATS", json.dumps(split_stats, sort_keys=True))
    train_loader = make_loader(dataset, split["train_indices"], args, shuffle=True)
    val_loader = make_loader(dataset, split["val_indices"], args, shuffle=False)

    model = build_model(args, device)
    init_report = None
    if args.global_init_mode == "baseline_checkpoint":
        init_report = load_global_checkpoint(
            model, args.init_global_checkpoint, args.split_json
        )
        save_json(output_dir / "global_checkpoint_load_report.json", init_report)
        evaluate_initial_base(
            model,
            val_loader,
            device,
            args,
            init_report["checkpoint_val_metrics"],
            output_dir,
        )
        if args.base_warmup_epochs != 0:
            print(
                "WARNING baseline_checkpoint normally uses --base_warmup_epochs 0; "
                f"current value is {args.base_warmup_epochs}",
                flush=True,
            )
    elif args.baseline_reference_checkpoint:
        config_report = compare_global_config(
            args, args.baseline_reference_checkpoint
        )
        save_json(output_dir / "global_config_match_report.json", config_report)
    else:
        print(
            "WARNING GLOBAL_CONFIG_MATCH_REPORT unavailable: "
            "--baseline_reference_checkpoint was not provided",
            flush=True,
        )
    optimizer, parameter_counts = build_optimizer(model, args)
    print("PARAMETER_COUNTS", json.dumps(parameter_counts, sort_keys=True))

    history = []
    pool = []
    ensemble_pool = []
    best_mse = float("inf")
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        warmup = epoch <= args.base_warmup_epochs
        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            device,
            args,
            optimizer=optimizer,
            epoch=epoch,
            disable_local=warmup,
        )
        val_metrics, _, val_drugs = run_epoch(
            model,
            val_loader,
            device,
            args,
            optimizer=None,
            epoch=epoch,
            disable_local=warmup,
        )
        val_drugs.to_csv(
            validation_dir / "per_drug_metrics_latest.csv", index=False
        )
        print_epoch(epoch, train_metrics, val_metrics, warmup)
        history.append(
            {
                "epoch": epoch,
                "warmup": warmup,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        save_json(output_dir / "history.json", history)
        save_checkpoint(
            output_dir / "latest_model.pt",
            model,
            optimizer,
            epoch,
            args,
            train_metrics,
            val_metrics,
        )
        pool = update_checkpoint_pool(
            pool,
            args.checkpoint_pool_size,
            checkpoint_dir,
            model,
            optimizer,
            epoch,
            args,
            train_metrics,
            val_metrics,
        )
        ensemble_pool = update_spaced_ensemble_pool(
            ensemble_pool,
            args.top_k_checkpoints,
            checkpoint_dir,
            model,
            optimizer,
            epoch,
            args,
            train_metrics,
            val_metrics,
        )

        current_mse = val_metrics["final"]["mse"]
        if current_mse < best_mse - args.early_stop_min_delta:
            best_mse = current_mse
            no_improve = 0
            save_checkpoint(
                output_dir / "best_overall_model.pt",
                model,
                optimizer,
                epoch,
                args,
                train_metrics,
                val_metrics,
            )
        else:
            no_improve += 1
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(
                f"EARLY_STOP epoch={epoch} best_val_mse={best_mse:.6f} "
                f"patience={args.early_stop_patience}",
                flush=True,
            )
            break

    robust, ensemble, near_best = select_checkpoints(
        pool,
        args.robust_mse_tolerance,
        args.top_k_checkpoints,
        args.checkpoint_min_epoch_gap,
    )
    if ensemble_pool:
        ensemble = sorted(
            ensemble_pool, key=lambda item: item["overall_mse"]
        )[: args.top_k_checkpoints]
    shutil.copy2(robust["path"], output_dir / "best_robust_model.pt")
    manifest = {
        "best_overall_checkpoint": str(output_dir / "best_overall_model.pt"),
        "best_robust_checkpoint": str(output_dir / "best_robust_model.pt"),
        "robust_source": robust,
        "near_best_candidates": near_best,
        "ensemble_checkpoints": ensemble,
        "spaced_ensemble_pool": ensemble_pool,
        "checkpoint_pool": pool,
        "minimum_epoch_gap": args.checkpoint_min_epoch_gap,
    }
    save_json(output_dir / "checkpoint_selection.json", manifest)

    for name, path in (
        ("best_overall", output_dir / "best_overall_model.pt"),
        ("best_robust", output_dir / "best_robust_model.pt"),
    ):
        selected_model, checkpoint = load_model_checkpoint(path, device)
        metrics, _, drug_table = run_epoch(
            selected_model,
            val_loader,
            device,
            args,
            optimizer=None,
            epoch=checkpoint["epoch"],
            disable_local=False,
        )
        save_json(validation_dir / f"{name}_metrics.json", metrics)
        drug_table.to_csv(
            validation_dir / f"{name}_per_drug_metrics.csv", index=False
        )
    print("TRAINING_COMPLETE", output_dir)
    print("TEST_NOT_RUN", "Use --mode test after hyperparameters are frozen.")


def _metrics_from_rows(rows, delta_limit):
    losses = {
        "total": None,
        "base": float(np.mean((rows.base_pred - rows.label) ** 2)),
        "residual": None,
        "delta_penalty": float(np.mean(rows.local_delta**2)),
        "final": float(np.mean((rows.final_pred - rows.label) ** 2)),
    }
    metrics, drug_table = _epoch_diagnostics(
        rows,
        losses,
        delta_limit,
        int(rows.local_valid.sum()) if "local_valid" in rows else len(rows),
        float(rows.local_valid.mean()) if "local_valid" in rows else 1.0,
        {},
    )
    return metrics, drug_table


def test_main(args, device):
    output_dir = Path(args.output_dir)
    evaluation_dir = Path(args.evaluation_output_dir or output_dir / "test")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args)
    split = load_split(args.split_json)
    split_stats = verify_drug_cold_split(dataset, split)
    evaluation_key = f"{args.evaluation_split}_indices"
    test_loader = make_loader(
        dataset, split[evaluation_key], args, shuffle=False
    )
    split_stats = {
        "evaluation_split": args.evaluation_split,
        **split_stats,
    }

    checkpoint_path = Path(args.checkpoint or output_dir / "best_robust_model.pt")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    single_metrics, single_rows, single_drugs = run_epoch(
        model,
        test_loader,
        device,
        args,
        optimizer=None,
        epoch=checkpoint["epoch"],
        disable_local=False,
    )
    single_rows.to_csv(
        evaluation_dir / "single_checkpoint_test_predictions.csv", index=False
    )
    single_drugs.to_csv(
        evaluation_dir / "single_checkpoint_per_drug_metrics.csv", index=False
    )
    single_drugs.to_csv(evaluation_dir / "per_drug_metrics.csv", index=False)
    save_json(
        evaluation_dir / "single_checkpoint_test_metrics.json",
        {"checkpoint": str(checkpoint_path), "split": split_stats, **single_metrics},
    )

    if args.ensemble_checkpoints:
        ensemble_paths = [Path(value) for value in args.ensemble_checkpoints]
    else:
        manifest_path = Path(
            args.ensemble_manifest or output_dir / "checkpoint_selection.json"
        )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        ensemble_paths = [
            Path(item["path"])
            for item in manifest["ensemble_checkpoints"][: args.top_k_checkpoints]
        ]
    if not ensemble_paths:
        raise RuntimeError("no ensemble checkpoints were selected")

    prediction_frames = []
    for path in ensemble_paths:
        if path.resolve() == checkpoint_path.resolve():
            frame = single_rows.copy()
        else:
            ensemble_model, ensemble_checkpoint = load_model_checkpoint(path, device)
            _, frame, _ = run_epoch(
                ensemble_model,
                test_loader,
                device,
                args,
                optimizer=None,
                epoch=ensemble_checkpoint["epoch"],
                disable_local=False,
            )
        prediction_frames.append(frame)
    reference = prediction_frames[0][
        ["drug_id", "protein_id", "label", "local_valid"]
    ].copy()
    for frame in prediction_frames[1:]:
        if not np.array_equal(reference.label.to_numpy(), frame.label.to_numpy()):
            raise RuntimeError("ensemble checkpoint prediction order mismatch")
    reference["base_pred"] = np.mean(
        [frame.base_pred.to_numpy(np.float64) for frame in prediction_frames], axis=0
    )
    reference["local_delta"] = np.mean(
        [frame.local_delta.to_numpy(np.float64) for frame in prediction_frames], axis=0
    )
    reference["final_pred"] = np.mean(
        [frame.final_pred.to_numpy(np.float64) for frame in prediction_frames], axis=0
    )
    ensemble_metrics, ensemble_drugs = _metrics_from_rows(
        reference, model.delta_limit
    )
    reference.to_csv(
        evaluation_dir / "ensemble_test_predictions.csv", index=False
    )
    ensemble_drugs.to_csv(
        evaluation_dir / "ensemble_per_drug_metrics.csv", index=False
    )
    save_json(
        evaluation_dir / "ensemble_test_metrics.json",
        {
            "checkpoints": list(map(str, ensemble_paths)),
            "num_checkpoints": len(ensemble_paths),
            "split": split_stats,
            **ensemble_metrics,
        },
    )
    print("TEST_COMPLETE", evaluation_dir)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--pairs_csv", default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", default="data/processed/davis/drug_1d_chemberta2")
    parser.add_argument("--drug_2d_dir", default="data/processed/davis/drug_2d")
    parser.add_argument("--drug_3d_dir", default="data/processed/davis/drug_3d")
    parser.add_argument("--protein_1d_dir", default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_3d_dir", default="data/processed/davis/protein_3d_gvp")
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evaluation_output_dir", default=None)
    parser.add_argument(
        "--evaluation_split",
        choices=("val", "test"),
        default="test",
        help="Use val only for smoke-testing the evaluation pipeline; formal evaluation uses test.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--global_lr", type=float, default=3e-4)
    parser.add_argument("--global_lr_after_init", type=float, default=5e-5)
    parser.add_argument("--local_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--early_stop_patience", type=int, default=60)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--base_warmup_epochs", type=int, default=0)
    parser.add_argument("--residual_huber_beta", type=float, default=0.5)
    parser.add_argument("--residual_loss_weight", type=float, default=1.0)
    parser.add_argument("--delta_penalty_weight", type=float, default=1e-3)
    parser.add_argument("--final_loss_weight", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16"), default="bf16")

    parser.add_argument("--drug_1d_in_dim", type=int, default=768)
    parser.add_argument("--drug_3d_node_in_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pocket_top_k", type=int, default=64)
    parser.add_argument("--interaction_heads", type=int, default=4)
    parser.add_argument("--delta_limit", type=float, default=1.0)
    parser.add_argument("--include_base_pred_in_delta_head", action="store_true")
    parser.add_argument(
        "--global_init_mode",
        choices=("scratch", "baseline_checkpoint"),
        default="scratch",
    )
    parser.add_argument("--init_global_checkpoint", default=None)
    parser.add_argument("--baseline_reference_checkpoint", default=None)
    parser.add_argument(
        "--scratch_optimizer",
        choices=("adam", "adamw"),
        default="adam",
        help="Optimizer used in scratch mode; Davis baseline used Adam.",
    )
    parser.add_argument("--baseline_metric_tolerance", type=float, default=1e-6)

    parser.add_argument("--checkpoint_pool_size", type=int, default=20)
    parser.add_argument("--robust_mse_tolerance", type=float, default=0.02)
    parser.add_argument("--top_k_checkpoints", type=int, default=5)
    parser.add_argument("--checkpoint_min_epoch_gap", type=int, default=5)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--ensemble_manifest", default=None)
    parser.add_argument("--ensemble_checkpoints", nargs="*", default=None)

    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    return parser


def validate_args(args):
    if args.delta_limit <= 0:
        raise ValueError("--delta_limit must be positive")
    if args.residual_huber_beta <= 0:
        raise ValueError("--residual_huber_beta must be positive")
    if args.checkpoint_pool_size < 10:
        raise ValueError("--checkpoint_pool_size must be at least 10")
    if args.top_k_checkpoints <= 0:
        raise ValueError("--top_k_checkpoints must be positive")
    if args.global_lr_after_init <= 0:
        raise ValueError("--global_lr_after_init must be positive")
    if args.baseline_metric_tolerance < 0:
        raise ValueError("--baseline_metric_tolerance cannot be negative")
    if args.mode == "train" and args.global_init_mode == "baseline_checkpoint":
        if not args.init_global_checkpoint:
            raise ValueError(
                "--global_init_mode baseline_checkpoint requires "
                "--init_global_checkpoint"
            )
    if args.mode == "train" and args.global_init_mode == "scratch":
        if args.init_global_checkpoint:
            raise ValueError(
                "--init_global_checkpoint is not allowed with "
                "--global_init_mode scratch"
            )
    if args.final_loss_weight != 0.0:
        print(
            "WARNING final_loss_weight is non-zero; this is experimental and "
            "must not be used in the protected v2 formal experiment.",
            flush=True,
        )


def main():
    args = build_parser().parse_args()
    validate_args(args)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp:
        torch.set_float32_matmul_precision("high")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
    print("DEVICE", device)
    print("MODEL", "Protected Residual E2 v2")
    print("GLOBAL_INIT_MODE", args.global_init_mode)
    print("FINAL_LOSS_WEIGHT", args.final_loss_weight)
    if args.mode == "train":
        train_main(args, device)
    else:
        test_main(args, device)


if __name__ == "__main__":
    main()
