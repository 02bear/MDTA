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

from datasets.conditional_pair_batch_sampler import ConditionalPairBatchSampler
from datasets.collate_p13d_hierarchical import (
    mdta_collate_fn_p13d_hierarchical,
    move_batch_to_device_hierarchical,
)
from datasets.davis_dataset_p13d_hierarchical import DavisDatasetP13DHierarchical
from models.model_p13d_hierarchical_multiscale_v45 import HierarchicalMultiScaleDTA


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
        grouped = {}
        for drug_id, squared_error in zip(drug_ids, error ** 2):
            grouped.setdefault(str(drug_id), []).append(float(squared_error))
        per_drug_mse = np.asarray([np.mean(values) for values in grouped.values()])
        result.update({
            "drug_macro_mse": float(per_drug_mse.mean()),
            "drug_median_mse": float(np.median(per_drug_mse)),
            "drug_worst_mse": float(per_drug_mse.max()),
            "num_drugs": len(grouped),
        })
    return result


def scale_losses(details, target, credit_temperature):
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
    # Gate credit is computed from the complete, ungated proposal.  This
    # avoids the v3 feedback loop where a small gate forced benefit toward
    # zero and therefore forced its own target toward 0.5.
    credit_target = torch.cat([
        torch.sigmoid(ar_candidate_benefit.detach() / credit_temperature),
        torch.sigmoid(fp_candidate_benefit.detach() / credit_temperature),
    ], dim=1)
    credit_loss = F.binary_cross_entropy_with_logits(
        details["scale_gate_logits"].float(), credit_target,
    )
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
        "abs_delta_ar": details["ar_delta"].abs().mean(),
        "abs_delta_fp": details["fp_delta"].abs().mean(),
        "abs_raw_delta_ar": details["ar_delta_raw"].abs().mean(),
        "abs_raw_delta_fp": details["fp_delta_raw"].abs().mean(),
    }
    return credit_loss, delta_reg, diagnostics


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


def rank_n_contrast_loss(features, target, temperature=0.1, tie_epsilon=0.0):
    """Rank-N-Contrast: closer labels should have more similar representations.

    For each anchor i and reference j, the denominator contains samples k whose
    affinity distance from i is at least the distance of j.  This constructs
    relative rankings dynamically and never creates a label-derived target
    similarity matrix.
    """
    features = F.normalize(features.float(), dim=-1)
    target = target.float().view(-1)
    batch_size = target.numel()
    if batch_size < 3:
        zero = features.sum() * 0.0
        return zero, {
            "order_accuracy": zero.detach(),
            "distance_alignment": zero.detach(),
            "feature_std": features.std(dim=0, unbiased=False).mean().detach(),
            "ordered_triplets": zero.detach(),
        }

    similarity = features @ features.T
    logits = similarity / temperature
    label_distance = (target[:, None] - target[None, :]).abs()
    eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    # [anchor, reference, denominator candidate].  B=16 in the experiment,
    # so the fully vectorized B^3 mask is tiny and avoids hundreds of Python
    # loops/GPU launches per training step.
    denominator_mask = (
        (~eye)[:, None, :]
        & (
            label_distance[:, None, :]
            >= label_distance[:, :, None] - tie_epsilon
        )
    )
    denominator_logits = logits[:, None, :].expand(
        -1, batch_size, -1
    ).masked_fill(~denominator_mask, -torch.inf)
    log_denominator = torch.logsumexp(denominator_logits, dim=-1)
    pair_loss = -logits + log_denominator
    loss = pair_loss[~eye].mean()

    # Diagnostic: across all valid triplets, does representation similarity
    # have the inverse ordering of affinity distance?
    distance_difference = (
        label_distance[:, :, None] - label_distance[:, None, :]
    )
    similarity_difference = similarity[:, :, None] - similarity[:, None, :]
    j_is_anchor = eye[:, :, None]
    k_is_anchor = eye[:, None, :]
    j_equals_k = torch.eye(
        batch_size, dtype=torch.bool, device=features.device
    )[None]
    valid_order = (
        ~j_is_anchor
        & ~k_is_anchor
        & ~j_equals_k
        & (distance_difference.abs() > tie_epsilon)
    )
    order_correct = (
        distance_difference[valid_order]
        * similarity_difference[valid_order]
        < 0
    ).float()
    order_accuracy = (
        order_correct.mean() if order_correct.numel() else loss.detach() * 0.0
    )

    pair_mask = ~eye
    flat_similarity = similarity[pair_mask]
    flat_negative_distance = -label_distance[pair_mask]
    centered_similarity = flat_similarity - flat_similarity.mean()
    centered_distance = (
        flat_negative_distance - flat_negative_distance.mean()
    )
    alignment_denominator = (
        centered_similarity.square().sum()
        * centered_distance.square().sum()
    ).sqrt().clamp_min(1e-12)
    distance_alignment = (
        centered_similarity * centered_distance
    ).sum() / alignment_denominator
    diagnostics = {
        "order_accuracy": order_accuracy.detach(),
        "distance_alignment": distance_alignment.detach(),
        "feature_std": features.std(
            dim=0, unbiased=False
        ).mean().detach(),
        "ordered_triplets": valid_order.sum().float().detach(),
    }
    return loss, diagnostics


def scheduled_rnc_weight(base_weight, epoch, warmup_epochs, ramp_epochs):
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(base_weight)
    progress = min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))
    return float(base_weight) * progress


def parameter_gradient_norm(loss, parameters):
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    squared_norm = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.float().square().sum()
    return squared_norm.sqrt()


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


def run_epoch(
    model, loader, device, optimizer, args, rnc_loader=None,
    rnc_ar_weight=0.0, rnc_fp_weight=0.0,
):
    """Run one epoch with a full-coverage task stream and optional RNC stream."""
    training = optimizer is not None
    model.train(training)
    predictions, targets, drug_ids, sums, count = [], [], [], {}, 0
    rnc_enabled = (
        training
        and rnc_loader is not None
        and (rnc_ar_weight > 0 or rnc_fp_weight > 0)
    )
    rnc_iterator = iter(rnc_loader) if rnc_enabled else None
    rnc_sums = {
        "rnc_weighted_loss": 0.0,
        "rnc_ar_loss": 0.0,
        "rnc_fp_loss": 0.0,
        "rnc_ar_order_accuracy": 0.0,
        "rnc_fp_order_accuracy": 0.0,
        "rnc_ar_alignment": 0.0,
        "rnc_fp_alignment": 0.0,
        "rnc_ar_feature_std": 0.0,
        "rnc_fp_feature_std": 0.0,
        "count": 0,
    }
    mode_sums = {
        "protein": {"ar": 0.0, "fp": 0.0, "count": 0},
        "drug": {"ar": 0.0, "fp": 0.0, "count": 0},
    }
    step_ar_weight = rnc_ar_weight * args.rnc_update_interval
    step_fp_weight = rnc_fp_weight * args.rnc_update_interval

    for step, batch in enumerate(loader, 1):
        batch = move_batch_to_device_hierarchical(
            batch, device, non_blocking=True
        )
        target = batch["label"]
        rnc_applied = (
            rnc_enabled and step % args.rnc_update_interval == 0
        )
        batch_mode = "task"
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
            base_mse_loss = F.mse_loss(
                details["base_pred"].float(), target.float()
            )
            ar_mse_loss = F.mse_loss(
                details["pred_after_ar"].float(), target.float()
            )
            credit_loss, delta_reg, diagnostics = scale_losses(
                details, target, args.credit_temperature
            )
            ranking_loss = pairwise_ranking_loss(
                details["pred"], target, args.ranking_minimum_gap,
            )
            task_loss = (
                mse_loss
                + args.base_aux_weight * base_mse_loss
                + args.ar_aux_weight * ar_mse_loss
                + args.credit_weight * credit_loss
                + args.delta_l2_weight * delta_reg
                + args.ranking_weight * ranking_loss
            )

            zero = task_loss.new_zeros(())
            rnc_ar_loss = rnc_fp_loss = zero
            rnc_weighted_loss = zero
            rnc_ar_diagnostics = rnc_fp_diagnostics = {
                "order_accuracy": zero,
                "distance_alignment": zero,
                "feature_std": zero,
            }
            if rnc_applied:
                try:
                    rnc_batch = next(rnc_iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        "RNC loader exhausted before the configured update "
                        "interval was completed"
                    ) from exc
                rnc_batch = move_batch_to_device_hierarchical(
                    rnc_batch, device, non_blocking=True
                )
                rnc_target = rnc_batch["label"]
                unique_drugs = len(set(map(str, rnc_batch["drug_id"])))
                unique_proteins = len(set(map(str, rnc_batch["protein_id"])))
                if (
                    unique_proteins == 1
                    and unique_drugs == rnc_target.size(0)
                ):
                    batch_mode = "protein"
                elif (
                    unique_drugs == 1
                    and unique_proteins == rnc_target.size(0)
                ):
                    batch_mode = "drug"
                else:
                    raise RuntimeError(
                        "Auxiliary RNC batch is not conditional: "
                        f"unique_drugs={unique_drugs}, "
                        f"unique_proteins={unique_proteins}, "
                        f"batch_size={rnc_target.size(0)}"
                    )
                rnc_details = model(rnc_batch, return_details=True)
                rnc_ar_loss, rnc_ar_diagnostics = rank_n_contrast_loss(
                    rnc_details["ar_feat"], rnc_target,
                    args.rnc_temperature, args.rnc_tie_epsilon,
                )
                rnc_fp_loss, rnc_fp_diagnostics = rank_n_contrast_loss(
                    rnc_details["fp_feat"], rnc_target,
                    args.rnc_temperature, args.rnc_tie_epsilon,
                )
                rnc_weighted_loss = (
                    step_ar_weight * rnc_ar_loss
                    + step_fp_weight * rnc_fp_loss
                )
            loss = task_loss + rnc_weighted_loss
            if check_finite:
                require_finite("total_loss", loss, step)
            if training:
                optimizer.zero_grad(set_to_none=True)
                audit_gradients = (
                    rnc_applied
                    and args.rnc_grad_interval > 0
                    and step % args.rnc_grad_interval == 0
                )
                if audit_gradients:
                    local_parameters = list(model.region_builder.parameters())
                    propagation_parameters = list(
                        model.region_propagation.parameters()
                    )
                    task_local_grad = parameter_gradient_norm(
                        task_loss, local_parameters
                    )
                    rnc_local_grad = parameter_gradient_norm(
                        rnc_weighted_loss, local_parameters
                    )
                    task_fp_grad = parameter_gradient_norm(
                        task_loss, propagation_parameters
                    )
                    rnc_fp_grad = parameter_gradient_norm(
                        rnc_weighted_loss, propagation_parameters
                    )
                loss.backward()
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            else:
                audit_gradients = False
                grad_norm_tensor = loss.new_zeros(())

        n = target.size(0)
        count += n
        values = {
            "objective": loss.detach(),
            "mse_loss": mse_loss.detach(),
            "base_mse_loss": base_mse_loss.detach(),
            "ar_mse_loss": ar_mse_loss.detach(),
            "credit_loss": credit_loss.detach(),
            "delta_l2": delta_reg.detach(),
            "ranking_loss": ranking_loss.detach(),
            "task_objective": task_loss.detach(),
            "grad_norm": grad_norm_tensor.detach(),
            **{
                key: value.detach()
                for key, value in diagnostics.items()
            },
        }
        for key, value in values.items():
            value = value.float()
            sums[key] = sums.get(
                key, torch.zeros(
                    (), dtype=torch.float32, device=value.device
                ),
            ) + value * n
        predictions.append(prediction.detach().view(-1))
        targets.append(target.detach().float().view(-1))
        drug_ids.extend(batch["drug_id"])

        if rnc_applied:
            rnc_values = {
                "rnc_weighted_loss": rnc_weighted_loss,
                "rnc_ar_loss": rnc_ar_loss,
                "rnc_fp_loss": rnc_fp_loss,
                "rnc_ar_order_accuracy":
                    rnc_ar_diagnostics["order_accuracy"],
                "rnc_fp_order_accuracy":
                    rnc_fp_diagnostics["order_accuracy"],
                "rnc_ar_alignment":
                    rnc_ar_diagnostics["distance_alignment"],
                "rnc_fp_alignment":
                    rnc_fp_diagnostics["distance_alignment"],
                "rnc_ar_feature_std":
                    rnc_ar_diagnostics["feature_std"],
                "rnc_fp_feature_std":
                    rnc_fp_diagnostics["feature_std"],
            }
            for key, value in rnc_values.items():
                rnc_sums[key] += value.detach().float().item()
            rnc_sums["count"] += 1
            mode_sums[batch_mode]["ar"] += rnc_ar_loss.detach().item()
            mode_sums[batch_mode]["fp"] += rnc_fp_loss.detach().item()
            mode_sums[batch_mode]["count"] += 1

        if (
            training and args.log_interval > 0
            and step % args.log_interval == 0
        ):
            print(
                f"  STEP {step:05d}/{len(loader):05d} | "
                f"AUX={batch_mode.upper()} | MSE={mse_loss.item():.6f} | "
                f"RNC_AR={rnc_ar_loss.item():.4f} "
                f"RNC_FP={rnc_fp_loss.item():.4f} | "
                f"GATE_AR={diagnostics['gate_ar'].item():.3f} | "
                f"GATE_FP={diagnostics['gate_fp'].item():.3f} | "
                f"GRAD={grad_norm_tensor.item():.3f}"
            )
        if training and audit_gradients:
            print(
                f"  RNC_GRAD {step:05d} | "
                f"AR_TASK={task_local_grad.item():.3e} "
                f"AR_RNC={rnc_local_grad.item():.3e} "
                f"RATIO={rnc_local_grad.item() / max(task_local_grad.item(), 1e-12):.3f} | "
                f"FP_TASK={task_fp_grad.item():.3e} "
                f"FP_RNC={rnc_fp_grad.item():.3e} "
                f"RATIO={rnc_fp_grad.item() / max(task_fp_grad.item(), 1e-12):.3f}"
            )

    p = torch.cat(predictions).cpu().numpy()
    y = torch.cat(targets).cpu().numpy()
    result = metrics(y, p, drug_ids)
    result.update({
        key: value.item() / count for key, value in sums.items()
    })
    rnc_count = max(int(rnc_sums["count"]), 1)
    for key, value in rnc_sums.items():
        if key != "count":
            result[key] = value / rnc_count
    result["rnc_updates"] = int(rnc_sums["count"])
    result["rnc_ar_weight"] = float(rnc_ar_weight)
    result["rnc_fp_weight"] = float(rnc_fp_weight)
    result["rnc_ar_step_weight"] = float(step_ar_weight)
    result["rnc_fp_step_weight"] = float(step_fp_weight)
    total_mode_count = sum(value["count"] for value in mode_sums.values())
    for mode, value in mode_sums.items():
        denominator = max(value["count"], 1)
        result[f"rnc_{mode}_batch_fraction"] = (
            value["count"] / max(total_mode_count, 1)
        )
        result[f"rnc_ar_loss_{mode}"] = value["ar"] / denominator
        result[f"rnc_fp_loss_{mode}"] = value["fp"] / denominator
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
    parser.add_argument("--credit_weight", type=float, default=0.05); parser.add_argument("--credit_temperature", type=float, default=0.1)
    parser.add_argument("--base_aux_weight", type=float, default=0.2)
    parser.add_argument("--ar_aux_weight", type=float, default=0.3)
    parser.add_argument("--delta_l2_weight", type=float, default=1e-2); parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--ranking_weight", type=float, default=0.01)
    parser.add_argument("--ranking_minimum_gap", type=float, default=0.5)
    parser.add_argument("--rnc_ar_weight", type=float, default=0.005)
    parser.add_argument("--rnc_fp_weight", type=float, default=0.005)
    parser.add_argument("--rnc_temperature", type=float, default=0.1)
    parser.add_argument("--rnc_tie_epsilon", type=float, default=0.0)
    parser.add_argument("--rnc_warmup_epochs", type=int, default=5)
    parser.add_argument("--rnc_ramp_epochs", type=int, default=10)
    parser.add_argument(
        "--protein_conditioned_fraction", type=float, default=0.7,
        help="Fraction of training batches with one protein and different drugs.",
    )
    parser.add_argument(
        "--conditional_batches_per_epoch", type=int, default=0,
        help="0 uses floor(number of task batches / RNC update interval).",
    )
    parser.add_argument(
        "--rnc_update_interval", type=int, default=4,
        help="Apply one auxiliary conditional RNC batch every N task batches.",
    )
    parser.add_argument(
        "--rnc_grad_interval", type=int, default=200,
        help="Print task-vs-RNC gradient ratios every N train steps; 0 disables.",
    )
    parser.add_argument("--prediction_floor", type=float, default=0.0,
                        help="Validation/test-only calibrated lower prediction bound; 0 disables it.")
    parser.add_argument("--delta_max", type=float, default=1.0,
                        help="Maximum absolute AR/FP residual correction.")
    parser.add_argument("--ar_gate_epsilon", type=float, default=0.05)
    parser.add_argument("--fp_gate_max", type=float, default=0.4)
    parser.add_argument("--balanced_selection_weight", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=30); parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--finite_check_interval", type=int, default=200,
                        help="Check detailed forward tensors every N steps; gradients are checked every step.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA autocast.")
    parser.add_argument("--amp_dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--allow_overwrite", action="store_true",
                        help="Allow an existing output directory containing checkpoints to be reused.")
    args = parser.parse_args()
    if not 0.0 <= args.balanced_selection_weight <= 1.0:
        raise ValueError("--balanced_selection_weight must be in [0, 1]")
    if not 0.0 <= args.protein_conditioned_fraction <= 1.0:
        raise ValueError("--protein_conditioned_fraction must be in [0, 1]")
    if args.rnc_temperature <= 0:
        raise ValueError("--rnc_temperature must be positive")
    if args.rnc_update_interval <= 0:
        raise ValueError("--rnc_update_interval must be positive")
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
    # The task stream exactly matches V4.1: ordinary shuffled, no-replacement
    # batches covering every training pair once per epoch.
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, **loader_options
    )
    default_rnc_batches = max(
        1, len(train_loader) // args.rnc_update_interval
    )
    rnc_batch_sampler = ConditionalPairBatchSampler(
        split["train_indices"], dataset.df, args.batch_size,
        protein_conditioned_fraction=args.protein_conditioned_fraction,
        num_batches=(
            args.conditional_batches_per_epoch
            if args.conditional_batches_per_epoch > 0
            else default_rnc_batches
        ),
        seed=args.seed,
    )
    rnc_generator = torch.Generator()
    rnc_generator.manual_seed(args.seed + 451)
    rnc_loader = DataLoader(
        train_set, batch_sampler=rnc_batch_sampler,
        generator=rnc_generator, **loader_options
    )
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, **loader_options)
    test_loader = (DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers,
                              collate_fn=collate, pin_memory=True,
                              persistent_workers=args.num_workers > 0) if test_set is not None else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalMultiScaleDTA(hidden_dim=args.hidden_dim, dropout=args.dropout,
                                      interaction_heads=args.interaction_heads, region_rounds=args.region_rounds,
                                      delta_max=args.delta_max,
                                      ar_gate_epsilon=args.ar_gate_epsilon,
                                      fp_gate_max=args.fp_gate_max).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    print(f"DEVICE={device} | TRAIN={len(train_set)} | VAL={len(val_set)} | "
          f"TEST={len(test_set) if test_set is not None else 0} | PARAMETERS={sum(p.numel() for p in model.parameters()):,}")
    print("MODEL=V4.5.1 | V4.1 full task stream + auxiliary conditional RNC stream")
    print(
        f"TASK_STREAM=shuffle_without_replacement | "
        f"BATCHES={len(train_loader)} | PAIRS_PER_EPOCH={len(train_set)}"
    )
    print(
        "AUX_CONDITIONAL_BATCHES="
        + json.dumps(rnc_batch_sampler.summary(), sort_keys=True)
    )
    print(
        f"RNC=AR_WEIGHT={args.rnc_ar_weight:.6f} "
        f"FP_WEIGHT={args.rnc_fp_weight:.6f} "
        f"TEMP={args.rnc_temperature:.4f} "
        f"WARMUP={args.rnc_warmup_epochs} RAMP={args.rnc_ramp_epochs} "
        f"UPDATE_INTERVAL={args.rnc_update_interval} "
        f"INTERVAL_COMPENSATION={args.rnc_update_interval}x"
    )
    print(f"EVAL_CALIBRATION=prediction_floor={args.prediction_floor:.4f} (validation/test only)")

    history_path = output / "epoch_metrics.jsonl"; history_path.write_text("")
    csv_path = output / "epoch_metrics.csv"
    csv_fields = [
        "epoch", "seconds", "lr", "improved_mse", "improved_balanced",
        "best_mse_epoch", "best_balanced_epoch", "balanced_score",
        "best_balanced_score",
    ] + [
        f"{split_name}_{key}" for split_name in ("train", "val", "best_val")
        for key in ("mse", "rmse", "ci", "rm2", "mae", "r2", "pearson", "spearman",
                    "drug_macro_mse", "drug_median_mse", "drug_worst_mse",
                    "gate_ar", "gate_fp", "benefit_ar", "benefit_fp",
                    "candidate_benefit_ar", "candidate_benefit_fp",
                    "abs_delta_ar", "abs_delta_fp", "abs_raw_delta_ar", "abs_raw_delta_fp",
                    "rnc_weighted_loss", "rnc_ar_loss", "rnc_fp_loss",
                    "rnc_ar_order_accuracy", "rnc_fp_order_accuracy",
                    "rnc_ar_alignment", "rnc_fp_alignment",
                    "rnc_ar_feature_std", "rnc_fp_feature_std",
                    "rnc_ar_weight", "rnc_fp_weight",
                    "rnc_ar_step_weight", "rnc_fp_step_weight",
                    "rnc_updates",
                    "rnc_protein_batch_fraction", "rnc_drug_batch_fraction",
                    "rnc_ar_loss_protein", "rnc_fp_loss_protein",
                    "rnc_ar_loss_drug", "rnc_fp_loss_drug")
    ]
    with csv_path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=csv_fields).writeheader()
    best_mse, best_mse_epoch = None, -1
    best_balanced, best_balanced_epoch, best_balanced_score = None, -1, float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        rnc_batch_sampler.set_epoch(epoch)
        active_rnc_ar_weight = scheduled_rnc_weight(
            args.rnc_ar_weight, epoch, args.rnc_warmup_epochs,
            args.rnc_ramp_epochs,
        )
        active_rnc_fp_weight = scheduled_rnc_weight(
            args.rnc_fp_weight, epoch, args.rnc_warmup_epochs,
            args.rnc_ramp_epochs,
        )
        started = time.time()
        train, _, _ = run_epoch(
            model, train_loader, device, optimizer, args,
            rnc_loader=rnc_loader,
            rnc_ar_weight=active_rnc_ar_weight,
            rnc_fp_weight=active_rnc_fp_weight,
        )
        val, val_y, val_p = run_epoch(model, val_loader, device, None, args); scheduler.step(val["mse"])
        balanced_score = (
            (1.0 - args.balanced_selection_weight) * val["mse"]
            + args.balanced_selection_weight * val["drug_median_mse"]
        )
        improved_mse = best_mse is None or val["mse"] < best_mse["mse"] - args.min_delta
        improved_balanced = balanced_score < best_balanced_score - args.min_delta
        if improved_mse:
            best_mse, best_mse_epoch, stale = dict(val), epoch, 0
            save_checkpoint(output / "checkpoint_best_val_mse.pt", model, optimizer, scheduler, epoch, train, val, args)
            np.savez(output / "best_val_predictions.npz", target=val_y, prediction=val_p)
            (output / "best_metrics.json").write_text(json.dumps({"epoch": epoch, "train": train, "val": val}, indent=2))
        else: stale += 1
        if improved_balanced:
            best_balanced = dict(val)
            best_balanced_epoch = epoch
            best_balanced_score = balanced_score
            save_checkpoint(
                output / "checkpoint_best_balanced.pt",
                model, optimizer, scheduler, epoch, train, val, args,
            )
            np.savez(
                output / "best_balanced_val_predictions.npz",
                target=val_y, prediction=val_p,
            )
            (output / "best_balanced_metrics.json").write_text(json.dumps({
                "epoch": epoch,
                "balanced_score": balanced_score,
                "balanced_selection_weight": args.balanced_selection_weight,
                "train": train,
                "val": val,
            }, indent=2))
        save_checkpoint(output / "checkpoint_last.pt", model, optimizer, scheduler, epoch, train, val, args)
        row = {"epoch": epoch, "seconds": time.time() - started, "lr": optimizer.param_groups[0]["lr"],
               "improved_mse": improved_mse, "improved_balanced": improved_balanced,
               "best_mse_epoch": best_mse_epoch, "best_balanced_epoch": best_balanced_epoch,
               "balanced_score": balanced_score, "best_balanced_score": best_balanced_score,
               "train": train, "val": val, "best_val": best_mse}
        with history_path.open("a") as handle: handle.write(json.dumps(row) + "\n")
        flat = {"epoch": epoch, "seconds": row["seconds"], "lr": row["lr"],
                "improved_mse": improved_mse, "improved_balanced": improved_balanced,
                "best_mse_epoch": best_mse_epoch, "best_balanced_epoch": best_balanced_epoch,
                "balanced_score": balanced_score, "best_balanced_score": best_balanced_score}
        for split_name, values in (("train", train), ("val", val), ("best_val", best_mse)):
            for key in ("mse", "rmse", "ci", "rm2", "mae", "r2", "pearson", "spearman",
                        "drug_macro_mse", "drug_median_mse", "drug_worst_mse",
                        "gate_ar", "gate_fp", "benefit_ar", "benefit_fp",
                        "candidate_benefit_ar", "candidate_benefit_fp",
                        "abs_delta_ar", "abs_delta_fp", "abs_raw_delta_ar", "abs_raw_delta_fp",
                        "rnc_weighted_loss", "rnc_ar_loss", "rnc_fp_loss",
                        "rnc_ar_order_accuracy", "rnc_fp_order_accuracy",
                        "rnc_ar_alignment", "rnc_fp_alignment",
                        "rnc_ar_feature_std", "rnc_fp_feature_std",
                        "rnc_ar_weight", "rnc_fp_weight",
                        "rnc_ar_step_weight", "rnc_fp_step_weight",
                        "rnc_updates",
                        "rnc_protein_batch_fraction", "rnc_drug_batch_fraction",
                        "rnc_ar_loss_protein", "rnc_fp_loss_protein",
                        "rnc_ar_loss_drug", "rnc_fp_loss_drug"):
                flat[f"{split_name}_{key}"] = values.get(key, "")
        with csv_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=csv_fields).writerow(flat)
        status = " ".join(filter(None, [
            "NEW_BEST_MSE" if improved_mse else "",
            "NEW_BEST_BALANCED" if improved_balanced else "",
        ])) or "no improvement"
        print(f"\n[EPOCH {epoch:03d}/{args.epochs:03d}] TIME={row['seconds']:.1f}s | LR={row['lr']:.2e} | {status}")
        print(fmt_metrics("CURRENT TRAIN |", train)); print(fmt_metrics("CURRENT VAL   |", val))
        print(fmt_metrics(f"BEST MSE VAL (epoch {best_mse_epoch:03d}) |", best_mse))
        print(f"SELECTION | BALANCED={balanced_score:.6f} BEST={best_balanced_score:.6f} "
              f"(epoch {best_balanced_epoch:03d}) | VAL_MEDIAN_DRUG_MSE={val['drug_median_mse']:.6f}")
        print(f"SCALE CURRENT | AR_GATE={val['gate_ar']:.4f} FP_GATE={val['gate_fp']:.4f} | "
              f"AR_BENEFIT={val['benefit_ar']:+.6f} FP_BENEFIT={val['benefit_fp']:+.6f} | "
              f"AR_CAND={val['candidate_benefit_ar']:+.6f} FP_CAND={val['candidate_benefit_fp']:+.6f} | "
              f"AR_DELTA={val['abs_delta_ar']:.6f} FP_DELTA={val['abs_delta_fp']:.6f} | "
              f"AR_RAW={val['abs_raw_delta_ar']:.6f} FP_RAW={val['abs_raw_delta_fp']:.6f} | "
              f"STALE={stale}/{args.patience}")
        print(
            f"RNC TRAIN | WEIGHT_AR={train['rnc_ar_weight']:.6f} "
            f"WEIGHT_FP={train['rnc_fp_weight']:.6f} | "
            f"STEP_AR={train['rnc_ar_step_weight']:.6f} "
            f"STEP_FP={train['rnc_fp_step_weight']:.6f} | "
            f"UPDATES={train['rnc_updates']} | "
            f"LOSS_AR={train['rnc_ar_loss']:.6f} "
            f"LOSS_FP={train['rnc_fp_loss']:.6f} | "
            f"ORDER_AR={train['rnc_ar_order_accuracy']:.4f} "
            f"ORDER_FP={train['rnc_fp_order_accuracy']:.4f} | "
            f"ALIGN_AR={train['rnc_ar_alignment']:+.4f} "
            f"ALIGN_FP={train['rnc_fp_alignment']:+.4f} | "
            f"STD_AR={train['rnc_ar_feature_std']:.6f} "
            f"STD_FP={train['rnc_fp_feature_std']:.6f}"
        )
        print(
            f"RNC MODES | PROTEIN={train['rnc_protein_batch_fraction']:.3f} "
            f"(AR={train['rnc_ar_loss_protein']:.4f} "
            f"FP={train['rnc_fp_loss_protein']:.4f}) | "
            f"DRUG={train['rnc_drug_batch_fraction']:.3f} "
            f"(AR={train['rnc_ar_loss_drug']:.4f} "
            f"FP={train['rnc_fp_loss_drug']:.4f})"
        )
        if args.patience > 0 and stale >= args.patience:
            print(f"EARLY_STOP | best_mse_epoch={best_mse_epoch} best_val_mse={best_mse['mse']:.6f}"); break
    print(fmt_metrics(f"TRAINING FINISHED | BEST MSE VAL (epoch {best_mse_epoch:03d}) |", best_mse))
    final_tests = {}
    if test_loader is not None:
        selections = (
            ("minimum_validation_mse", "checkpoint_best_val_mse.pt", best_mse_epoch),
            ("balanced_validation", "checkpoint_best_balanced.pt", best_balanced_epoch),
        )
        for selection_name, checkpoint_name, selected_epoch in selections:
            checkpoint = torch.load(output / checkpoint_name, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            test_metrics, test_y, test_p = run_epoch(model, test_loader, device, None, args)
            final_tests[selection_name] = {
                "epoch": selected_epoch,
                "checkpoint": checkpoint_name,
                "metrics": test_metrics,
            }
            np.savez(
                output / f"final_test_predictions_{selection_name}.npz",
                target=test_y, prediction=test_p,
            )
            print(fmt_metrics(
                f"FINAL TEST ({selection_name}, epoch {selected_epoch:03d}) |",
                test_metrics,
            ))
        (output / "final_test_metrics.json").write_text(json.dumps({
            "test_was_not_used_for_checkpoint_selection": True,
            "selections": final_tests,
        }, indent=2), encoding="utf-8")
    (output / "training_summary.json").write_text(json.dumps({
        "best_mse_epoch": best_mse_epoch, "best_mse_val": best_mse,
        "best_balanced_epoch": best_balanced_epoch,
        "best_balanced_score": best_balanced_score,
        "best_balanced_val": best_balanced,
        "final_tests": final_tests,
        "epochs_completed": epoch, "stopped_early": stale >= args.patience > 0,
        "best_mse_checkpoint": "checkpoint_best_val_mse.pt",
        "best_balanced_checkpoint": "checkpoint_best_balanced.pt",
        "last_checkpoint": "checkpoint_last.pt",
    }, indent=2), encoding="utf-8")
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
