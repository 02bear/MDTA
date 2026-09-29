from __future__ import annotations

import argparse
import csv
import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.collate_p13d_caviar_subpocket import (
    mdta_collate_fn_p13d_caviar_subpocket,
    move_batch_to_device_caviar_subpocket,
)
from datasets.davis_dataset_p13d_caviar_subpocket import DavisDatasetP13DCaviarSubpocket
from models.model_p13d_three_granularity_caviar import (
    ThreeGranularityCaviarDTA,
)
from train_p13d_caviar_subpocket import (
    fmt_metrics,
    metrics,
    pairwise_ranking_loss,
    rankdata,
    require_finite,
    set_seed,
)


def mean_or_zero(values, mask):
    if mask.any():
        return values[mask].mean()
    return values.sum() * 0.0


def censored_tobit_loss(
    details, target, censor_threshold=5.0, fixed_sigma=0.75
):
    """Balanced normal Tobit NLL for left-censored pKd observations."""
    target = target.float().view_as(details["affinity_mean"])
    mean = details["affinity_mean"].float()
    sigma = torch.full_like(mean, float(fixed_sigma))
    is_censored = target <= censor_threshold + 1e-6
    is_active = ~is_censored

    z = (censor_threshold - mean) / sigma
    censored_nll = -torch.special.log_ndtr(z)
    active_nll = (
        0.5 * ((target - mean) / sigma).pow(2)
        + math.log(fixed_sigma)
        + 0.5 * math.log(2.0 * math.pi)
    )
    floor_loss = mean_or_zero(censored_nll, is_censored)
    active_loss = mean_or_zero(active_nll, is_active)
    if is_censored.any() and is_active.any():
        balanced = 0.5 * floor_loss + 0.5 * active_loss
    else:
        balanced = floor_loss + active_loss
    return balanced, floor_loss, active_loss, sigma


def balanced_activity_loss(logit, target, threshold=5.0):
    active = (target.float().view_as(logit) > threshold + 1e-6)
    positive = F.softplus(-logit.float())
    negative = F.softplus(logit.float())
    positive_loss = mean_or_zero(positive, active)
    negative_loss = mean_or_zero(negative, ~active)
    if active.any() and (~active).any():
        return 0.5 * positive_loss + 0.5 * negative_loss
    return positive_loss + negative_loss


def active_huber_loss(mean, target, threshold=5.0):
    target = target.float().view_as(mean)
    active = target > threshold + 1e-6
    if not active.any():
        return mean.sum() * 0.0
    return F.smooth_l1_loss(mean[active].float(), target[active], beta=0.5)


def binary_auroc(target, probability):
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    positive = target == 1
    negative = target == 0
    if not positive.any() or not negative.any():
        return 0.0
    ranks = rankdata(probability)
    positive_count = positive.sum()
    negative_count = negative.sum()
    return float(
        (
            ranks[positive].sum()
            - positive_count * (positive_count + 1) / 2.0
        )
        / (positive_count * negative_count)
    )


def label_metrics(target, prediction, raw_prediction, activity_probability):
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    raw_prediction = np.asarray(raw_prediction)
    squared = (prediction - target) ** 2
    result = {
        "prediction_floor_rate": float(np.mean(prediction <= 5.0001)),
        "raw_below_floor_rate": float(np.mean(raw_prediction < 5.0)),
        "raw_prediction_mean": float(raw_prediction.mean()),
    }
    masks = {
        "eq5": np.isclose(target, 5.0),
        "gt5": target > 5.0 + 1e-6,
        "ge7": target >= 7.0,
    }
    for name, mask in masks.items():
        result[f"n_{name}"] = int(mask.sum())
        result[f"mse_{name}"] = (
            float(squared[mask].mean()) if mask.any() else float("nan")
        )
    active_target = (target > 5.0 + 1e-6).astype(np.int64)
    result["activity_auroc"] = binary_auroc(
        active_target, activity_probability
    )
    result["activity_brier"] = float(
        np.mean((activity_probability - active_target) ** 2)
    )
    result["balanced_floor_active_mse"] = 0.5 * (
        result["mse_eq5"] + result["mse_gt5"]
    )
    return result


def selection_diagnostics(details):
    edge_mask = details["edge_mask"]
    selected = details["selected_subpockets"]
    soft_weight = details["soft_selection_weight"].float()
    soft_entropy = -(soft_weight.clamp_min(1e-8).log() * soft_weight).sum(-1)
    valid_fragment = edge_mask.any(-1)
    entropy = mean_or_zero(soft_entropy, valid_fragment)

    diversity_values = []
    for batch_index in range(selected.size(0)):
        valid_selected = selected[batch_index][edge_mask[batch_index]]
        if valid_selected.numel():
            diversity_values.append(
                valid_selected.unique().numel() / float(valid_selected.numel())
            )
    diversity = (
        edge_mask.new_tensor(diversity_values, dtype=torch.float32).mean()
        if diversity_values
        else edge_mask.new_zeros((), dtype=torch.float32)
    )
    interaction = details["interaction_weight"].float()
    interaction_entropy = -(
        interaction.clamp_min(1e-8).log() * interaction
    ).sum((-2, -1))
    interaction_entropy = mean_or_zero(interaction_entropy, edge_mask)
    return {
        "selection_entropy": entropy,
        "cross_fragment_pocket_diversity": diversity,
        "interaction_entropy": interaction_entropy,
    }


def run_epoch(model, loader, device, optimizer, args):
    training = optimizer is not None
    model.train(training)
    predictions, raw_predictions, targets = [], [], []
    activity_probabilities, drug_ids = [], []
    sums, count = {}, 0
    for step, batch in enumerate(loader, 1):
        batch = move_batch_to_device_caviar_subpocket(
            batch, device, non_blocking=True
        )
        target = batch["label"].float()
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            details = model(batch, return_details=True)
            mean = details["affinity_mean"].float()
            prediction = mean.clamp_min(args.censor_threshold)
            tobit, floor_nll, active_nll, sigma = censored_tobit_loss(
                details,
                target,
                args.censor_threshold,
                args.tobit_sigma,
            )
            activity = balanced_activity_loss(
                details["activity_logit"], target, args.censor_threshold
            )
            active_huber = active_huber_loss(
                mean, target, args.censor_threshold
            )
            ranking = pairwise_ranking_loss(
                mean, target, args.ranking_minimum_gap
            )
            attention = selection_diagnostics(details)
            loss = (
                tobit
                + args.activity_weight * activity
                + args.active_huber_weight * active_huber
                + args.ranking_weight * ranking
            )
            check_finite = (
                args.finite_check_interval > 0
                and (step == 1 or step % args.finite_check_interval == 0)
            )
            if check_finite:
                for name in (
                    "affinity_mean",
                    "activity_logit",
                    "ar_edge_tokens",
                    "fp_edge_tokens",
                ):
                    require_finite(name, details[name], step)
                require_finite("total_loss", loss, step)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            else:
                grad_norm = loss.new_zeros(())

        n = target.size(0)
        count += n
        values = {
            "objective": loss.detach(),
            "tobit_loss": tobit.detach(),
            "floor_nll": floor_nll.detach(),
            "active_nll": active_nll.detach(),
            "activity_loss": activity.detach(),
            "active_huber_loss": active_huber.detach(),
            "ranking_loss": ranking.detach(),
            "sigma": sigma.mean().detach(),
            "grad_norm": grad_norm.detach(),
            **{key: value.detach() for key, value in attention.items()},
        }
        for key, value in values.items():
            sums[key] = sums.get(
                key, torch.zeros((), device=value.device, dtype=torch.float32)
            ) + value.float() * n
        predictions.append(prediction.detach().view(-1))
        raw_predictions.append(mean.detach().view(-1))
        targets.append(target.detach().view(-1))
        activity_probabilities.append(
            torch.sigmoid(details["activity_logit"].float()).detach().view(-1)
        )
        drug_ids.extend(batch["drug_id"])
        if training and args.log_interval > 0 and step % args.log_interval == 0:
            batch_mse = F.mse_loss(prediction, target.view_as(prediction))
            print(
                f"  STEP {step:05d}/{len(loader):05d} | "
                f"MSE={batch_mse.item():.6f} | "
                f"TOBIT={tobit.item():.4f} | "
                f"ACT={activity.item():.4f} | "
                f"SIGMA={sigma.mean().item():.3f} | "
                f"GRAD={grad_norm.item():.3f}"
            )

    prediction_array = torch.cat(predictions).cpu().numpy()
    raw_array = torch.cat(raw_predictions).cpu().numpy()
    target_array = torch.cat(targets).cpu().numpy()
    activity_array = torch.cat(activity_probabilities).cpu().numpy()
    result = metrics(target_array, prediction_array, drug_ids)
    result.update(
        label_metrics(
            target_array, prediction_array, raw_array, activity_array
        )
    )
    result.update({key: value.item() / count for key, value in sums.items()})
    return result, target_array, prediction_array, raw_array, activity_array


LOG_KEYS = (
    "mse", "rmse", "ci", "rm2", "mae", "r2", "pearson", "spearman",
    "drug_median_mse", "drug_worst_mse", "mse_eq5", "mse_gt5", "mse_ge7",
    "balanced_floor_active_mse", "prediction_floor_rate",
    "raw_below_floor_rate", "activity_auroc", "activity_brier",
    "tobit_loss", "floor_nll", "active_nll", "activity_loss",
    "active_huber_loss", "ranking_loss", "sigma", "selection_entropy",
    "cross_fragment_pocket_diversity", "interaction_entropy",
)


def save_checkpoint(path, model, optimizer, scheduler, epoch, train, val, args):
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


def build_parser():
    parser = argparse.ArgumentParser()
    for name in (
        "pairs_csv", "drug_1d_dir", "protein_1d_dir", "protein_3d_dir",
        "drug_atom_v2_dir", "protein_residue_v2_dir",
        "protein_subpocket_dir", "split_json", "output_dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fragment_rounds", type=int, default=2)
    parser.add_argument("--subpocket_rounds", type=int, default=2)
    parser.add_argument("--cross_rounds", type=int, default=2)
    parser.add_argument("--pockets_per_fragment", type=int, default=4)
    parser.add_argument("--max_fragments", type=int, default=12)
    parser.add_argument("--max_atoms_per_fragment", type=int, default=24)
    parser.add_argument("--max_subpockets", type=int, default=30)
    parser.add_argument("--max_residues_per_subpocket", type=int, default=48)
    parser.add_argument("--censor_threshold", type=float, default=5.0)
    parser.add_argument("--tobit_sigma", type=float, default=0.75)
    parser.add_argument("--activity_weight", type=float, default=0.1)
    parser.add_argument("--active_huber_weight", type=float, default=0.2)
    parser.add_argument("--ranking_weight", type=float, default=0.02)
    parser.add_argument("--ranking_minimum_gap", type=float, default=0.5)
    parser.add_argument("--balanced_selection_weight", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--finite_check_interval", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke_samples", type=int, default=0)
    parser.add_argument("--allow_overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not 0.0 <= args.balanced_selection_weight <= 1.0:
        raise ValueError("--balanced_selection_weight must be in [0, 1]")
    if args.tobit_sigma <= 0:
        raise ValueError("--tobit_sigma must be positive")
    set_seed(args.seed)
    output = Path(args.output_dir)
    if (
        (output / "checkpoint_last.pt").exists()
        and not args.allow_overwrite
    ):
        raise FileExistsError(
            f"Output directory already contains a run: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    dataset = DavisDatasetP13DCaviarSubpocket(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.protein_3d_dir,
        use_drug_2d=False,
        use_drug_3d=False,
        drug_atom_v2_dir=args.drug_atom_v2_dir,
        protein_residue_v2_dir=args.protein_residue_v2_dir,
        protein_subpocket_dir=args.protein_subpocket_dir,
    )
    split = json.loads(Path(args.split_json).read_text())
    train_indices = split["train_indices"]
    val_indices = split["val_indices"]
    test_indices = split["test_indices"]
    if args.smoke_samples > 0:
        def spread(indices):
            if len(indices) <= args.smoke_samples:
                return indices
            positions = np.linspace(
                0, len(indices) - 1, args.smoke_samples, dtype=np.int64
            )
            return [indices[int(position)] for position in positions]

        train_indices = spread(train_indices)
        val_indices = spread(val_indices)
        test_indices = spread(test_indices)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    collate = partial(
        mdta_collate_fn_p13d_caviar_subpocket,
        max_fragments=args.max_fragments,
        max_atoms_per_fragment=args.max_atoms_per_fragment,
        max_subpockets=args.max_subpockets,
        max_residues_per_subpocket=args.max_residues_per_subpocket,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeGranularityCaviarDTA(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        fragment_rounds=args.fragment_rounds,
        subpocket_rounds=args.subpocket_rounds,
        cross_rounds=args.cross_rounds,
        pockets_per_fragment=args.pockets_per_fragment,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    print(
        f"DEVICE={device} | TRAIN={len(train_set)} | VAL={len(val_set)} | "
        f"TEST={len(test_set)} | PARAMETERS="
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        "MODEL=THREE-GRAIN-CAVIAR-V1 | AR evidence -> "
        "BRICS/CAVIAR FP dual graph -> one Global affinity head"
    )
    print(
        "LOSS=balanced left-censored Tobit + activity auxiliary + "
        "active Huber + ranking"
    )

    history = output / "epoch_metrics.jsonl"
    history.write_text("", encoding="utf-8")
    csv_fields = [
        "epoch", "seconds", "lr", "improved_mse", "improved_balanced",
        "best_mse_epoch", "best_balanced_epoch", "balanced_score",
    ] + [
        f"{prefix}_{key}"
        for prefix in ("train", "val", "best_val")
        for key in LOG_KEYS
    ]
    with (output / "epoch_metrics.csv").open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=csv_fields).writeheader()

    best_mse, best_mse_epoch = None, -1
    best_balanced, best_balanced_epoch = None, -1
    best_balanced_score = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train, _, _, _, _ = run_epoch(
            model, train_loader, device, optimizer, args
        )
        val, val_y, val_p, val_raw, val_activity = run_epoch(
            model, val_loader, device, None, args
        )
        scheduler.step(val["mse"])
        balanced_score = (
            (1.0 - args.balanced_selection_weight) * val["mse"]
            + args.balanced_selection_weight
            * val["balanced_floor_active_mse"]
        )
        improved_mse = (
            best_mse is None
            or val["mse"] < best_mse["mse"] - args.min_delta
        )
        improved_balanced = (
            balanced_score < best_balanced_score - args.min_delta
        )
        if improved_mse:
            best_mse, best_mse_epoch, stale = dict(val), epoch, 0
            save_checkpoint(
                output / "checkpoint_best_val_mse.pt",
                model, optimizer, scheduler, epoch, train, val, args,
            )
            np.savez(
                output / "best_val_predictions.npz",
                target=val_y,
                prediction=val_p,
                raw_prediction=val_raw,
                activity_probability=val_activity,
            )
        else:
            stale += 1
        if improved_balanced:
            best_balanced = dict(val)
            best_balanced_epoch = epoch
            best_balanced_score = balanced_score
            save_checkpoint(
                output / "checkpoint_best_balanced.pt",
                model, optimizer, scheduler, epoch, train, val, args,
            )
        save_checkpoint(
            output / "checkpoint_last.pt",
            model, optimizer, scheduler, epoch, train, val, args,
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
        }
        for prefix, values in (
            ("train", train), ("val", val), ("best_val", best_mse)
        ):
            for key in LOG_KEYS:
                row[f"{prefix}_{key}"] = values.get(key, "")
        with (output / "epoch_metrics.csv").open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=csv_fields).writerow(row)
        with history.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

        status = " ".join(
            filter(
                None,
                (
                    "NEW_BEST_MSE" if improved_mse else "",
                    "NEW_BEST_BALANCED" if improved_balanced else "",
                ),
            )
        ) or "no improvement"
        print(
            f"\n[EPOCH {epoch:03d}/{args.epochs:03d}] "
            f"TIME={row['seconds']:.1f}s | LR={row['lr']:.2e} | {status}"
        )
        print(fmt_metrics("CURRENT TRAIN |", train))
        print(fmt_metrics("CURRENT VAL   |", val))
        print(fmt_metrics(f"BEST VAL (epoch {best_mse_epoch:03d}) |", best_mse))
        print(
            "LABEL CURRENT | "
            f"EQ5={val['mse_eq5']:.6f} | "
            f"GT5={val['mse_gt5']:.6f} | "
            f"GE7={val['mse_ge7']:.6f} | "
            f"RAW<5={val['raw_below_floor_rate']:.3f} | "
            f"ACT_AUROC={val['activity_auroc']:.4f}"
        )
        print(
            "HIERARCHY CURRENT | "
            f"FP_DIVERSITY={val['cross_fragment_pocket_diversity']:.3f} | "
            f"SELECT_ENTROPY={val['selection_entropy']:.3f} | "
            f"AR_ENTROPY={val['interaction_entropy']:.3f}"
        )
        print(
            f"SELECTION | BALANCED={balanced_score:.6f} "
            f"BEST={best_balanced_score:.6f} "
            f"(epoch {best_balanced_epoch:03d}) | "
            f"STALE={stale}/{args.patience}"
        )
        if args.patience > 0 and stale >= args.patience:
            print(
                f"EARLY_STOP | best_epoch={best_mse_epoch} "
                f"best_val_mse={best_mse['mse']:.6f}"
            )
            break

    final_tests = {}
    for selection, checkpoint_name, selected_epoch in (
        ("minimum_validation_mse", "checkpoint_best_val_mse.pt", best_mse_epoch),
        ("balanced_validation", "checkpoint_best_balanced.pt", best_balanced_epoch),
    ):
        checkpoint = torch.load(
            output / checkpoint_name, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test, test_y, test_p, test_raw, test_activity = run_epoch(
            model, test_loader, device, None, args
        )
        final_tests[selection] = {
            "epoch": selected_epoch,
            "checkpoint": checkpoint_name,
            "metrics": test,
        }
        np.savez(
            output / f"final_test_predictions_{selection}.npz",
            target=test_y,
            prediction=test_p,
            raw_prediction=test_raw,
            activity_probability=test_activity,
        )
        print(fmt_metrics(
            f"FINAL TEST ({selection}, epoch {selected_epoch:03d}) |", test
        ))
        print(
            f"FINAL LABEL ({selection}) | "
            f"EQ5={test['mse_eq5']:.6f} | GT5={test['mse_gt5']:.6f} | "
            f"GE7={test['mse_ge7']:.6f} | "
            f"RAW<5={test['raw_below_floor_rate']:.3f} | "
            f"ACT_AUROC={test['activity_auroc']:.4f}"
        )

    summary = {
        "test_was_not_used_for_checkpoint_selection": True,
        "best_mse_epoch": best_mse_epoch,
        "best_mse_val": best_mse,
        "best_balanced_epoch": best_balanced_epoch,
        "best_balanced_score": best_balanced_score,
        "best_balanced_val": best_balanced,
        "epochs_completed": epoch,
        "stopped_early": stale >= args.patience > 0,
        "final_tests": final_tests,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
