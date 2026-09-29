from __future__ import annotations

import argparse
import csv
import json
import random
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
from models.model_p13d_caviar_staged_ar import CaviarStagedARDTA
from train_p13d_caviar_subpocket import (
    fmt_metrics,
    metrics,
    pairwise_ranking_loss,
    require_finite,
    set_seed,
)


def label_strata(target, prediction):
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    squared_error = (prediction - target) ** 2
    result = {"prediction_floor_rate": float(np.mean(prediction <= 5.0001))}
    masks = {
        "eq5": np.isclose(target, 5.0),
        "gt5": target > 5.0 + 1e-6,
        "ge7": target >= 7.0,
    }
    for name, mask in masks.items():
        result[f"n_{name}"] = int(mask.sum())
        result[f"mse_{name}"] = (
            float(squared_error[mask].mean()) if mask.any() else float("nan")
        )
    return result


def ar_losses(details, target, temperature):
    target = target.float()
    base_error = (details["base_pred"].float() - target).pow(2)
    candidate_error = (details["ar_candidate"].float() - target).pow(2)
    final_error = (details["pred"].float() - target).pow(2)
    candidate_benefit = base_error - candidate_error
    actual_benefit = base_error - final_error
    credit_target = torch.sigmoid(candidate_benefit.detach() / temperature)
    credit_loss = F.binary_cross_entropy_with_logits(
        details["ar_gate_logit"].float(), credit_target
    )
    delta_l2 = details["ar_delta_raw"].float().pow(2).mean()
    selected = details["selected_subpockets"]
    sorted_selected = selected.sort(dim=-1).values
    unique_count = torch.ones_like(sorted_selected[..., 0], dtype=torch.float32)
    if sorted_selected.size(-1) > 1:
        unique_count = unique_count + (
            sorted_selected[..., 1:] != sorted_selected[..., :-1]
        ).float().sum(-1)
    diagnostics = {
        "base_mse_loss": base_error.mean(),
        "credit_loss": credit_loss,
        "delta_l2": delta_l2,
        "gate_ar": details["ar_gate"].mean(),
        "benefit_ar": actual_benefit.mean(),
        "candidate_benefit_ar": candidate_benefit.mean(),
        "abs_delta_ar": details["ar_delta"].abs().mean(),
        "abs_raw_delta_ar": details["ar_delta_raw"].abs().mean(),
        "positive_correction_rate": (details["ar_delta"] > 0).float().mean(),
        "selected_unique_ratio": (
            unique_count / float(sorted_selected.size(-1))
        ).mean(),
    }
    return credit_loss, delta_l2, diagnostics


def run_epoch(model, loader, device, optimizer, stage, args):
    training = optimizer is not None
    model.configure_stage(stage, training)
    predictions, targets, drug_ids = [], [], []
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
            details = model.forward_stage(batch, stage)
            raw_prediction = details["pred"].float()
            prediction = raw_prediction
            if not training and args.prediction_floor > 0:
                prediction = prediction.clamp_min(args.prediction_floor)
            mse_loss = F.mse_loss(prediction, target)
            ranking_loss = pairwise_ranking_loss(
                raw_prediction, target, args.ranking_minimum_gap
            )
            if stage == "global":
                diagnostics = {
                    "base_mse_loss": F.mse_loss(
                        details["base_pred"].float(), target
                    ),
                    "credit_loss": mse_loss.new_zeros(()),
                    "delta_l2": mse_loss.new_zeros(()),
                    "gate_ar": mse_loss.new_zeros(()),
                    "benefit_ar": mse_loss.new_zeros(()),
                    "candidate_benefit_ar": mse_loss.new_zeros(()),
                    "abs_delta_ar": mse_loss.new_zeros(()),
                    "abs_raw_delta_ar": mse_loss.new_zeros(()),
                    "positive_correction_rate": mse_loss.new_zeros(()),
                    "selected_unique_ratio": mse_loss.new_zeros(()),
                }
                loss = mse_loss + args.ranking_weight * ranking_loss
            else:
                credit_loss, delta_l2, diagnostics = ar_losses(
                    details, target, args.credit_temperature
                )
                loss = (
                    mse_loss
                    + args.credit_weight * credit_loss
                    + args.delta_l2_weight * delta_l2
                    + args.ranking_weight * ranking_loss
                )
            check_finite = (
                args.finite_check_interval > 0
                and (step == 1 or step % args.finite_check_interval == 0)
            )
            if check_finite:
                require_finite("prediction", raw_prediction, step)
                require_finite("total_loss", loss, step)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
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
            "mse_loss": mse_loss.detach(),
            "ranking_loss": ranking_loss.detach(),
            "grad_norm": grad_norm.detach(),
            **{key: value.detach() for key, value in diagnostics.items()},
        }
        for key, value in values.items():
            sums[key] = sums.get(
                key, torch.zeros((), device=value.device, dtype=torch.float32)
            ) + value.float() * n
        predictions.append(prediction.detach().view(-1))
        targets.append(target.detach().view(-1))
        drug_ids.extend(batch["drug_id"])
        if training and args.log_interval > 0 and step % args.log_interval == 0:
            if stage == "global":
                print(
                    f"  {stage.upper()} STEP {step:05d}/{len(loader):05d} | "
                    f"MSE={mse_loss.item():.6f} | GRAD={grad_norm.item():.3f}"
                )
            else:
                print(
                    f"  {stage.upper()} STEP {step:05d}/{len(loader):05d} | "
                    f"MSE={mse_loss.item():.6f} | "
                    f"GATE_AR={diagnostics['gate_ar'].item():.3f} | "
                    f"AR_BENEFIT={diagnostics['benefit_ar'].item():+.4f} | "
                    f"GRAD={grad_norm.item():.3f}"
                )
    prediction_array = torch.cat(predictions).cpu().numpy()
    target_array = torch.cat(targets).cpu().numpy()
    result = metrics(target_array, prediction_array, drug_ids)
    result.update(label_strata(target_array, prediction_array))
    result.update({key: value.item() / count for key, value in sums.items()})
    return result, target_array, prediction_array


def save_checkpoint(path, model, optimizer, scheduler, epoch, stage, train, val, args):
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_metrics": train,
            "val_metrics": val,
            "args": vars(args),
        },
        path,
    )


def global_snapshot(model):
    names = set(model.GLOBAL_MODULE_NAMES)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.split(".", 1)[0] in names
    }


def max_global_drift(model, reference):
    current = dict(model.named_parameters())
    return max(
        float((current[name].detach().cpu() - value).abs().max())
        for name, value in reference.items()
    )


CSV_METRICS = (
    "mse", "rmse", "ci", "rm2", "mae", "r2", "pearson", "spearman",
    "drug_median_mse", "drug_worst_mse", "mse_eq5", "mse_gt5", "mse_ge7",
    "prediction_floor_rate", "base_mse_loss", "gate_ar", "benefit_ar",
    "candidate_benefit_ar", "abs_delta_ar", "positive_correction_rate",
    "selected_unique_ratio",
)


def train_stage(
    model,
    stage,
    train_loader,
    val_loader,
    device,
    output,
    args,
    epochs,
    patience,
    learning_rate,
    initial_best=None,
):
    stage_dir = output / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    model.configure_stage(stage, True)
    trainable = [p for p in model.stage_parameters(stage) if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    print(
        f"\n{'=' * 88}\nSTART STAGE={stage.upper()} | "
        f"TRAINABLE_PARAMETERS={sum(p.numel() for p in trainable):,} | "
        f"LR={learning_rate:.2e}\n{'=' * 88}"
    )

    best = None if initial_best is None else dict(initial_best)
    best_epoch = -1 if initial_best is None else 0
    stale = 0
    if initial_best is not None:
        save_checkpoint(
            stage_dir / "checkpoint_best_protected.pt",
            model,
            optimizer,
            scheduler,
            0,
            stage,
            {},
            best,
            args,
        )
        print(fmt_metrics("PROTECTED INITIAL VAL (epoch 000) |", best))

    csv_fields = [
        "epoch", "seconds", "lr", "improved", "best_epoch", "stale"
    ] + [
        f"{prefix}_{key}"
        for prefix in ("train", "val", "best_val")
        for key in CSV_METRICS
    ]
    with (stage_dir / "epoch_metrics.csv").open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=csv_fields).writeheader()
    (stage_dir / "epoch_metrics.jsonl").write_text("", encoding="utf-8")

    completed = 0
    for epoch in range(1, epochs + 1):
        completed = epoch
        started = time.time()
        train, _, _ = run_epoch(
            model, train_loader, device, optimizer, stage, args
        )
        val, val_y, val_p = run_epoch(
            model, val_loader, device, None, stage, args
        )
        scheduler.step(val["mse"])
        improved = best is None or val["mse"] < best["mse"] - args.min_delta
        if improved:
            best = dict(val)
            best_epoch = epoch
            stale = 0
            checkpoint_name = (
                "checkpoint_best.pt"
                if stage == "global"
                else "checkpoint_best_protected.pt"
            )
            save_checkpoint(
                stage_dir / checkpoint_name,
                model,
                optimizer,
                scheduler,
                epoch,
                stage,
                train,
                val,
                args,
            )
            np.savez(
                stage_dir / "best_val_predictions.npz",
                target=val_y,
                prediction=val_p,
            )
        else:
            stale += 1
        save_checkpoint(
            stage_dir / "checkpoint_last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            stage,
            train,
            val,
            args,
        )
        row = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "improved": improved,
            "best_epoch": best_epoch,
            "stale": stale,
        }
        for prefix, values in (("train", train), ("val", val), ("best_val", best)):
            for key in CSV_METRICS:
                row[f"{prefix}_{key}"] = values.get(key, "")
        with (stage_dir / "epoch_metrics.csv").open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=csv_fields).writerow(row)
        with (stage_dir / "epoch_metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

        status = "NEW_BEST" if improved else "no improvement"
        print(
            f"\n[{stage.upper()} EPOCH {epoch:03d}/{epochs:03d}] "
            f"TIME={row['seconds']:.1f}s | LR={row['lr']:.2e} | {status}"
        )
        print(fmt_metrics("CURRENT TRAIN |", train))
        print(fmt_metrics("CURRENT VAL   |", val))
        print(fmt_metrics(f"BEST VAL (epoch {best_epoch:03d}) |", best))
        print(
            "LABEL CURRENT | "
            f"EQ5_MSE={val['mse_eq5']:.6f} | "
            f"GT5_MSE={val['mse_gt5']:.6f} | "
            f"GE7_MSE={val['mse_ge7']:.6f} | "
            f"FLOOR={val['prediction_floor_rate']:.3f}"
        )
        if stage == "ar":
            print(
                "AR CURRENT | "
                f"GATE={val['gate_ar']:.4f} | "
                f"BENEFIT={val['benefit_ar']:+.6f} | "
                f"CANDIDATE={val['candidate_benefit_ar']:+.6f} | "
                f"DELTA={val['abs_delta_ar']:.6f} | "
                f"POSITIVE={val['positive_correction_rate']:.3f} | "
                f"SELECT_UNIQUE={val['selected_unique_ratio']:.3f}"
            )
        print(f"STALE={stale}/{patience}")
        if patience > 0 and stale >= patience:
            print(
                f"EARLY_STOP STAGE={stage.upper()} | "
                f"best_epoch={best_epoch} best_val_mse={best['mse']:.6f}"
            )
            break

    checkpoint_name = (
        "checkpoint_best.pt"
        if stage == "global"
        else "checkpoint_best_protected.pt"
    )
    checkpoint = torch.load(
        stage_dir / checkpoint_name, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    summary = {
        "stage": stage,
        "best_epoch": best_epoch,
        "best_val": best,
        "epochs_completed": completed,
        "stopped_early": stale >= patience > 0,
        "checkpoint": str(stage_dir / checkpoint_name),
        "protected_fallback_selected": stage == "ar" and best_epoch == 0,
    }
    (stage_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument("--drug_1d_dir", required=True)
    parser.add_argument("--protein_1d_dir", required=True)
    parser.add_argument("--protein_3d_dir", required=True)
    parser.add_argument("--drug_atom_v2_dir", required=True)
    parser.add_argument("--protein_residue_v2_dir", required=True)
    parser.add_argument("--protein_subpocket_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stage1_epochs", type=int, default=200)
    parser.add_argument("--stage2_epochs", type=int, default=200)
    parser.add_argument("--stage1_patience", type=int, default=20)
    parser.add_argument("--stage2_patience", type=int, default=30)
    parser.add_argument("--global_lr", type=float, default=3e-4)
    parser.add_argument("--ar_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--subpocket_rounds", type=int, default=2)
    parser.add_argument("--pockets_per_fragment", type=int, default=4)
    parser.add_argument("--max_fragments", type=int, default=12)
    parser.add_argument("--max_atoms_per_fragment", type=int, default=24)
    parser.add_argument("--max_subpockets", type=int, default=30)
    parser.add_argument("--max_residues_per_subpocket", type=int, default=48)
    parser.add_argument("--prediction_floor", type=float, default=5.0)
    parser.add_argument("--delta_max", type=float, default=1.0)
    parser.add_argument("--ar_gate_epsilon", type=float, default=0.05)
    parser.add_argument("--credit_weight", type=float, default=0.05)
    parser.add_argument("--credit_temperature", type=float, default=0.1)
    parser.add_argument("--delta_l2_weight", type=float, default=0.01)
    parser.add_argument("--ranking_weight", type=float, default=0.01)
    parser.add_argument("--ranking_minimum_gap", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--finite_check_interval", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--smoke_samples",
        type=int,
        default=0,
        help="Use at most N samples per split for implementation testing; 0 uses all data.",
    )
    parser.add_argument("--allow_overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    output = Path(args.output_dir)
    existing_run = (
        (output / "final_summary.json").exists()
        or (output / "stage_global" / "checkpoint_last.pt").exists()
        or (output / "stage_ar" / "checkpoint_last.pt").exists()
    )
    if existing_run and not args.allow_overwrite:
        raise FileExistsError(
            f"Completed output already exists: {output}. "
            "Choose another directory or pass --allow_overwrite."
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
        train_indices = train_indices[: args.smoke_samples]
        val_indices = val_indices[: args.smoke_samples]
        test_indices = test_indices[: args.smoke_samples]
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
    common_loader = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **common_loader)
    val_loader = DataLoader(val_set, shuffle=False, **common_loader)
    test_loader = DataLoader(test_set, shuffle=False, **common_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaviarStagedARDTA(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        subpocket_rounds=args.subpocket_rounds,
        pockets_per_fragment=args.pockets_per_fragment,
        delta_max=args.delta_max,
        ar_gate_epsilon=args.ar_gate_epsilon,
    ).to(device)
    print(
        f"DEVICE={device} | TRAIN={len(train_set)} | VAL={len(val_set)} | "
        f"TEST={len(test_set)} | TOTAL_PARAMETERS="
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        "MODEL=CAVIAR-SP2-STAGED | Stage1 Global -> "
        "frozen/eval Global + CAVIAR AR -> validation-protected fallback"
    )
    print(
        f"EVAL_CALIBRATION=prediction_floor={args.prediction_floor:.4f} "
        "(validation/test only)"
    )

    stage1 = train_stage(
        model,
        "global",
        train_loader,
        val_loader,
        device,
        output,
        args,
        args.stage1_epochs,
        args.stage1_patience,
        args.global_lr,
    )
    global_checkpoint = torch.load(
        stage1["checkpoint"], map_location=device, weights_only=False
    )
    model.load_state_dict(global_checkpoint["model_state_dict"])
    global_reference = global_snapshot(model)
    global_val, _, _ = run_epoch(
        model, val_loader, device, None, "global", args
    )
    print("\nGLOBAL WARMUP COMPLETE")
    print(fmt_metrics("LOCKED GLOBAL VAL |", global_val))

    # Stage 2 starts from an exact zero residual, so epoch 0 equals Global.
    torch.nn.init.zeros_(model.ar_delta_head[-1].weight)
    torch.nn.init.zeros_(model.ar_delta_head[-1].bias)
    initial_ar_val, initial_y, initial_p = run_epoch(
        model, val_loader, device, None, "ar", args
    )
    initial_difference = abs(initial_ar_val["mse"] - global_val["mse"])
    # Two independent bf16 evaluation passes can differ by a few 1e-4 even
    # with identical parameters. Parameter immutability is checked exactly
    # after Stage 2; this tolerance only guards the zero-residual prediction.
    if initial_difference > 1e-3:
        raise RuntimeError(
            "Stage-2 zero residual does not reproduce the protected Global "
            f"validation MSE: difference={initial_difference:.9g}"
        )
    (output / "stage_ar").mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "stage_ar" / "protected_initial_val_predictions.npz",
        target=initial_y,
        prediction=initial_p,
    )
    stage2 = train_stage(
        model,
        "ar",
        train_loader,
        val_loader,
        device,
        output,
        args,
        args.stage2_epochs,
        args.stage2_patience,
        args.ar_lr,
        initial_best=global_val,
    )
    drift = max_global_drift(model, global_reference)
    if drift != 0.0:
        raise RuntimeError(f"Protected Global parameters drifted by {drift:.9g}")

    final_tests = {}
    for name, stage, checkpoint_path in (
        ("stage1_global", "global", Path(stage1["checkpoint"])),
        ("stage2_protected_ar", "ar", Path(stage2["checkpoint"])),
    ):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics, test_y, test_p = run_epoch(
            model, test_loader, device, None, stage, args
        )
        final_tests[name] = {
            "selection_used_test": False,
            "stage": stage,
            "checkpoint": str(checkpoint_path),
            "metrics": test_metrics,
        }
        np.savez(
            output / f"final_test_predictions_{name}.npz",
            target=test_y,
            prediction=test_p,
        )
        print(fmt_metrics(f"FINAL TEST ({name}) |", test_metrics))
        print(
            f"LABEL TEST ({name}) | "
            f"EQ5_MSE={test_metrics['mse_eq5']:.6f} | "
            f"GT5_MSE={test_metrics['mse_gt5']:.6f} | "
            f"GE7_MSE={test_metrics['mse_ge7']:.6f} | "
            f"FLOOR={test_metrics['prediction_floor_rate']:.3f}"
        )

    selected = (
        "stage2_protected_ar"
        if stage2["best_epoch"] > 0
        else "stage1_global"
    )
    summary = {
        "test_was_not_used_for_checkpoint_selection": True,
        "selected_by_validation": selected,
        "stage1": stage1,
        "stage2": stage2,
        "stage2_initial_mse_difference_from_global": initial_difference,
        "protected_global_max_parameter_drift": drift,
        "final_tests": final_tests,
    }
    (output / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"FINAL SELECTION={selected} | GLOBAL_PARAMETER_DRIFT={drift:.9g}"
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
