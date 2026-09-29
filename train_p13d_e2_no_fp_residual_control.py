# -*- coding: utf-8 -*-
"""
Train no-FP residual control.

Experimental control
--------------------
Same fold-specific E2 checkpoint, same E2 freeze, same final MSE objective and
same early-stopping criterion as Stage A, but NO BRICS/CAVIAR/FP information.

    frozen E2 -> [D, P, z_AR] -> generic residual MLP -> delta_control
    y_final = y_E2 + delta_control

This isolates whether Stage-A improvement comes from genuine fragment-pocket
evidence or simply from adding another residual head on top of frozen E2.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d
from models.model_p13d_e2_no_fp_residual_control import (
    E2NoFPResidualControlDTA,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


class FenwickTree:
    def __init__(self, n: int):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def update(self, i: int) -> None:
        while i <= self.n:
            self.tree[i] += 1
            i += i & -i

    def query(self, i: int) -> int:
        value = 0
        while i > 0:
            value += int(self.tree[i])
            i -= i & -i
        return value


def cindex(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    if len(y) <= 1:
        return 0.0

    unique_p = np.unique(p)
    ranks = {value: i + 1 for i, value in enumerate(unique_p)}
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


def r_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sum(yt * yt) * np.sum(yp * yp)
    if denom == 0:
        return 0.0
    return float(np.sum(yt * yp) ** 2 / denom)


def squared_error_zero(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.sum(y_pred * y_pred)
    if denom == 0:
        return 0.0
    k = np.sum(y_true * y_pred) / denom
    down = np.sum((y_true - y_true.mean()) ** 2)
    if down == 0:
        return 0.0
    return float(1.0 - np.sum((y_true - k * y_pred) ** 2) / down)


def rm2(y_true, y_pred):
    r2 = r_squared_error(y_true, y_pred)
    r02 = squared_error_zero(y_true, y_pred)
    return float(r2 * (1 - np.sqrt(abs(r2 ** 2 - r02 ** 2))))


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    err = p - y
    mse = float(np.mean(err ** 2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(err))),
        "ci": cindex(y, p),
        "rm2": rm2(y, p),
    }


def fmt(prefix: str, m: Dict[str, float]) -> str:
    return (
        f"{prefix} MSE={m['mse']:.6f} | RMSE={m['rmse']:.6f} | "
        f"MAE={m['mae']:.6f} | CI={m['ci']:.6f} | RM2={m['rm2']:.6f}"
    )


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


def build_loaders(args, dataset, split):
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=mdta_collate_fn_p13d,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        Subset(dataset, split["train_indices"]),
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        Subset(dataset, split["val_indices"]),
        shuffle=False,
        **common,
    )
    test_indices = split.get("test_indices", [])
    test_loader = (
        DataLoader(
            Subset(dataset, test_indices),
            shuffle=False,
            **common,
        )
        if test_indices
        else None
    )
    return train_loader, val_loader, test_loader


def load_e2_checkpoint(model, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} has no model_state_dict")

    model.load_e2_checkpoint_state(checkpoint["model_state_dict"])

    print(f"LOADED E2 CHECKPOINT: {checkpoint_path}", flush=True)
    if "epoch" in checkpoint:
        print(f"E2 CHECKPOINT EPOCH: {checkpoint['epoch']}", flush=True)
    if "val_metrics" in checkpoint:
        print(
            "E2 SAVED VAL: "
            + json.dumps(checkpoint["val_metrics"], ensure_ascii=False),
            flush=True,
        )
    return checkpoint


def run_epoch(model, loader, device, optimizer, args):
    training = optimizer is not None
    model.train(training)

    final_preds = []
    e2_preds = []
    base_preds = []
    targets = []

    sum_loss = 0.0
    sum_abs_ar_delta = 0.0
    sum_abs_control_delta = 0.0
    count = 0

    for step, batch in enumerate(loader, 1):
        batch = move_to_device(batch, device)
        target = batch["label"].float()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            details = model(batch, return_details=True)
            prediction = details["pred"].float()
            loss = F.mse_loss(prediction, target)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"NON_FINITE loss at step={step}: {loss.item()}"
                )

            if training:
                loss.backward()

                if step == 1:
                    bad_e2_grads = [
                        name
                        for name, parameter in model.e2.named_parameters()
                        if parameter.grad is not None
                    ]
                    if bad_e2_grads:
                        raise RuntimeError(
                            "Frozen E2 unexpectedly received gradients: "
                            + ", ".join(bad_e2_grads[:10])
                        )

                    control_has_grad = any(
                        p.grad is not None
                        and torch.isfinite(p.grad).all()
                        and p.grad.detach().abs().sum().item() > 0
                        for p in model.control_delta_head.parameters()
                        if p.requires_grad
                    )
                    if not control_has_grad:
                        raise RuntimeError(
                            "Control residual head received no nonzero gradient."
                        )

                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.control_delta_head.parameters(),
                        max_norm=args.grad_clip,
                        error_if_nonfinite=True,
                    )
                else:
                    grad_sq = torch.zeros((), device=device)
                    for p in model.control_delta_head.parameters():
                        if p.grad is not None:
                            grad_sq = grad_sq + p.grad.detach().float().pow(2).sum()
                    grad_norm = grad_sq.sqrt()

                optimizer.step()
            else:
                grad_norm = torch.zeros((), device=device)

        n = target.size(0)
        count += n
        sum_loss += float(loss.detach().item()) * n
        sum_abs_ar_delta += (
            float(details["ar_delta"].detach().abs().mean().item()) * n
        )
        sum_abs_control_delta += (
            float(details["control_delta"].detach().abs().mean().item()) * n
        )

        final_preds.append(details["pred"].detach().float().view(-1).cpu())
        e2_preds.append(details["e2_pred"].detach().float().view(-1).cpu())
        base_preds.append(details["base_pred"].detach().float().view(-1).cpu())
        targets.append(target.detach().float().view(-1).cpu())

        if training and args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"  STEP {step:05d}/{len(loader):05d} "
                f"| LOSS={loss.item():.6f} "
                f"| |AR_DELTA|={details['ar_delta'].detach().abs().mean().item():.4f} "
                f"| |CONTROL_DELTA|={details['control_delta'].detach().abs().mean().item():.4f} "
                f"| GRAD={float(grad_norm):.3f}",
                flush=True,
            )

    y = torch.cat(targets).numpy()
    final_p = torch.cat(final_preds).numpy()
    e2_p = torch.cat(e2_preds).numpy()
    base_p = torch.cat(base_preds).numpy()

    result = {
        "final": metrics(y, final_p),
        "e2": metrics(y, e2_p),
        "base": metrics(y, base_p),
        "loss": sum_loss / max(count, 1),
        "mean_abs_ar_delta": sum_abs_ar_delta / max(count, 1),
        "mean_abs_control_delta": sum_abs_control_delta / max(count, 1),
    }
    return result, y, final_p


def print_epoch(prefix, result):
    print(fmt(f"{prefix} FINAL", result["final"]), flush=True)
    print(fmt(f"{prefix} E2   ", result["e2"]), flush=True)
    print(fmt(f"{prefix} BASE ", result["base"]), flush=True)
    print(
        f"{prefix} DIAG | |AR_DELTA|={result['mean_abs_ar_delta']:.6f} "
        f"| |CONTROL_DELTA|={result['mean_abs_control_delta']:.6f}",
        flush=True,
    )


def save_checkpoint(path, model, optimizer, epoch, train_result, val_result, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_result": train_result,
            "val_result": val_result,
            "args": vars(args),
            "model_version": "E2_NO_FP_RESIDUAL_CONTROL",
        },
        path,
    )


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs_csv", default="data/raw/davis/pairs.csv")
    parser.add_argument(
        "--drug_1d_dir",
        default="data/processed/davis/drug_1d_chemberta2",
    )
    parser.add_argument("--drug_2d_dir", default="data/processed/davis/drug_2d")
    parser.add_argument("--drug_3d_dir", default="data/processed/davis/drug_3d")
    parser.add_argument(
        "--protein_1d_dir",
        default="data/processed/davis/protein_1d_esm2",
    )
    parser.add_argument(
        "--protein_3d_dir",
        default="data/processed/davis/protein_3d_gvp",
    )

    parser.add_argument("--e2_checkpoint", required=True)
    parser.add_argument(
        "--split_json",
        default=None,
        help="If omitted, use split_indices.json next to the E2 checkpoint.",
    )
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=60)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--drug_1d_in_dim", type=int, default=768)
    parser.add_argument("--drug_3d_node_in_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pocket_top_k", type=int, default=64)
    parser.add_argument("--interaction_heads", type=int, default=4)

    parser.add_argument("--allow_overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    e2_checkpoint_path = Path(args.e2_checkpoint)
    if not e2_checkpoint_path.exists():
        raise FileNotFoundError(e2_checkpoint_path)

    if args.split_json is None:
        inferred = e2_checkpoint_path.parent / "split_indices.json"
        if not inferred.exists():
            raise FileNotFoundError(
                f"Could not infer fold split from {inferred}"
            )
        args.split_json = str(inferred)

    output = Path(args.output_dir)
    if output.exists() and not args.allow_overwrite:
        meaningful_existing = [
            p for p in output.iterdir()
            if p.name not in {"train.log", "pid.txt", "launcher.log"}
        ]
        if meaningful_existing:
            raise FileExistsError(
                f"Output directory already contains experiment artifacts: {output}"
            )

    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    if "train_indices" not in split or "val_indices" not in split:
        raise KeyError("split_json must contain train_indices and val_indices.")

    dataset = build_dataset(args)
    train_loader, val_loader, test_loader = build_loaders(args, dataset, split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E2NoFPResidualControlDTA(
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
        freeze_e2=True,
    ).to(device)

    checkpoint = load_e2_checkpoint(model, e2_checkpoint_path, device)
    model.set_e2_frozen(True)

    e2_trainable = sum(
        p.numel() for p in model.e2.parameters() if p.requires_grad
    )
    control_parameters = [
        p for p in model.control_delta_head.parameters() if p.requires_grad
    ]
    control_trainable = sum(p.numel() for p in control_parameters)

    if e2_trainable != 0:
        raise RuntimeError(
            f"E2 should be frozen, but {e2_trainable:,} E2 parameters remain trainable."
        )
    if control_trainable == 0:
        raise RuntimeError("No trainable control residual parameters.")

    print(
        f"DEVICE={device} | TRAIN={len(train_loader.dataset)} "
        f"| VAL={len(val_loader.dataset)} "
        f"| TEST={len(test_loader.dataset) if test_loader else 0}",
        flush=True,
    )
    print(
        f"TRAINABLE PARAMS | E2={e2_trainable:,} (FROZEN) "
        f"| CONTROL={control_trainable:,}",
        flush=True,
    )
    print(
        "MODEL=E2_NO_FP_RESIDUAL_CONTROL "
        "| E2 FROZEN/EVAL "
        "| INPUT=[D,P,z_AR] "
        "| BRICS=OFF | CAVIAR=OFF | FP=OFF "
        "| FINAL LOSS=MSE only",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        control_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.eval()
    first_batch = move_to_device(next(iter(val_loader)), device)
    with torch.no_grad():
        sanity = model(first_batch, return_details=True)
    max_initial_diff = float(
        (sanity["pred"] - sanity["e2_pred"]).abs().max().item()
    )
    print(
        f"SANITY | max_abs(final-e2) before training = {max_initial_diff:.12g}",
        flush=True,
    )
    if max_initial_diff > 1e-7:
        raise RuntimeError("Zero initialization failed for control residual head.")

    init_val, _, _ = run_epoch(model, val_loader, device, None, args)
    print_epoch("INIT-VAL", init_val)

    saved_val = checkpoint.get("val_metrics")
    if isinstance(saved_val, dict) and "mse" in saved_val:
        diff = abs(init_val["e2"]["mse"] - float(saved_val["mse"]))
        print(
            f"E2 REPRO CHECK | checkpoint MSE={float(saved_val['mse']):.6f} "
            f"| reloaded E2 MSE={init_val['e2']['mse']:.6f} "
            f"| abs_diff={diff:.6g}",
            flush=True,
        )
        if diff > 1e-3:
            print(
                "WARNING: E2 validation MSE differs from checkpoint by >1e-3.",
                flush=True,
            )

    history = []
    best_rmse = float("inf")
    best_epoch = -1
    best_val = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        print(
            f"\nEPOCH {epoch}/{args.epochs} "
            f"| E2=FROZEN | CONTROL_LR={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )

        train_result, _, _ = run_epoch(
            model, train_loader, device, optimizer, args
        )
        val_result, val_y, val_p = run_epoch(
            model, val_loader, device, None, args
        )

        print_epoch("TRAIN", train_result)
        print_epoch("VAL  ", val_result)

        history.append(
            {"epoch": epoch, "train": train_result, "val": val_result}
        )
        (output / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        save_checkpoint(
            output / "latest_model.pt",
            model,
            optimizer,
            epoch,
            train_result,
            val_result,
            args,
        )

        current_rmse = val_result["final"]["rmse"]
        improved = current_rmse < best_rmse - args.early_stop_min_delta

        if improved:
            best_rmse = current_rmse
            best_epoch = epoch
            best_val = val_result
            stale = 0

            save_checkpoint(
                output / "best_model.pt",
                model,
                optimizer,
                epoch,
                train_result,
                val_result,
                args,
            )
            np.savez(
                output / "best_val_predictions.npz",
                target=val_y,
                prediction=val_p,
            )
            print(
                f"SAVED BEST | EPOCH={epoch} "
                f"| FINAL_MSE={val_result['final']['mse']:.6f} "
                f"| E2_MSE={val_result['e2']['mse']:.6f}",
                flush=True,
            )
        else:
            stale += 1
            print(
                f"NO IMPROVEMENT | STALE={stale}/{args.early_stop_patience} "
                f"| BEST_EPOCH={best_epoch} | BEST_RMSE={best_rmse:.6f}",
                flush=True,
            )

        if (
            args.early_stop_patience > 0
            and stale >= args.early_stop_patience
        ):
            print(
                f"EARLY STOP | EPOCH={epoch} | BEST_EPOCH={best_epoch}",
                flush=True,
            )
            break

    summary = {
        "model_version": "E2_NO_FP_RESIDUAL_CONTROL",
        "e2_checkpoint": str(e2_checkpoint_path),
        "split_json": args.split_json,
        "e2_frozen": True,
        "brics_used": False,
        "caviar_used": False,
        "fp_used": False,
        "control_input": ["drug_feat", "protein_feat", "local_interaction_feat"],
        "loss": "final_mse_only",
        "best_epoch": best_epoch,
        "best_val": best_val,
        "initial_val": init_val,
    }

    if test_loader is not None and (output / "best_model.pt").exists():
        best_checkpoint = torch.load(
            output / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            best_checkpoint["model_state_dict"],
            strict=True,
        )
        test_result, test_y, test_p = run_epoch(
            model, test_loader, device, None, args
        )
        summary["test"] = test_result
        np.savez(
            output / "test_predictions_best_model.npz",
            target=test_y,
            prediction=test_p,
        )
        print_epoch("TEST ", test_result)

    (output / "best_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nOUTPUT_DIR={output.resolve()}", flush=True)
    print(
        f"BEST_EPOCH={best_epoch} | "
        f"BEST_VAL_FINAL_MSE="
        f"{best_val['final']['mse'] if best_val else float('nan'):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
