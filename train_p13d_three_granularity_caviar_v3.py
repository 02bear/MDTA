from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.collate_p13d_caviar_subpocket_full import (
    mdta_collate_fn_p13d_caviar_subpocket_full,
    move_batch_to_device_caviar_subpocket,
)
from datasets.davis_dataset_p13d_caviar_subpocket import (
    DavisDatasetP13DCaviarSubpocket,
)
from models.model_p13d_three_granularity_caviar_v3 import (
    ThreeGranularityCaviarDTA,
)
from train_p13d_caviar_subpocket import (
    fmt_metrics,
    metrics,
    require_finite,
    set_seed,
)


MODEL_VERSION = "three_grain_caviar_v3"


def mean_or_zero(values, mask):
    if mask.any():
        return values[mask].mean()
    return values.sum() * 0.0


def masked_sample_mean(values, mask):
    """Mean per sample first, then across samples.

    This prevents proteins with more fragment--pocket candidates from receiving
    a larger regularization weight merely because they have more valid edges.
    """
    values = values.float().reshape(values.size(0), -1)
    mask = mask.reshape(mask.size(0), -1)
    weight = mask.to(values.dtype)
    per_sample = (values * weight).sum(-1) / weight.sum(-1).clamp_min(1.0)
    valid_sample = mask.any(-1)
    return mean_or_zero(per_sample, valid_sample)


def clean_censored_regression_loss(mean, target, censor_threshold=5.0):
    """Left-censored auxiliary loss with the natural label proportions."""
    target = target.float().view_as(mean)
    is_floor = target <= censor_threshold + 1e-6
    is_active = ~is_floor
    floor_pointwise = F.relu(mean.float() - censor_threshold).pow(2)
    active_pointwise = F.smooth_l1_loss(
        mean.float(), target, beta=0.5, reduction="none"
    )
    pointwise = torch.where(is_floor, floor_pointwise, active_pointwise)
    return pointwise.mean(), floor_pointwise, active_pointwise, is_floor, is_active


def label_metrics(target, prediction, raw_prediction):
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    raw_prediction = np.asarray(raw_prediction)
    floor = target <= 5.0 + 1e-6
    active = ~floor
    high = target >= 7.0

    def subset_stats(mask, suffix):
        if not np.any(mask):
            return {
                f"n_{suffix}": 0,
                f"mse_{suffix}": 0.0,
                f"prediction_mean_{suffix}": 0.0,
                f"raw_prediction_mean_{suffix}": 0.0,
                f"prediction_bias_{suffix}": 0.0,
            }
        error = prediction[mask] - target[mask]
        return {
            f"n_{suffix}": int(mask.sum()),
            f"mse_{suffix}": float(np.mean(error ** 2)),
            f"prediction_mean_{suffix}": float(prediction[mask].mean()),
            f"raw_prediction_mean_{suffix}": float(raw_prediction[mask].mean()),
            f"prediction_bias_{suffix}": float(error.mean()),
        }

    result = {
        "prediction_floor_rate": float(np.mean(prediction <= 5.0 + 1e-6)),
        "raw_below_floor_rate": float(np.mean(raw_prediction < 5.0)),
        "prediction_mean": float(prediction.mean()),
        "raw_prediction_mean": float(raw_prediction.mean()),
        "prediction_bias": float(np.mean(prediction - target)),
    }
    result.update(subset_stats(floor, "eq5"))
    result.update(subset_stats(active, "gt5"))
    result.update(subset_stats(high, "ge7"))
    overall_mse = float(np.mean((prediction - target) ** 2))
    if floor.any() and active.any():
        result["balanced_floor_active_mse"] = 0.5 * (
            result["mse_eq5"] + result["mse_gt5"]
        )
    else:
        # Smoke subsets may contain only one label group.
        result["balanced_floor_active_mse"] = overall_mse
    return result


def edge_regularizers(details):
    gate = details["fp_edge_gate"].float()
    mask = details["edge_mask"].bool()
    sparsity = masked_sample_mean(gate, mask)
    probability = gate.clamp(1e-6, 1.0 - 1e-6)
    binary_entropy = -(
        probability * probability.log()
        + (1.0 - probability) * (1.0 - probability).log()
    )
    entropy = masked_sample_mean(binary_entropy, mask)
    return sparsity, entropy


def multiview_info_nce(embeddings, temperature, enabled):
    """Symmetric InfoNCE over AR, FP and Global representations."""
    if not enabled or embeddings.size(0) < 2:
        return embeddings.sum() * 0.0
    views = F.normalize(embeddings.float(), dim=-1)
    labels = torch.arange(views.size(0), device=views.device)
    losses = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        logits = views[:, left] @ views[:, right].t()
        logits = logits / temperature
        losses.append(F.cross_entropy(logits, labels))
        losses.append(F.cross_entropy(logits.t(), labels))
    return torch.stack(losses).mean()


def valid_gate_mean(gate, valid=None):
    if gate is None:
        raise ValueError("Expected a gate tensor, received None")
    gate = gate.float()
    if valid is None:
        return gate.mean()
    valid = valid.bool()
    if valid.any():
        return gate[valid].mean()
    return gate.sum() * 0.0


def selection_diagnostics(details, active_threshold):
    edge_mask = details["edge_mask"].bool()
    ar_gate = details["ar_edge_gate"].float()
    fp_gate = details["fp_edge_gate"].float()
    interaction_entropy = details["interaction_entropy"].float()

    pocket_mask = details["pocket_edge_mask"].bool()
    n_pocket = pocket_mask.size(-1)
    eye = torch.eye(
        n_pocket, dtype=torch.bool, device=pocket_mask.device
    )[None]
    pocket_nonself = pocket_mask & ~eye
    pocket_weight = details["pocket_edge_weight"].float()

    return {
        "mean_ar_edge_gate": masked_sample_mean(ar_gate, edge_mask),
        "mean_fp_edge_gate": masked_sample_mean(fp_gate, edge_mask),
        "active_edge_fraction": masked_sample_mean(
            (fp_gate > active_threshold).float(), edge_mask
        ),
        "mean_interaction_entropy": masked_sample_mean(
            interaction_entropy, edge_mask
        ),
        "mean_pocket_nonself_weight": masked_sample_mean(
            pocket_weight, pocket_nonself
        ),
        "active_pocket_edge_fraction": masked_sample_mean(
            (pocket_weight > active_threshold).float(), pocket_nonself
        ),
        "mean_atom_update_gate": valid_gate_mean(
            details["atom_update_gate"], details["atom_update_valid"]
        ),
        "mean_residue_update_gate": valid_gate_mean(
            details["residue_update_gate"], details["residue_update_valid"]
        ),
        "mean_global_drug_gate": details["global_drug_gate"].float().mean(),
        "mean_global_protein_gate": details[
            "global_protein_gate"
        ].float().mean(),
    }


def run_epoch(model, loader, device, optimizer, args):
    training = optimizer is not None
    model.train(training)
    predictions, raw_predictions, targets = [], [], []
    drug_ids = []
    sums, count = {}, 0
    floor_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    active_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    floor_count = 0
    active_count = 0

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
            details = model(
                batch,
                return_details=True,
                compute_contrastive=args.contrastive_weight > 0.0,
            )
            mean = details["affinity_mean"].float()
            prediction = mean
            target_view = target.view_as(mean)
            main_mse = F.mse_loss(mean, target_view)
            ar_aux_loss = F.mse_loss(
                details["ar_prediction"].float(), target_view
            )
            fp_aux_loss = F.mse_loss(
                details["fp_prediction"].float(), target_view
            )
            global_aux_loss = F.mse_loss(
                details["global_prediction"].float(), target_view
            )
            (
                censored_loss,
                floor_pointwise,
                active_pointwise,
                is_floor,
                is_active,
            ) = clean_censored_regression_loss(
                mean, target, args.censor_threshold
            )
            edge_sparsity, edge_entropy = edge_regularizers(details)
            contrastive_loss = multiview_info_nce(
                details["contrast_embeddings"],
                args.contrastive_temperature,
                enabled=args.contrastive_weight > 0.0,
            )
            loss = (
                main_mse
                + args.aux_head_weight
                * (ar_aux_loss + fp_aux_loss + global_aux_loss)
                + args.censored_aux_weight * censored_loss
                + args.edge_sparsity_weight * edge_sparsity
                + args.edge_entropy_weight * edge_entropy
                + args.contrastive_weight * contrastive_loss
            )
            diagnostics = selection_diagnostics(
                details, args.edge_active_threshold
            )

            check_finite = (
                args.finite_check_interval > 0
                and (step == 1 or step % args.finite_check_interval == 0)
            )
            if check_finite:
                for name in (
                    "affinity_mean",
                    "ar_prediction",
                    "fp_prediction",
                    "global_prediction",
                    "ar_embedding",
                    "fp_embedding",
                    "global_embedding",
                    "ar_edge_tokens",
                    "fp_edge_tokens",
                    "fp_edge_gate",
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
            "main_mse_loss": main_mse.detach(),
            "ar_aux_loss": ar_aux_loss.detach(),
            "fp_aux_loss": fp_aux_loss.detach(),
            "global_aux_loss": global_aux_loss.detach(),
            "censored_loss": censored_loss.detach(),
            "edge_sparsity_loss": edge_sparsity.detach(),
            "edge_entropy_loss": edge_entropy.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "grad_norm": grad_norm.detach(),
            **{key: value.detach() for key, value in diagnostics.items()},
        }
        for key, value in values.items():
            sums[key] = sums.get(
                key, torch.zeros((), device=value.device, dtype=torch.float32)
            ) + value.float() * n

        floor_loss_sum += floor_pointwise[is_floor].sum().detach()
        active_loss_sum += active_pointwise[is_active].sum().detach()
        floor_count += int(is_floor.sum().item())
        active_count += int(is_active.sum().item())
        predictions.append(prediction.detach().view(-1))
        raw_predictions.append(mean.detach().view(-1))
        targets.append(target.detach().view(-1))
        drug_ids.extend(batch["drug_id"])

        if training and args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"  STEP {step:05d}/{len(loader):05d} | "
                f"MAIN={main_mse.item():.6f} | "
                f"AR={ar_aux_loss.item():.4f} | "
                f"FP={fp_aux_loss.item():.4f} | "
                f"GLOBAL={global_aux_loss.item():.4f} | "
                f"EDGE={edge_sparsity.item():.3f} | "
                f"GRAD={grad_norm.item():.3f}",
                flush=True,
            )

    prediction_array = torch.cat(predictions).cpu().numpy()
    raw_array = torch.cat(raw_predictions).cpu().numpy()
    target_array = torch.cat(targets).cpu().numpy()
    result = metrics(target_array, prediction_array, drug_ids)
    result.update(label_metrics(target_array, prediction_array, raw_array))
    result.update({key: value.item() / count for key, value in sums.items()})
    result["floor_one_sided_loss"] = (
        floor_loss_sum.item() / max(floor_count, 1)
    )
    result["active_huber_loss"] = (
        active_loss_sum.item() / max(active_count, 1)
    )
    result["floor_count"] = floor_count
    result["active_count"] = active_count
    return result, target_array, prediction_array, raw_array


LOG_KEYS = (
    "mse", "rmse", "mae", "ci", "rm2", "pearson", "spearman", "r2",
    "bias", "drug_macro_mse", "drug_median_mse", "drug_worst_mse",
    "num_drugs", "prediction_floor_rate", "raw_below_floor_rate",
    "prediction_mean", "raw_prediction_mean", "prediction_bias",
    "n_eq5", "mse_eq5", "prediction_mean_eq5", "raw_prediction_mean_eq5",
    "prediction_bias_eq5", "n_gt5", "mse_gt5", "prediction_mean_gt5",
    "raw_prediction_mean_gt5", "prediction_bias_gt5", "n_ge7", "mse_ge7",
    "prediction_mean_ge7", "raw_prediction_mean_ge7", "prediction_bias_ge7",
    "balanced_floor_active_mse", "objective", "main_mse_loss",
    "ar_aux_loss", "fp_aux_loss", "global_aux_loss", "censored_loss",
    "edge_sparsity_loss", "edge_entropy_loss", "contrastive_loss",
    "floor_one_sided_loss", "active_huber_loss", "floor_count",
    "active_count", "mean_ar_edge_gate", "mean_fp_edge_gate",
    "active_edge_fraction", "mean_interaction_entropy",
    "mean_pocket_nonself_weight", "active_pocket_edge_fraction",
    "mean_atom_update_gate", "mean_residue_update_gate",
    "mean_global_drug_gate", "mean_global_protein_gate", "grad_norm",
)


def save_checkpoint(path, model, optimizer, scheduler, epoch, train, val, args):
    torch.save(
        {
            "model_version": MODEL_VERSION,
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
    parser.add_argument("--interaction_rounds", type=int, default=2)
    parser.add_argument(
        "--ar_pocket_chunk_size", type=int, default=4,
        help="Memory chunk size only; every valid pocket is still evaluated",
    )
    parser.add_argument("--contrastive_projection_dim", type=int, default=128)
    parser.add_argument("--aux_head_weight", type=float, default=0.03,
                        help="Applied separately to each of AR/FP/Global MSE")
    parser.add_argument("--censored_aux_weight", type=float, default=0.05)
    parser.add_argument("--edge_sparsity_weight", type=float, default=0.01)
    parser.add_argument("--edge_entropy_weight", type=float, default=0.005)
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--edge_active_threshold", type=float, default=0.5)
    parser.add_argument("--censor_threshold", type=float, default=5.0)
    parser.add_argument("--balanced_selection_weight", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--finite_check_interval", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke_samples", type=int, default=0)
    parser.add_argument("--allow_overwrite", action="store_true")

    # Accepted only so old launch templates fail gracefully rather than at
    # argument parsing.  v3 uses two outer AR->FP interaction rounds.
    parser.add_argument("--fragment_rounds", type=int, default=0,
                        help="Deprecated and ignored by v3")
    parser.add_argument("--subpocket_rounds", type=int, default=0,
                        help="Deprecated and ignored by v3")
    parser.add_argument("--cross_rounds", type=int, default=0,
                        help="Deprecated and ignored by v3")
    parser.add_argument("--pockets_per_fragment", type=int, default=0,
                        help="Deprecated: v3 scores every valid candidate")
    for name in (
        "max_fragments", "max_atoms_per_fragment", "max_subpockets",
        "max_residues_per_subpocket",
    ):
        parser.add_argument(f"--{name}", type=int, default=0,
                            help="Deprecated: full collate uses batch maxima")
    return parser


def main():
    args = build_parser().parse_args()
    if args.interaction_rounds != 2:
        raise ValueError("v3 implements exactly two AR->FP interaction rounds")
    if args.ar_pocket_chunk_size < 1:
        raise ValueError("--ar_pocket_chunk_size must be positive")
    if not 0.0 <= args.balanced_selection_weight <= 1.0:
        raise ValueError("--balanced_selection_weight must be in [0, 1]")
    if args.contrastive_temperature <= 0.0:
        raise ValueError("--contrastive_temperature must be positive")
    if args.contrastive_projection_dim < 1:
        raise ValueError("--contrastive_projection_dim must be positive")
    if not 0.0 <= args.edge_active_threshold <= 1.0:
        raise ValueError("--edge_active_threshold must be in [0, 1]")
    for name in (
        "aux_head_weight", "censored_aux_weight", "edge_sparsity_weight",
        "edge_entropy_weight", "contrastive_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name} must be non-negative")

    set_seed(args.seed)
    output = Path(args.output_dir)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}")
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
                0, len(indices) - 1, args.smoke_samples, dtype=int
            )
            return [indices[int(position)] for position in positions]
        train_indices = spread(train_indices)
        val_indices = spread(val_indices)
        test_indices = spread(test_indices)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": mdta_collate_fn_p13d_caviar_subpocket_full,
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
        interaction_rounds=args.interaction_rounds,
        ar_pocket_chunk_size=args.ar_pocket_chunk_size,
        contrast_projection_dim=args.contrastive_projection_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"DEVICE={device} | TRAIN={len(train_set)} | VAL={len(val_set)} | "
        f"TEST={len(test_set)} | PARAMS={parameter_count:,}",
        flush=True,
    )
    print(
        "MODEL=THREE-GRAIN-CAVIAR-V3 | dual-token gated AR1->FP1->AR2->FP2 "
        "| full candidates with independently closable FP edges | independent "
        "Global interaction | one main head + three auxiliary heads",
        flush=True,
    )
    print(
        "LOSS=main MSE "
        f"+ {args.aux_head_weight:g} each AR/FP/Global "
        f"+ {args.censored_aux_weight:g} censored "
        f"+ {args.edge_sparsity_weight:g} edge sparsity "
        f"+ {args.edge_entropy_weight:g} binary entropy "
        f"+ {args.contrastive_weight:g} three-pair InfoNCE | "
        f"ALL_POCKETS=True CHUNK_SIZE={args.ar_pocket_chunk_size}",
        flush=True,
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
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train, _, _, _ = run_epoch(
            model, train_loader, device, optimizer, args
        )
        val, val_y, val_p, val_raw = run_epoch(
            model, val_loader, device, None, args
        )
        scheduler.step(val["mse"])
        balanced_component = val["balanced_floor_active_mse"]
        if not math.isfinite(balanced_component):
            balanced_component = val["mse"]
        balanced_score = (
            (1.0 - args.balanced_selection_weight) * val["mse"]
            + args.balanced_selection_weight * balanced_component
        )
        improved_mse = (
            best_mse is None
            or val["mse"] < best_mse["mse"] - args.min_delta
        )
        improved_balanced = balanced_score < best_balanced_score - args.min_delta
        if improved_mse:
            best_mse, best_mse_epoch, stale = dict(val), epoch, 0
            save_checkpoint(
                output / "checkpoint_best_val_mse.pt",
                model, optimizer, scheduler, epoch, train, val, args,
            )
            np.savez(
                output / "best_val_predictions.npz",
                target=val_y, prediction=val_p, raw_prediction=val_raw,
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
            "improved_mse": int(improved_mse),
            "improved_balanced": int(improved_balanced),
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
            item for item in (
                "NEW_BEST_MSE" if improved_mse else "",
                "NEW_BEST_BALANCED" if improved_balanced else "",
            ) if item
        ) or "no improvement"
        print(
            f"EPOCH {epoch:03d} | {fmt_metrics('TRAIN', train)} | "
            f"{fmt_metrics('VAL', val)} | EDGE={val['mean_fp_edge_gate']:.3f} "
            f"| ATOM_GATE={val['mean_atom_update_gate']:.3f} "
            f"| RES_GATE={val['mean_residue_update_gate']:.3f} "
            f"| {status} | STALE={stale}/{args.patience} | "
            f"SEC={time.time() - started:.1f}",
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print(f"EARLY_STOP at epoch {epoch}", flush=True)
            break

    final_tests = {}
    checkpoint_specs = [
        ("minimum_validation_mse", "checkpoint_best_val_mse.pt", best_mse_epoch),
        ("balanced_validation", "checkpoint_best_balanced.pt", best_balanced_epoch),
    ]
    for selection, checkpoint_name, selected_epoch in checkpoint_specs:
        checkpoint_path = output / checkpoint_name
        if not checkpoint_path.exists():
            print(f"SKIP missing checkpoint: {checkpoint_path}", flush=True)
            continue
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("model_version") != MODEL_VERSION:
            raise ValueError(f"Unexpected checkpoint model version: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        test, test_y, test_p, test_raw = run_epoch(
            model, test_loader, device, None, args
        )
        final_tests[selection] = {
            "epoch": selected_epoch,
            "checkpoint": checkpoint_name,
            "metrics": test,
        }
        np.savez(
            output / f"final_test_predictions_{selection}.npz",
            target=test_y, prediction=test_p, raw_prediction=test_raw,
        )
        print(fmt_metrics(f"TEST[{selection}]", test), flush=True)

    summary = {
        "model_version": MODEL_VERSION,
        "test_was_not_used_for_checkpoint_selection": True,
        "best_mse_epoch": best_mse_epoch,
        "best_mse_val": best_mse,
        "best_balanced_epoch": best_balanced_epoch,
        "best_balanced_score": best_balanced_score,
        "best_balanced_val": best_balanced,
        "epochs_completed": epoch,
        "stopped_early": args.patience > 0 and stale >= args.patience,
        "final_tests": final_tests,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"OUTPUT_DIR={output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
