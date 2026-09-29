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

from datasets.collate_p13d_triscale import (
    mdta_collate_fn_p13d_triscale,
    move_batch_to_device_triscale,
)
from datasets.davis_dataset_p13d_triscale import DavisDatasetP13DTriScale
from models.model_p13d_triscale import (
    MyModelMDTAP13DTriScale,
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


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    collect_rows=False,
    compute_full_metrics=True,
    log_interval=100,
    *,
    epoch=None,
    amp=False,
    amp_dtype="bf16",
):
    """Run one epoch with no per-sample CUDA-to-host synchronization."""
    is_train = optimizer is not None
    model.train(is_train)
    total_batches = len(loader)
    start_time = time.perf_counter()
    total_samples = 0
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    fragment_sum = torch.zeros_like(loss_sum)
    pocket_sum = torch.zeros_like(loss_sum)
    fp_pair_sum = torch.zeros_like(loss_sum)
    ar_pair_sum = torch.zeros_like(loss_sum)
    local_missing_sum = torch.zeros_like(loss_sum)

    keep_predictions = compute_full_metrics or collect_rows
    prediction_batches = [] if keep_predictions else None
    base_prediction_batches = [] if keep_predictions else None
    label_batches = [] if keep_predictions else None
    delta_batches = (
        {"fp_delta": [], "ar_delta": [], "multiscale_delta": []}
        if collect_rows
        else None
    )
    drug_ids = [] if collect_rows else None
    protein_ids = [] if collect_rows else None
    autocast_enabled = bool(amp and device.type in {"cuda", "cpu"})
    non_blocking = device.type == "cuda"

    grad_context = torch.enable_grad if is_train else torch.no_grad
    with grad_context():
        for batch_index, batch in enumerate(loader, start=1):
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
            label = batch["label"]
            loss = F.mse_loss(output["pred"].float(), label.float())
            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = label.shape[0]
            total_samples += batch_size
            loss_sum = loss_sum + loss.detach() * batch_size
            stats = output["stats"]
            fragment_sum = fragment_sum + stats["fragments_per_sample"].float() * batch_size
            pocket_sum = pocket_sum + stats["pockets_per_sample"].float() * batch_size
            fp_pair_sum = fp_pair_sum + stats["fp_pairs_per_sample"].float() * batch_size
            ar_pair_sum = ar_pair_sum + stats["ar_pairs_per_sample"].float() * batch_size
            local_missing_sum = local_missing_sum + stats["local_missing_rate"].float() * batch_size

            if keep_predictions:
                prediction_batches.append(output["pred"].detach())
                base_prediction_batches.append(output["base_pred"].detach())
                label_batches.append(label.detach())
            if collect_rows:
                for key in delta_batches:
                    delta_batches[key].append(output[key].detach())
                drug_ids.extend(batch["drug_id"])
                protein_ids.extend(batch["protein_id"])

            if log_interval > 0 and (
                batch_index % log_interval == 0 or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - start_time
                # One batched synchronization per log interval, never per sample.
                running_loss = (loss_sum / max(total_samples, 1)).detach().item()
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

    # A single compact synchronization for epoch aggregates.
    aggregate = torch.stack(
        [
            loss_sum,
            fragment_sum,
            pocket_sum,
            fp_pair_sum,
            ar_pair_sum,
            local_missing_sum,
        ]
    ).detach().cpu()
    denominator = max(total_samples, 1)
    metrics = {
        "loss": float(aggregate[0] / denominator),
        "fragments_per_sample": float(aggregate[1] / denominator),
        "pockets_per_sample": float(aggregate[2] / denominator),
        "avg_fp_pairs": float(aggregate[3] / denominator),
        "avg_ar_pairs": float(aggregate[4] / denominator),
        "local_missing_rate": float(aggregate[5] / denominator),
    }
    rows = None

    if keep_predictions:
        predictions = torch.cat(prediction_batches, dim=0).float().cpu()
        base_predictions = torch.cat(base_prediction_batches, dim=0).float().cpu()
        labels = torch.cat(label_batches, dim=0).float().cpu()
        if compute_full_metrics:
            metrics.update(compute_regression_metrics(predictions, labels))
            metrics["base_pred_mse"] = float(
                torch.mean((base_predictions - labels) ** 2)
            )
            metrics["final_pred_mse"] = metrics["mse"]

        if collect_rows:
            fp_delta = torch.cat(delta_batches["fp_delta"], dim=0).float().cpu().view(-1).tolist()
            ar_delta = torch.cat(delta_batches["ar_delta"], dim=0).float().cpu().view(-1).tolist()
            multiscale_delta = (
                torch.cat(delta_batches["multiscale_delta"], dim=0)
                .float()
                .cpu()
                .view(-1)
                .tolist()
            )
            labels_list = labels.view(-1).tolist()
            base_list = base_predictions.view(-1).tolist()
            prediction_list = predictions.view(-1).tolist()
            rows = [
                {
                    "drug_id": drug_id,
                    "protein_id": protein_id,
                    "label": labels_list[index],
                    "base_pred": base_list[index],
                    "final_pred": prediction_list[index],
                    "fp_delta": fp_delta[index],
                    "ar_delta": ar_delta[index],
                    "multiscale_delta": multiscale_delta[index],
                }
                for index, (drug_id, protein_id) in enumerate(
                    zip(drug_ids, protein_ids)
                )
            ]
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
        ("output_dir", "outputs/davis_p13d_triscale_fixedsplit"),
        ("drug_fragment_cache", "data/processed/davis/multiscale/drug_brics_fragments.pt"),
        ("protein_pocket_cache", "data/processed/davis/multiscale/protein_pockets_p2rank_top3.pt"),
        ("active_scales", "global"),
        ("cross_scale_mode", "none"),
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
    ]
    for name, default in float_defaults:
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--top_k_fp_for_ar", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    scales = parse_active_scales(args.active_scales)
    if scales != {"global"} and (
        not Path(args.drug_fragment_cache).exists()
        or not Path(args.protein_pocket_cache).exists()
    ):
        raise SystemExit("local scales require both offline cache files")
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
    model = MyModelMDTAP13DTriScale(
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
    ).to(device)
    groups = {
        "GLOBAL_PARAMS": sum(value.numel() for value in model.global_model.parameters()),
        "FP_PARAMS": sum(
            value.numel()
            for name, value in model.named_parameters()
            if name.startswith("fp_") or "fragment" in name or "pocket" in name
        ),
        "AR_PARAMS": sum(
            value.numel()
            for name, value in model.named_parameters()
            if name.startswith("ar_") or "atom_" in name or "residue" in name
        ),
        "TOTAL_TRAINABLE_PARAMS": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
    }
    print("ACTIVE_SCALES", sorted(scales))
    print("CROSS_SCALE_MODE", args.cross_scale_mode)
    print("DRUG_FRAGMENT_CACHE", args.drug_fragment_cache)
    print("PROTEIN_POCKET_CACHE", args.protein_pocket_cache)
    print("MAX_FRAGMENTS", args.max_fragments)
    print("MAX_ATOMS_PER_FRAGMENT", args.max_atoms_per_fragment)
    print("MAX_POCKETS", args.max_pockets)
    print("MAX_RESIDUES_PER_POCKET", args.max_residues_per_pocket)
    print("AR_REGION_CHUNK_SIZE", args.ar_region_chunk_size)
    print("TOP_K_FP_FOR_AR", args.top_k_fp_for_ar)
    print("AR_MODE", "full_t4" if args.top_k_fp_for_ar is None else "coarse_to_fine")
    print("AMP", args.amp)
    print("AMP_DTYPE", args.amp_dtype)
    for name, value in groups.items():
        print(name, value)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best = float("inf")
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
        )
        assert val_rows is None
        print(
            f"EPOCH {epoch} TRAIN_LOSS={train_metrics['loss']:.6f} "
            f"BASE_PRED_MSE={val_metrics['base_pred_mse']:.6f} "
            f"FINAL_PRED_MSE={val_metrics['final_pred_mse']:.6f} "
            f"FP_ALPHA={model.fp_alpha.item():.6f} "
            f"AR_ALPHA={model.ar_alpha.item():.6f} "
            f"MULTISCALE_ALPHA={model.multiscale_alpha.item():.6f} "
            f"LOCAL_MISSING_RATE={val_metrics['local_missing_rate']:.6f} "
            f"AVG_FP_PAIRS={val_metrics['avg_fp_pairs']:.3f} "
            f"AVG_AR_PAIRS={val_metrics['avg_ar_pairs']:.3f}",
            flush=True,
        )
        if val_metrics["rmse"] < best - args.early_stop_min_delta:
            best = val_metrics["rmse"]
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
