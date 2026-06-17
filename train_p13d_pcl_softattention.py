# -*- coding: utf-8 -*-
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d, move_batch_to_device
from models.model_p13d_pcl_softattention import MyModelMDTAP13DPCLSOFTATTENTION


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep deterministic behavior as much as possible.
    # If speed is more important than reproducibility, you may set benchmark=True.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_fixed_subsets(dataset, split_json_path):
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    train_indices = split_info["train_indices"]
    val_indices = split_info["val_indices"]
    n = len(dataset)

    bad_train = [i for i in train_indices if i < 0 or i >= n]
    bad_val = [i for i in val_indices if i < 0 or i >= n]
    if bad_train or bad_val:
        raise ValueError(
            f"split_json indices out of range for dataset size={n}. "
            f"bad_train[:5]={bad_train[:5]}, bad_val[:5]={bad_val[:5]}"
        )

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


class FenwickTree:
    def __init__(self, size: int):
        self.n = size
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def update(self, idx: int, delta: int = 1):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx: int):
        s = 0
        while idx > 0:
            s += self.tree[idx]
            idx -= idx & -idx
        return s

    def range_query(self, left: int, right: int):
        if right < left:
            return 0
        return self.query(right) - self.query(left - 1)


def get_cindex(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = len(y_true)
    if n <= 1:
        return 0.0

    unique_pred = np.unique(y_pred)
    pred_rank_map = {v: i + 1 for i, v in enumerate(unique_pred)}
    pred_ranks = np.array([pred_rank_map[v] for v in y_pred], dtype=np.int64)

    order = np.argsort(y_true, kind="mergesort")
    y_sorted = y_true[order]
    r_sorted = pred_ranks[order]

    bit = FenwickTree(len(unique_pred))
    total_prev = 0
    concordant = 0.0
    comparable = 0.0

    start = 0
    while start < n:
        end = start
        while end < n and y_sorted[end] == y_sorted[start]:
            end += 1

        for k in range(start, end):
            r = r_sorted[k]
            num_less = bit.query(r - 1)
            num_equal = bit.range_query(r, r)
            concordant += num_less + 0.5 * num_equal
            comparable += total_prev

        for k in range(start, end):
            bit.update(r_sorted[k], 1)
            total_prev += 1

        start = end

    if comparable == 0:
        return 0.0
    return float(concordant / comparable)


def r_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    mult = np.sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) ** 2
    denom = np.sum((y_true - y_true_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2)

    if denom == 0:
        return 0.0
    return float(mult / denom)


def squared_error_zero(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    denom = np.sum(y_pred * y_pred)
    if denom == 0:
        return 0.0

    k = np.sum(y_true * y_pred) / denom
    y_true_mean = np.mean(y_true)

    upp = np.sum((y_true - k * y_pred) ** 2)
    down = np.sum((y_true - y_true_mean) ** 2)

    if down == 0:
        return 0.0
    return float(1 - upp / down)


def get_rm2(y_true, y_pred):
    r2 = r_squared_error(y_true, y_pred)
    r02 = squared_error_zero(y_true, y_pred)
    return float(r2 * (1 - np.sqrt(abs(r2 ** 2 - r02 ** 2))))


def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.view(-1).detach().cpu()
    targets = targets.view(-1).detach().cpu()

    mse = torch.mean((preds - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(preds - targets))

    preds_np = preds.numpy()
    targets_np = targets.numpy()

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "ci": float(get_cindex(targets_np, preds_np)),
        "rm2": float(get_rm2(targets_np, preds_np)),
    }


def build_subset_lookup(dataset, subset):
    sub_df = dataset.df.iloc[list(subset.indices)].copy()
    lookup = {}
    for row in sub_df.itertuples(index=False):
        lookup[(str(row.drug_id), str(row.protein_id))] = float(row.label)
    return lookup


def build_batch_affinity_matrix(batch, lookup, device):
    drug_ids = batch["drug_id"]
    protein_ids = batch["protein_id"]
    B = len(drug_ids)

    affinity = torch.zeros((B, B), dtype=torch.float32, device=device)
    mask = torch.zeros((B, B), dtype=torch.bool, device=device)

    for i, d in enumerate(drug_ids):
        for j, p in enumerate(protein_ids):
            key = (str(d), str(p))
            if key in lookup:
                affinity[i, j] = float(lookup[key])
                mask[i, j] = True

    diag_ok = torch.all(torch.diag(mask))
    if not bool(diag_ok.item()):
        missing_idx = [i for i in range(B) if not bool(mask[i, i].item())]
        raise ValueError(f"Diagonal pair missing in lookup for indices: {missing_idx}")

    return affinity, mask


def build_dataloaders(args):
    dataset = DavisDatasetP13D(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.protein_3d_dir,
        drug_3d_dir=args.drug_3d_dir,
        use_drug_3d=True,
    )

    train_set, val_set = build_fixed_subsets(dataset, args.split_json)
    train_lookup = build_subset_lookup(dataset, train_set)
    val_lookup = build_subset_lookup(dataset, val_set)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    pin_memory = bool(torch.cuda.is_available())

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mdta_collate_fn_p13d,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mdta_collate_fn_p13d,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )

    return dataset, train_set, val_set, train_loader, val_loader, train_lookup, val_lookup


def build_model(args, device):
    return MyModelMDTAP13DPCLSOFTATTENTION(
        drug_1d_in_dim=args.drug_1d_in_dim,
        drug_3d_node_in_dim=args.drug_3d_node_in_dim,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=args.hidden_dim,
        contrastive_dim=args.contrastive_dim,
        dropout=args.dropout,
        task="regression",
        temperature_init=args.temperature_init,
        affinity_temperature=args.affinity_temperature,
        pcl_temperature_init=args.pcl_temperature_init,
        drug_pcl_temperature_init=args.drug_pcl_temperature_init,
        labelsim_tau=args.labelsim_tau,
        labelsim_mix=args.labelsim_mix,
    ).to(device)


def check_finite_loss(total_loss, reg_loss, attn_loss, pcl_loss, step):
    if not torch.isfinite(total_loss):
        raise FloatingPointError(
            f"Non-finite loss at step={step}: "
            f"total={total_loss.item()}, reg={reg_loss.item()}, "
            f"attn={attn_loss.item()}, pcl={pcl_loss.item()}"
        )


def train_one_epoch(model, loader, lookup, criterion, optimizer, device, args, log_interval=100):
    model.train()

    running_total_loss = 0.0
    running_reg_loss = 0.0
    running_attn_loss = 0.0
    running_pcl_loss = 0.0
    running_protein_pcl_loss = 0.0
    running_drug_pcl_loss = 0.0
    running_valid_per_row = 0.0
    running_valid_offdiag = 0.0
    running_diag_prior_mass = 0.0

    all_preds, all_targets = [], []

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        affinity_matrix, affinity_mask = build_batch_affinity_matrix(batch, lookup, device)

        optimizer.zero_grad(set_to_none=True)

        pred, aux = model(
            batch,
            affinity_matrix=affinity_matrix,
            affinity_mask=affinity_mask,
            return_aux=True,
        )

        target = batch["label"].to(device=device, dtype=pred.dtype).view_as(pred)
        reg_loss = criterion(pred, target)
        attn_loss = aux["attn_loss"]
        pcl_loss = aux["pcl_loss"]

        protein_pcl_loss = aux.get("protein_pcl_loss", torch.tensor(0.0, device=device))
        drug_pcl_loss = aux.get("drug_pcl_loss", torch.tensor(0.0, device=device))

        total_loss = reg_loss + args.lambda_attn * attn_loss + args.lambda_pcl * pcl_loss
        check_finite_loss(total_loss, reg_loss, attn_loss, pcl_loss, step)

        total_loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer.step()

        bs = target.size(0)

        running_total_loss += float(total_loss.item()) * bs
        running_reg_loss += float(reg_loss.item()) * bs
        running_attn_loss += float(attn_loss.item()) * bs
        running_pcl_loss += float(pcl_loss.item()) * bs
        running_protein_pcl_loss += float(protein_pcl_loss.item()) * bs
        running_drug_pcl_loss += float(drug_pcl_loss.item()) * bs
        running_valid_per_row += float(aux["mean_valid_per_row"].item()) * bs
        running_valid_offdiag += float(aux["mean_valid_offdiag"].item()) * bs
        running_diag_prior_mass += float(aux["mean_diag_prior_mass"].item()) * bs

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

        if (step + 1) % log_interval == 0:
            batch_rmse = torch.sqrt(reg_loss.detach().clamp_min(0.0))
            print(
                f"  STEP {step + 1}/{len(loader)} | "
                f"TOTAL={total_loss.item():.6f} | REG={reg_loss.item():.6f} | "
                f"ATTN={attn_loss.item():.6f} | PCL={pcl_loss.item():.6f} | "
                f"DRUG_PCL={drug_pcl_loss.item():.6f} | PROT_PCL={protein_pcl_loss.item():.6f} | "
                f"MEAN_GATE={aux['mean_gate'].item():.4f} | "
                f"VALID_ROW={aux['mean_valid_per_row'].item():.2f} | "
                f"VALID_OFFDIAG={aux['mean_valid_offdiag'].item():.2f} | "
                f"DIAG_PRIOR={aux['mean_diag_prior_mass'].item():.4f} | "
                f"LBL_TAU={aux['labelsim_tau'].item():.2f} | "
                f"LBL_MIX={aux['labelsim_mix'].item():.2f} | "
                f"BATCH_RMSE={batch_rmse.item():.6f}"
            )

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_regression_metrics(all_preds, all_targets)
    denom = len(loader.dataset)

    metrics["loss"] = running_total_loss / denom
    metrics["reg_loss"] = running_reg_loss / denom
    metrics["attn_loss"] = running_attn_loss / denom
    metrics["pcl_loss"] = running_pcl_loss / denom
    metrics["protein_pcl_loss"] = running_protein_pcl_loss / denom
    metrics["drug_pcl_loss"] = running_drug_pcl_loss / denom
    metrics["mean_valid_per_row"] = running_valid_per_row / denom
    metrics["mean_valid_offdiag"] = running_valid_offdiag / denom
    metrics["mean_diag_prior_mass"] = running_diag_prior_mass / denom

    return metrics


@torch.no_grad()
def evaluate(model, loader, lookup, criterion, device, args):
    model.eval()

    running_total_loss = 0.0
    running_reg_loss = 0.0
    running_attn_loss = 0.0
    running_pcl_loss = 0.0
    running_protein_pcl_loss = 0.0
    running_drug_pcl_loss = 0.0
    running_valid_per_row = 0.0
    running_valid_offdiag = 0.0
    running_diag_prior_mass = 0.0

    all_preds, all_targets = [], []

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        affinity_matrix, affinity_mask = build_batch_affinity_matrix(batch, lookup, device)

        pred, aux = model(
            batch,
            affinity_matrix=affinity_matrix,
            affinity_mask=affinity_mask,
            return_aux=True,
        )

        target = batch["label"].to(device=device, dtype=pred.dtype).view_as(pred)
        reg_loss = criterion(pred, target)
        attn_loss = aux["attn_loss"]
        pcl_loss = aux["pcl_loss"]

        protein_pcl_loss = aux.get("protein_pcl_loss", torch.tensor(0.0, device=device))
        drug_pcl_loss = aux.get("drug_pcl_loss", torch.tensor(0.0, device=device))

        total_loss = reg_loss + args.lambda_attn * attn_loss + args.lambda_pcl * pcl_loss
        check_finite_loss(total_loss, reg_loss, attn_loss, pcl_loss, step)

        bs = target.size(0)

        running_total_loss += float(total_loss.item()) * bs
        running_reg_loss += float(reg_loss.item()) * bs
        running_attn_loss += float(attn_loss.item()) * bs
        running_pcl_loss += float(pcl_loss.item()) * bs
        running_protein_pcl_loss += float(protein_pcl_loss.item()) * bs
        running_drug_pcl_loss += float(drug_pcl_loss.item()) * bs
        running_valid_per_row += float(aux["mean_valid_per_row"].item()) * bs
        running_valid_offdiag += float(aux["mean_valid_offdiag"].item()) * bs
        running_diag_prior_mass += float(aux["mean_diag_prior_mass"].item()) * bs

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_regression_metrics(all_preds, all_targets)
    denom = len(loader.dataset)

    metrics["loss"] = running_total_loss / denom
    metrics["reg_loss"] = running_reg_loss / denom
    metrics["attn_loss"] = running_attn_loss / denom
    metrics["pcl_loss"] = running_pcl_loss / denom
    metrics["protein_pcl_loss"] = running_protein_pcl_loss / denom
    metrics["drug_pcl_loss"] = running_drug_pcl_loss / denom
    metrics["mean_valid_per_row"] = running_valid_per_row / denom
    metrics["mean_valid_offdiag"] = running_valid_offdiag / denom
    metrics["mean_diag_prior_mass"] = running_diag_prior_mass / denom

    return metrics


def save_split_indices(train_set, val_set, output_dir):
    split_info = {
        "train_indices": list(train_set.indices),
        "val_indices": list(val_set.indices),
    }
    with open(Path(output_dir) / "split_indices.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)


def save_checkpoint(path, model, optimizer, epoch, train_metrics, val_metrics, args):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def print_epoch_metrics(epoch, total_epochs, split_name, metrics):
    print(
        f"[EPOCH {epoch:03d}/{total_epochs:03d}] {split_name:<5}: "
        f"TOTAL={metrics['loss']:.6f} | REG={metrics['reg_loss']:.6f} | "
        f"ATTN={metrics['attn_loss']:.6f} | PCL={metrics['pcl_loss']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f} | "
        f"CI={metrics['ci']:.6f} | RM2={metrics['rm2']:.6f} | "
        f"DRUG_PCL={metrics.get('drug_pcl_loss', 0.0):.6f} | "
        f"PROT_PCL={metrics.get('protein_pcl_loss', 0.0):.6f} | "
        f"VALID_OFFDIAG={metrics.get('mean_valid_offdiag', 0.0):.2f} | "
        f"DIAG_PRIOR={metrics.get('mean_diag_prior_mass', 0.0):.4f}"
    )


def replay_early_stop_state(history, min_delta):
    replay_best = float("inf")
    replay_no_improve = 0

    for item in history:
        rmse = float(item["val"]["rmse"])
        if rmse < replay_best - min_delta:
            replay_best = rmse
            replay_no_improve = 0
        else:
            replay_no_improve += 1

    return replay_best, replay_no_improve


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs_csv", type=str, default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", type=str, default="data/processed/davis/drug_1d_chemberta2")
    parser.add_argument("--drug_3d_dir", type=str, default="data/processed/davis/drug_3d")
    parser.add_argument("--protein_1d_dir", type=str, default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_3d_dir", type=str, default="data/processed/davis/protein_3d_gvp")
    parser.add_argument("--split_json", type=str, default="data/splits/davis_fixed_split_size2.json")
    parser.add_argument("--output_dir", type=str, default="outputs/davis_p13d_pcl_softattention")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)

    parser.add_argument("--early_stop_patience", type=int, default=60)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    parser.add_argument("--drug_1d_in_dim", type=int, default=768)
    parser.add_argument("--drug_3d_node_in_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--contrastive_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--temperature_init", type=float, default=0.07)
    parser.add_argument("--affinity_temperature", type=float, default=1.0)
    parser.add_argument("--pcl_temperature_init", type=float, default=0.07)
    parser.add_argument("--drug_pcl_temperature_init", type=float, default=0.07)
    parser.add_argument("--labelsim_tau", type=float, default=1.0)
    parser.add_argument("--labelsim_mix", type=float, default=0.3)

    # Keep attention KL as a weak auxiliary regularizer.
    parser.add_argument("--lambda_attn", type=float, default=1e-4)
    parser.add_argument("--lambda_pcl", type=float, default=0.1)

    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--resume_optimizer",
        action="store_true",
        help="Load optimizer state when resuming. Use only when checkpoint matches current model/optimizer.",
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset, train_set, val_set, train_loader, val_loader, train_lookup, val_lookup = build_dataloaders(args)

    print("TOTAL SIZE:", len(dataset))
    print("TRAIN SIZE:", len(train_set))
    print("VAL SIZE:", len(val_set))
    print("SPLIT JSON:", args.split_json)
    print("MODEL: drug_1d + drug_3d + protein_1d + protein_3d + PCL + batch-level softattention(label-sim)")
    print(f"LOSS WEIGHTS: lambda_attn={args.lambda_attn}, lambda_pcl={args.lambda_pcl}")

    save_split_indices(train_set, val_set, args.output_dir)

    model = build_model(args, device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_rmse = float("inf")
    best_epoch = -1
    best_val_metrics = None
    best_train_metrics = None
    history = []
    epochs_no_improve = 0
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)

        load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"RESUME LOAD | missing_keys={load_result.missing_keys} | unexpected_keys={load_result.unexpected_keys}")

        if args.resume_optimizer:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("RESUME OPTIMIZER: loaded")
            except Exception as exc:
                print(f"RESUME OPTIMIZER: skipped because loading failed: {exc}")
        else:
            print("RESUME OPTIMIZER: skipped. Use --resume_optimizer to load it.")

        start_epoch = int(ckpt["epoch"]) + 1

        if ckpt.get("val_metrics") is not None:
            best_val_metrics = dict(ckpt["val_metrics"])
            best_val_rmse = float(best_val_metrics.get("rmse", best_val_rmse))
        if ckpt.get("train_metrics") is not None:
            best_train_metrics = dict(ckpt["train_metrics"])
        best_epoch = int(ckpt["epoch"])

        history_path = Path(args.output_dir) / "history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            _, epochs_no_improve = replay_early_stop_state(history, args.early_stop_min_delta)

        print(
            f"RESUMED FROM: {resume_path} | start_epoch={start_epoch} | "
            f"best_val_rmse={best_val_rmse:.6f} | no_improve={epochs_no_improve}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEPOCH {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            train_lookup,
            criterion,
            optimizer,
            device,
            args,
            log_interval=100,
        )
        val_metrics = evaluate(model, val_loader, val_lookup, criterion, device, args)

        print_epoch_metrics(epoch, args.epochs, "TRAIN", train_metrics)
        print_epoch_metrics(epoch, args.epochs, "VAL", val_metrics)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        save_checkpoint(
            Path(args.output_dir) / "latest_model.pt",
            model,
            optimizer,
            epoch,
            train_metrics,
            val_metrics,
            args,
        )

        current_val_rmse = val_metrics["rmse"]
        improved = current_val_rmse < best_val_rmse - args.early_stop_min_delta

        if improved:
            best_val_rmse = current_val_rmse
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_train_metrics = dict(train_metrics)
            epochs_no_improve = 0

            save_checkpoint(
                Path(args.output_dir) / "best_model.pt",
                model,
                optimizer,
                epoch,
                train_metrics,
                val_metrics,
                args,
            )

            print(
                f"  SAVED NEW BEST MODEL | EPOCH={best_epoch:03d} | "
                f"BEST_VAL_RMSE={best_val_metrics['rmse']:.6f} | "
                f"BEST_VAL_MAE={best_val_metrics['mae']:.6f} | "
                f"BEST_VAL_CI={best_val_metrics['ci']:.6f} | "
                f"BEST_VAL_RM2={best_val_metrics['rm2']:.6f}"
            )
        else:
            epochs_no_improve += 1
            print(
                f"  EARLY_STOP COUNTER | NO_IMPROVE={epochs_no_improve}/{args.early_stop_patience} | "
                f"BEST_EPOCH={best_epoch:03d} | BEST_VAL_RMSE={best_val_rmse:.6f} | "
                f"CURRENT_VAL_RMSE={current_val_rmse:.6f}"
            )

        with open(Path(args.output_dir) / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"\nEARLY STOPPING TRIGGERED | BEST_EPOCH={best_epoch:03d} | "
                f"BEST_VAL_RMSE={best_val_rmse:.6f} | "
                f"PATIENCE={args.early_stop_patience} | MIN_DELTA={args.early_stop_min_delta}"
            )
            break

    print("\nTRAINING FINISHED.")

    if best_val_metrics is not None:
        best_summary = {
            "best_epoch": best_epoch,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
        }
        with open(Path(args.output_dir) / "best_summary.json", "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

        print(
            f"BEST MODEL SUMMARY | EPOCH={best_epoch:03d} | "
            f"VAL_RMSE={best_val_metrics['rmse']:.6f} | "
            f"VAL_MAE={best_val_metrics['mae']:.6f} | "
            f"VAL_CI={best_val_metrics['ci']:.6f} | "
            f"VAL_RM2={best_val_metrics['rm2']:.6f}"
        )

    print(f"SAVED OUTPUTS TO: {args.output_dir}")


if __name__ == "__main__":
    main()