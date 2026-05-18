# -*- coding: utf-8 -*-
import json
import random
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d, move_batch_to_device

_MODEL_FILE = Path(__file__).resolve().parent / "models" / "model_p13d_softclip_pcl_cross-entropy.py"
_MODEL_SPEC = importlib.util.spec_from_file_location("model_p13d_softclip_pcl_cross_entropy", _MODEL_FILE)
if _MODEL_SPEC is None or _MODEL_SPEC.loader is None:
    raise ImportError(f"Cannot load model module from {_MODEL_FILE}")
_MODEL_MODULE = importlib.util.module_from_spec(_MODEL_SPEC)
_MODEL_SPEC.loader.exec_module(_MODEL_MODULE)
MyModelMDTAP13DSOFTCLIPPCL = _MODEL_MODULE.MyModelMDTAP13DSOFTCLIPPCL


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        raise ValueError(f"split_json indices out of range for dataset size={n}. bad_train[:5]={bad_train[:5]}, bad_val[:5]={bad_val[:5]}")
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
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "ci": float(get_cindex(targets_np, preds_np)), "rm2": float(get_rm2(targets_np, preds_np))}


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
        drug_2d_dir=args.drug_2d_dir,
        use_drug_2d=False,
    )
    train_set, val_set = build_fixed_subsets(dataset, args.split_json)
    train_lookup = build_subset_lookup(dataset, train_set)
    val_lookup = build_subset_lookup(dataset, val_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=mdta_collate_fn_p13d, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=mdta_collate_fn_p13d, pin_memory=True)
    return dataset, train_set, val_set, train_loader, val_loader, train_lookup, val_lookup


def build_model(args, device):
    return MyModelMDTAP13DSOFTCLIPPCL(
        drug_1d_in_dim=args.drug_1d_in_dim,
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
        diag_prior_weight=args.diag_prior_weight,
    ).to(device)


def train_one_epoch(model, loader, lookup, criterion, optimizer, device, args, log_interval=100):
    model.train()
    running_total_loss = running_reg_loss = running_clip_loss = running_pcl_loss = 0.0
    all_preds, all_targets = [], []
    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        affinity_matrix, affinity_mask = build_batch_affinity_matrix(batch, lookup, device)
        optimizer.zero_grad()
        pred, aux = model(batch, affinity_matrix=affinity_matrix, affinity_mask=affinity_mask, return_aux=True)
        target = batch["label"]
        reg_loss = criterion(pred, target)
        clip_loss = aux["clip_loss"]
        pcl_loss = aux["pcl_loss"]
        total_loss = reg_loss + args.lambda_clip * clip_loss + args.lambda_pcl * pcl_loss
        total_loss.backward()
        optimizer.step()
        bs = target.size(0)
        running_total_loss += float(total_loss.item()) * bs
        running_reg_loss += float(reg_loss.item()) * bs
        running_clip_loss += float(clip_loss.item()) * bs
        running_pcl_loss += float(pcl_loss.item()) * bs
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())
        if (step + 1) % log_interval == 0:
            batch_rmse = torch.sqrt(reg_loss.detach())
            print(f"  STEP {step + 1}/{len(loader)} | TOTAL_LOSS={total_loss.item():.6f} | REG_LOSS={reg_loss.item():.6f} | SOFTCLIP_LOSS={clip_loss.item():.6f} | PCL_LOSS={pcl_loss.item():.6f} | MEAN_GATE={aux['mean_gate'].item():.4f} | MEAN_VALID_PER_ROW={aux['mean_valid_per_row'].item():.2f} | MEAN_VALID_OFFDIAG={aux['mean_valid_offdiag'].item():.2f} | MEAN_DIAG_PRIOR_MASS={aux['mean_diag_prior_mass'].item():.4f} | DIAG_W={aux['diag_prior_weight'].item():.2f} | BATCH_RMSE={batch_rmse.item():.6f}")
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_regression_metrics(all_preds, all_targets)
    metrics["loss"] = running_total_loss / len(loader.dataset)
    metrics["reg_loss"] = running_reg_loss / len(loader.dataset)
    metrics["clip_loss"] = running_clip_loss / len(loader.dataset)
    metrics["pcl_loss"] = running_pcl_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, lookup, criterion, device, args):
    model.eval()
    running_total_loss = running_reg_loss = running_clip_loss = running_pcl_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        affinity_matrix, affinity_mask = build_batch_affinity_matrix(batch, lookup, device)
        pred, aux = model(batch, affinity_matrix=affinity_matrix, affinity_mask=affinity_mask, return_aux=True)
        target = batch["label"]
        reg_loss = criterion(pred, target)
        clip_loss = aux["clip_loss"]
        pcl_loss = aux["pcl_loss"]
        total_loss = reg_loss + args.lambda_clip * clip_loss + args.lambda_pcl * pcl_loss
        bs = target.size(0)
        running_total_loss += float(total_loss.item()) * bs
        running_reg_loss += float(reg_loss.item()) * bs
        running_clip_loss += float(clip_loss.item()) * bs
        running_pcl_loss += float(pcl_loss.item()) * bs
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_regression_metrics(all_preds, all_targets)
    metrics["loss"] = running_total_loss / len(loader.dataset)
    metrics["reg_loss"] = running_reg_loss / len(loader.dataset)
    metrics["clip_loss"] = running_clip_loss / len(loader.dataset)
    metrics["pcl_loss"] = running_pcl_loss / len(loader.dataset)
    return metrics


def save_split_indices(train_set, val_set, output_dir):
    split_info = {"train_indices": list(train_set.indices), "val_indices": list(val_set.indices)}
    with open(Path(output_dir) / "split_indices.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)


def save_checkpoint(path, model, optimizer, epoch, train_metrics, val_metrics, args):
    ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "train_metrics": train_metrics, "val_metrics": val_metrics, "args": vars(args)}
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", type=str, default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", type=str, default="data/processed/davis/drug_1d_chemberta2")
    parser.add_argument("--drug_2d_dir", type=str, default="data/processed/davis/drug_2d")
    parser.add_argument("--protein_1d_dir", type=str, default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_3d_dir", type=str, default="data/processed/davis/protein_3d_min")
    parser.add_argument("--split_json", type=str, default="data/splits/davis_fixed_split_size2.json")
    parser.add_argument("--output_dir", type=str, default="outputs/davis_p13d_softclip_pcl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--drug_1d_in_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--contrastive_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature_init", type=float, default=0.07)
    parser.add_argument("--affinity_temperature", type=float, default=1.0)
    parser.add_argument("--pcl_temperature_init", type=float, default=0.07)
    parser.add_argument("--diag_prior_weight", type=float, default=0.5)
    parser.add_argument("--lambda_clip", type=float, default=0.1)
    parser.add_argument("--lambda_pcl", type=float, default=0.1)
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
    print("MODEL: drug_1d + protein_1d + protein_3d + softclip(drug-protein) + pcl(protein1d-protein3d)")
    save_split_indices(train_set, val_set, args.output_dir)
    model = build_model(args, device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_rmse = float("inf")
    best_epoch = -1
    best_val_metrics = best_train_metrics = None
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEPOCH {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, train_lookup, criterion, optimizer, device, args, log_interval=100)
        val_metrics = evaluate(model, val_loader, val_lookup, criterion, device, args)
        print(f"[EPOCH {epoch:03d}/{args.epochs:03d}] TRAIN: TOTAL={train_metrics['loss']:.6f} | REG={train_metrics['reg_loss']:.6f} | SOFTCLIP={train_metrics['clip_loss']:.6f} | PCL={train_metrics['pcl_loss']:.6f} | RMSE={train_metrics['rmse']:.6f} | MAE={train_metrics['mae']:.6f} | CI={train_metrics['ci']:.6f} | RM2={train_metrics['rm2']:.6f}")
        print(f"[EPOCH {epoch:03d}/{args.epochs:03d}] VAL  : TOTAL={val_metrics['loss']:.6f} | REG={val_metrics['reg_loss']:.6f} | SOFTCLIP={val_metrics['clip_loss']:.6f} | PCL={val_metrics['pcl_loss']:.6f} | RMSE={val_metrics['rmse']:.6f} | MAE={val_metrics['mae']:.6f} | CI={val_metrics['ci']:.6f} | RM2={val_metrics['rm2']:.6f}")
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        save_checkpoint(Path(args.output_dir) / "latest_model.pt", model, optimizer, epoch, train_metrics, val_metrics, args)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_train_metrics = dict(train_metrics)
            save_checkpoint(Path(args.output_dir) / "best_model.pt", model, optimizer, epoch, train_metrics, val_metrics, args)
            print(f"  SAVED NEW BEST MODEL | EPOCH={best_epoch:03d} | BEST_VAL_RMSE={best_val_metrics['rmse']:.6f} | BEST_VAL_MAE={best_val_metrics['mae']:.6f} | BEST_VAL_CI={best_val_metrics['ci']:.6f} | BEST_VAL_RM2={best_val_metrics['rm2']:.6f}")
        with open(Path(args.output_dir) / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    print("\nTRAINING FINISHED.")
    if best_val_metrics is not None:
        best_summary = {"best_epoch": best_epoch, "best_train_metrics": best_train_metrics, "best_val_metrics": best_val_metrics}
        with open(Path(args.output_dir) / "best_summary.json", "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)
        print(f"BEST MODEL SUMMARY | EPOCH={best_epoch:03d} | VAL_RMSE={best_val_metrics['rmse']:.6f} | VAL_MAE={best_val_metrics['mae']:.6f} | VAL_CI={best_val_metrics['ci']:.6f} | VAL_RM2={best_val_metrics['rm2']:.6f}")
    print(f"SAVED OUTPUTS TO: {args.output_dir}")


if __name__ == "__main__":
    main()
