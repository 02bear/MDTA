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
from models.model_p13d_finegrained_residual import MyModelMDTAP13DFineGrained


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path):
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

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    return train_set, val_set


# =========================
# DTA-style metrics
# =========================

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

    ci = get_cindex(targets_np, preds_np)
    rm2 = get_rm2(targets_np, preds_np)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "ci": float(ci),
        "rm2": float(rm2),
    }


def build_dataloaders(args):
    dataset = DavisDatasetP13D(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.protein_3d_dir,
        drug_2d_dir=args.drug_2d_dir,
        use_drug_2d=False,
        drug_3d_dir=args.drug_3d_dir,
        use_drug_3d=True,
    )

    train_set, val_set = build_fixed_subsets(dataset, args.split_json)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mdta_collate_fn_p13d,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mdta_collate_fn_p13d,
        pin_memory=True,
    )

    return dataset, train_set, val_set, train_loader, val_loader


def build_model(args, device):
    model = MyModelMDTAP13DFineGrained(
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
    ).to(device)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, log_interval=200):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        pred = model(batch)
        target = batch["label"]

        loss = criterion(pred, target)
        loss.backward()
        
        # ============================================================
        # DEBUG: 仅检查第一个 batch 的梯度，判断细粒度模块是否真正可训练
        # ============================================================
        if step == 0:
            print("\n========== GRADIENT CHECK ==========")
        
            def print_module_grad(module_name, module):
                total_sq_norm = 0.0
                has_grad = False
        
                print(f"\n[{module_name}]")
        
                for name, param in module.named_parameters():
                    if not param.requires_grad:
                        print(f"  {name}: requires_grad=False")
                        continue
        
                    if param.grad is None:
                        print(f"  {name}: grad=None")
                    else:
                        grad_norm = param.grad.detach().norm().item()
                        print(f"  {name}: grad_norm={grad_norm:.10f}")
                        total_sq_norm += grad_norm ** 2
                        has_grad = True
        
                total_norm = total_sq_norm ** 0.5
        
                print(
                    f"  SUMMARY | has_grad={has_grad} | "
                    f"total_grad_norm={total_norm:.10f}"
                )
        
            # 1. 最关键：负责选择候选残基的模块
            print_module_grad(
                "pocket_selector",
                model.pocket_selector,
            )
        
            # 2. 原子—残基交互模块，作为对照
            print_module_grad(
                "atom_residue_interaction",
                model.atom_residue_interaction,
            )
        
            # 3. 最终预测头，作为对照
            print_module_grad(
                "decoder",
                model.decoder,
            )
        
            print("======== END GRADIENT CHECK ========\n")
        
        optimizer.step()

        running_loss += float(loss.item()) * target.size(0)

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

        if (step + 1) % log_interval == 0:
            batch_rmse = torch.sqrt(loss.detach())
            print(
                f"  STEP {step + 1}/{len(loader)} | "
                f"BATCH_LOSS={loss.item():.6f} | "
                f"BATCH_RMSE={batch_rmse.item():.6f}"
            )

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_regression_metrics(all_preds, all_targets)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(batch)
        target = batch["label"]

        loss = criterion(pred, target)
        running_loss += float(loss.item()) * target.size(0)

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_regression_metrics(all_preds, all_targets)
    metrics["loss"] = avg_loss
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


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--pairs_csv", type=str, default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", type=str, default="data/processed/davis/drug_1d_chemberta2")
    parser.add_argument("--drug_2d_dir", type=str, default="data/processed/davis/drug_2d")
    parser.add_argument("--drug_3d_dir", type=str, default="data/processed/davis/drug_3d")
    parser.add_argument("--protein_1d_dir", type=str, default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_3d_dir", type=str, default="data/processed/davis/protein_3d_gvp")
    parser.add_argument("--split_json", type=str, default="data/splits/davis_fixed_split_size2.json")

    # train
    parser.add_argument("--output_dir", type=str, default="outputs/davis_p13d_fixedsplit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)  # 保留兼容，不再实际使�?
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # early stopping
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=60,
        help="Stop training if val RMSE does not improve for this many epochs. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="Minimum val RMSE decrease required to reset early stopping patience.",
    )

    # model
    parser.add_argument("--drug_1d_in_dim", type=int, default=768)
    parser.add_argument("--drug_3d_node_in_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pocket_top_k",
        type=int,
        default=64,
        help="Number of drug-conditioned candidate protein residues selected for local atom-residue interaction.",
    )
    parser.add_argument(
        "--interaction_heads",
        type=int,
        default=4,
        help="Number of attention heads used in the atom-residue interaction module.",
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset, train_set, val_set, train_loader, val_loader = build_dataloaders(args)
    print("TOTAL SIZE:", len(dataset))
    print("TRAIN SIZE:", len(train_set))
    print("VAL SIZE:", len(val_set))
    print("SPLIT JSON:", args.split_json)
    print("MODEL: drug_1d + drug_3d + protein_1d + protein_3d + fine-grained atom-residue local interaction")
    print(f"POCKET_TOP_K: {args.pocket_top_k}")
    print(f"INTERACTION_HEADS: {args.interaction_heads}")

    save_split_indices(train_set, val_set, args.output_dir)

    model = build_model(args, device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    best_val_metrics = None
    best_train_metrics = None
    history = []
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\\nEPOCH {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_interval=200,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"[EPOCH {epoch:03d}/{args.epochs:03d}] "
            f"TRAIN: LOSS={train_metrics['loss']:.6f} | "
            f"MSE={train_metrics['mse']:.6f} | "
            f"RMSE={train_metrics['rmse']:.6f} | "
            f"MAE={train_metrics['mae']:.6f} | "
            f"CI={train_metrics['ci']:.6f} | "
            f"RM2={train_metrics['rm2']:.6f}"
        )
        print(
            f"[EPOCH {epoch:03d}/{args.epochs:03d}] "
            f"VAL  : LOSS={val_metrics['loss']:.6f} | "
            f"MSE={val_metrics['mse']:.6f} | "
            f"RMSE={val_metrics['rmse']:.6f} | "
            f"MAE={val_metrics['mae']:.6f} | "
            f"CI={val_metrics['ci']:.6f} | "
            f"RM2={val_metrics['rm2']:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        save_checkpoint(
            path=Path(args.output_dir) / "latest_model.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
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
                path=Path(args.output_dir) / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
            )
            print(
                f"  SAVED NEW BEST MODEL | "
                f"EPOCH={best_epoch:03d} | "
                f"BEST_VAL_MSE={best_val_metrics['mse']:.6f} | "
                f"BEST_VAL_RMSE={best_val_metrics['rmse']:.6f} | "
                f"BEST_VAL_MAE={best_val_metrics['mae']:.6f} | "
                f"BEST_VAL_CI={best_val_metrics['ci']:.6f} | "
                f"BEST_VAL_RM2={best_val_metrics['rm2']:.6f}"
            )
        else:
            epochs_no_improve += 1
            print(
                f"  EARLY_STOP COUNTER | "
                f"NO_IMPROVE={epochs_no_improve}/{args.early_stop_patience} | "
                f"BEST_EPOCH={best_epoch:03d} | "
                f"BEST_VAL_RMSE={best_val_rmse:.6f} | "
                f"CURRENT_VAL_RMSE={current_val_rmse:.6f}"
            )

        with open(Path(args.output_dir) / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"\nEARLY STOPPING TRIGGERED | "
                f"BEST_EPOCH={best_epoch:03d} | "
                f"BEST_VAL_RMSE={best_val_rmse:.6f} | "
                f"PATIENCE={args.early_stop_patience} | "
                f"MIN_DELTA={args.early_stop_min_delta}"
            )
            break

    print("\\nTRAINING FINISHED.")

    if best_val_metrics is not None:
        print(
            f"BEST MODEL SUMMARY | "
            f"EPOCH={best_epoch:03d} | "
            f"VAL_LOSS={best_val_metrics['loss']:.6f} | "
            f"VAL_MSE={best_val_metrics['mse']:.6f} | "
            f"VAL_RMSE={best_val_metrics['rmse']:.6f} | "
            f"VAL_MAE={best_val_metrics['mae']:.6f} | "
            f"VAL_CI={best_val_metrics['ci']:.6f} | "
            f"VAL_RM2={best_val_metrics['rm2']:.6f}"
        )

        print(
            f"BEST MODEL TRAIN METRICS | "
            f"EPOCH={best_epoch:03d} | "
            f"TRAIN_LOSS={best_train_metrics['loss']:.6f} | "
            f"TRAIN_MSE={best_train_metrics['mse']:.6f} | "
            f"TRAIN_RMSE={best_train_metrics['rmse']:.6f} | "
            f"TRAIN_MAE={best_train_metrics['mae']:.6f} | "
            f"TRAIN_CI={best_train_metrics['ci']:.6f} | "
            f"TRAIN_RM2={best_train_metrics['rm2']:.6f}"
        )

        best_summary = {
            "best_epoch": best_epoch,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
        }
        with open(Path(args.output_dir) / "best_summary.json", "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

    print(f"SAVED OUTPUTS TO: {args.output_dir}")


if __name__ == "__main__":
    main()