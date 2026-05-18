# -*- coding: utf-8 -*-
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from datasets.davis_dataset import DavisDataset
from datasets.collate import mdta_collate_fn, move_batch_to_device
from models.model_absolution import MyModelMDTAAblation


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


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


def apply_modality_mask(batch, args):
    if not bool(args.use_drug_1d):
        batch["drug_1d"] = None
    if not bool(args.use_drug_2d):
        batch["drug_2d"] = None
    if not bool(args.use_protein_1d):
        batch["protein_1d"] = None
    if not bool(args.use_protein_2d):
        batch["protein_2d"] = None
    if not bool(args.use_protein_3d):
        batch["protein_3d"] = None
    return batch


def build_dataloaders(args):
    dataset = DavisDataset(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        drug_2d_dir=args.drug_2d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_2d_dir=args.protein_2d_dir,
        use_protein_3d=False,
    )

    total_len = len(dataset)
    train_len = int(total_len * args.train_ratio)
    val_len = total_len - train_len

    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mdta_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mdta_collate_fn,
        pin_memory=True,
    )

    return dataset, train_set, val_set, train_loader, val_loader


def build_model(args, device):
    model = MyModelMDTAAblation(
        drug_1d_in_dim=768,
        drug_2d_node_dim=43,
        protein_1d_in_dim=1280,
        protein_2d_node_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        task="regression",
        use_drug_1d=bool(args.use_drug_1d),
        use_drug_2d=bool(args.use_drug_2d),
        use_protein_1d=bool(args.use_protein_1d),
        use_protein_2d=bool(args.use_protein_2d),
        use_protein_3d=bool(args.use_protein_3d),
    ).to(device)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, args, log_interval=200):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for step, batch in enumerate(loader):
        batch = apply_modality_mask(batch, args)
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        pred = model(batch)
        target = batch["label"]

        loss = criterion(pred, target)
        loss.backward()
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
def evaluate(model, loader, criterion, device, args):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = apply_modality_mask(batch, args)
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
        json.dump(split_info, f)


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
    parser.add_argument("--protein_1d_dir", type=str, default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_2d_dir", type=str, default="data/processed/davis/protein_2d_graph")

    # train
    parser.add_argument("--output_dir", type=str, default="outputs/davis_absolution")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # model
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # modality switches
    parser.add_argument("--use_drug_1d", type=int, default=1)
    parser.add_argument("--use_drug_2d", type=int, default=1)
    parser.add_argument("--use_protein_1d", type=int, default=1)
    parser.add_argument("--use_protein_2d", type=int, default=1)
    parser.add_argument("--use_protein_3d", type=int, default=0)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset, train_set, val_set, train_loader, val_loader = build_dataloaders(args)
    print("TOTAL SIZE:", len(dataset))
    print("TRAIN SIZE:", len(train_set))
    print("VAL SIZE:", len(val_set))
    print(
        "MODALITIES | "
        f"DRUG_1D={args.use_drug_1d} | "
        f"DRUG_2D={args.use_drug_2d} | "
        f"PROTEIN_1D={args.use_protein_1d} | "
        f"PROTEIN_2D={args.use_protein_2d} | "
        f"PROTEIN_3D={args.use_protein_3d}"
    )

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

    for epoch in range(1, args.epochs + 1):
        print(f"\nEPOCH {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
            log_interval=200,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
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

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_train_metrics = dict(train_metrics)

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

        with open(Path(args.output_dir) / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("\nTRAINING FINISHED.")

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