# -*- coding: utf-8 -*-
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.davis_dataset import DavisDataset
from datasets.collate import mdta_collate_fn, move_batch_to_device
from models.model import MyModelMDTA


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.view(-1).detach().cpu()
    targets = targets.view(-1).detach().cpu()

    mse = torch.mean((preds - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(preds - targets))

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def build_model(args, device):
    model = MyModelMDTA(
        drug_1d_in_dim=768,
        drug_2d_node_dim=43,
        protein_1d_in_dim=1280,
        protein_2d_node_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        task="regression",
    ).to(device)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        pred = model(batch)          # [B, 1]
        target = batch["label"]      # [B, 1]

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * target.size(0)
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs_csv", type=str, default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", type=str, default="data/processed/davis/drug_1d_chemberta2")
    parser.add_argument("--drug_2d_dir", type=str, default="data/processed/davis/drug_2d")
    parser.add_argument("--protein_1d_dir", type=str, default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_2d_dir", type=str, default="data/processed/davis/protein_2d_graph")

    parser.add_argument("--output_dir", type=str, default="outputs/davis_overfit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    full_dataset = DavisDataset(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        drug_2d_dir=args.drug_2d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_2d_dir=args.protein_2d_dir,
        use_protein_3d=False,
    )

    subset_size = min(args.subset_size, len(full_dataset))
    subset_indices = list(range(subset_size))
    dataset = Subset(full_dataset, subset_indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mdta_collate_fn,
        pin_memory=True,
    )

    model = build_model(args, device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Overfit subset size:", len(dataset))

    history = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loader, criterion, optimizer, device)
        eval_metrics = evaluate(model, loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_rmse={train_metrics['rmse']:.6f} "
            f"eval_loss_same_subset={eval_metrics['loss']:.6f} "
            f"eval_rmse_same_subset={eval_metrics['rmse']:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "eval_same_subset": eval_metrics,
        })

        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "metrics": eval_metrics,
                    "subset_indices": subset_indices,
                },
                Path(args.output_dir) / "best_overfit_model.pt"
            )

        with open(Path(args.output_dir) / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("\nFinished overfit check.")
    print(f"Best same-subset loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()