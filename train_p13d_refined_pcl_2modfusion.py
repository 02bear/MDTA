# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d, move_batch_to_device
from models.model_p13d_refined_pcl_2modfusion import MyModelMDTAP13DRefined
from train_p13d import (
    build_fixed_subsets,
    compute_regression_metrics,
    ensure_dir,
    save_checkpoint,
    save_split_indices,
    set_seed,
)


def load_config_defaults(parser: argparse.ArgumentParser):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    if config_args.config is None:
        return

    with open(config_args.config, "r", encoding="utf-8") as f:
        defaults = json.load(f)
    parser.set_defaults(**defaults)


def build_dataloaders(args):
    dataset = DavisDatasetP13D(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.protein_3d_dir,
        drug_2d_dir=None,
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
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mdta_collate_fn_p13d,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, train_set, val_set, train_loader, val_loader


def build_model(args, device):
    return MyModelMDTAP13DRefined(
        drug_1d_in_dim=args.drug_1d_in_dim,
        drug_3d_node_in_dim=args.drug_3d_node_in_dim,
        protein_1d_in_dim=args.protein_1d_in_dim,
        protein_3d_node_s_dim=args.protein_3d_node_s_dim,
        protein_3d_node_v_dim=args.protein_3d_node_v_dim,
        hidden_dim=args.hidden_dim,
        contrastive_dim=args.contrastive_dim,
        dropout=args.dropout,
        temperature=args.temperature,
        use_drug_pcl=args.use_drug_pcl,
        i2moe_gate_temperature=args.i2moe_gate_temperature,
        i2moe_interaction_margin=args.i2moe_interaction_margin,
        i2moe_expert_hidden_mult=args.i2moe_expert_hidden_mult,
        task="regression",
    ).to(device)


def _empty_aux_sums():
    return {
        "pcl_loss": 0.0,
        "protein_pcl_loss": 0.0,
        "drug_pcl_loss": 0.0,
        "i2moe_loss": 0.0,
        "drug_i2moe_loss": 0.0,
        "protein_i2moe_loss": 0.0,
        "mean_drug_gate": 0.0,
        "mean_protein_gate": 0.0,
        "drug_i2moe_w_uni_drug1d": 0.0,
        "drug_i2moe_w_uni_drug3d": 0.0,
        "drug_i2moe_w_synergy": 0.0,
        "drug_i2moe_w_redundancy": 0.0,
        "protein_i2moe_w_uni_prot1d": 0.0,
        "protein_i2moe_w_uni_prot3d": 0.0,
        "protein_i2moe_w_synergy": 0.0,
        "protein_i2moe_w_redundancy": 0.0,
        "pair_cosine": 0.0,
    }


def _accumulate_aux(aux_sums, aux, batch_size):
    for key in aux_sums:
        if key in aux:
            aux_sums[key] += float(aux[key].item()) * batch_size


def train_one_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    running_total_loss = 0.0
    running_reg_loss = 0.0
    aux_sums = _empty_aux_sums()
    all_preds = []
    all_targets = []

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        pred, aux = model(batch, return_aux=True)
        target = batch["label"]

        reg_loss = criterion(pred, target)
        total_loss = (
            reg_loss
            + args.pcl_weight * aux["pcl_loss"]
            + args.i2moe_weight * aux["i2moe_loss"]
        )
        total_loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer.step()

        bs = target.size(0)
        running_total_loss += float(total_loss.item()) * bs
        running_reg_loss += float(reg_loss.item()) * bs
        _accumulate_aux(aux_sums, aux, bs)

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

        if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
            print(
                f"  STEP {step + 1}/{len(loader)} | "
                f"TOTAL={total_loss.item():.6f} | "
                f"REG={reg_loss.item():.6f} | "
                f"PCL={aux['pcl_loss'].item():.6f} | "
                f"I2M={aux['i2moe_loss'].item():.6f} | "
                f"D1={aux['drug_i2moe_w_uni_drug1d'].item():.4f} | "
                f"D3={aux['drug_i2moe_w_uni_drug3d'].item():.4f} | "
                f"P1={aux['protein_i2moe_w_uni_prot1d'].item():.4f} | "
                f"P3={aux['protein_i2moe_w_uni_prot3d'].item():.4f}"
            )

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_regression_metrics(all_preds, all_targets)
    n = len(loader.dataset)
    metrics["loss"] = running_total_loss / n
    metrics["reg_loss"] = running_reg_loss / n
    for key, value in aux_sums.items():
        metrics[key] = value / n
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    model.eval()
    running_total_loss = 0.0
    running_reg_loss = 0.0
    aux_sums = _empty_aux_sums()
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred, aux = model(batch, return_aux=True)
        target = batch["label"]

        reg_loss = criterion(pred, target)
        total_loss = (
            reg_loss
            + args.pcl_weight * aux["pcl_loss"]
            + args.i2moe_weight * aux["i2moe_loss"]
        )

        bs = target.size(0)
        running_total_loss += float(total_loss.item()) * bs
        running_reg_loss += float(reg_loss.item()) * bs
        _accumulate_aux(aux_sums, aux, bs)

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_regression_metrics(all_preds, all_targets)
    n = len(loader.dataset)
    metrics["loss"] = running_total_loss / n
    metrics["reg_loss"] = running_reg_loss / n
    for key, value in aux_sums.items():
        metrics[key] = value / n
    return metrics


def print_epoch_metrics(epoch, total_epochs, split, metrics):
    print(
        f"[EPOCH {epoch:03d}/{total_epochs:03d}] {split:<5}: "
        f"TOTAL={metrics['loss']:.6f} | "
        f"REG={metrics['reg_loss']:.6f} | "
        f"PCL={metrics['pcl_loss']:.6f} | "
        f"I2M={metrics['i2moe_loss']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | "
        f"MAE={metrics['mae']:.6f} | "
        f"CI={metrics['ci']:.6f} | "
        f"RM2={metrics['rm2']:.6f} | "
        f"D1={metrics['drug_i2moe_w_uni_drug1d']:.4f} | "
        f"D3={metrics['drug_i2moe_w_uni_drug3d']:.4f} | "
        f"P1={metrics['protein_i2moe_w_uni_prot1d']:.4f} | "
        f"P3={metrics['protein_i2moe_w_uni_prot3d']:.4f} | "
        f"PAIR_COS={metrics['pair_cosine']:.4f}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--pairs_csv", type=str, default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", type=str, default="data/processed/davis/drug_1d_chemberta2")
    parser.add_argument("--drug_3d_dir", type=str, default="data/processed/davis/drug_3d")
    parser.add_argument("--protein_1d_dir", type=str, default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_3d_dir", type=str, default="data/processed/davis/protein_3d_gvp")
    parser.add_argument("--split_json", type=str, default="data/splits/davis_fixed_split_full.json")

    parser.add_argument("--output_dir", type=str, default="outputs/davis_mdta_refined_pcl_2modfusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--log_interval", type=int, default=100)

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

    parser.add_argument("--drug_1d_in_dim", type=int, default=768)
    parser.add_argument("--drug_3d_node_in_dim", type=int, default=10)
    parser.add_argument("--protein_1d_in_dim", type=int, default=1280)
    parser.add_argument("--protein_3d_node_s_dim", type=int, default=6)
    parser.add_argument("--protein_3d_node_v_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--contrastive_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--pcl_weight", type=float, default=0.05)
    parser.add_argument(
        "--i2moe_weight",
        type=float,
        default=0.01,
        help="Weight for entity-level I2MoE interaction auxiliary loss.",
    )
    parser.add_argument("--i2moe_gate_temperature", type=float, default=1.0)
    parser.add_argument("--i2moe_interaction_margin", type=float, default=0.05)
    parser.add_argument("--i2moe_expert_hidden_mult", type=int, default=2)

    drug_pcl_group = parser.add_mutually_exclusive_group()
    drug_pcl_group.add_argument(
        "--use_drug_pcl",
        dest="use_drug_pcl",
        action="store_true",
        default=True,
        help="Enable drug 1D-3D PCL. Default: True.",
    )
    drug_pcl_group.add_argument(
        "--no_drug_pcl",
        dest="use_drug_pcl",
        action="store_false",
        help="Disable drug 1D-3D PCL and keep protein PCL only.",
    )

    return parser


def main():
    parser = build_parser()
    load_config_defaults(parser)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(
        "MODEL: MDTA-REFINED-2MODFUSION | "
        "PCL -> drug/protein I2MoE feature fusion -> pair interaction regression"
    )
    print(
        f"PCL_WEIGHT={args.pcl_weight} | "
        f"I2MOE_WEIGHT={args.i2moe_weight} | "
        f"USE_DRUG_PCL={args.use_drug_pcl}"
    )

    dataset, train_set, val_set, train_loader, val_loader = build_dataloaders(args)
    print("TOTAL SIZE:", len(dataset))
    print("TRAIN SIZE:", len(train_set))
    print("VAL SIZE:", len(val_set))
    print("SPLIT JSON:", args.split_json)

    save_split_indices(train_set, val_set, args.output_dir)
    with open(Path(args.output_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model = build_model(args, device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    best_train_metrics = None
    best_val_metrics = None
    history = []
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEPOCH {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, args)
        val_metrics = evaluate(model, val_loader, criterion, device, args)

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
            best_train_metrics = dict(train_metrics)
            best_val_metrics = dict(val_metrics)
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
            print(f"  SAVED NEW BEST MODEL | EPOCH={best_epoch:03d} | VAL_RMSE={best_val_rmse:.6f}")
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

    if best_val_metrics is not None:
        best_summary = {
            "best_epoch": best_epoch,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
        }
        with open(Path(args.output_dir) / "best_summary.json", "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)
        print(
            f"\nBEST MODEL SUMMARY | EPOCH={best_epoch:03d} | "
            f"VAL_RMSE={best_val_metrics['rmse']:.6f} | "
            f"VAL_MAE={best_val_metrics['mae']:.6f} | "
            f"VAL_CI={best_val_metrics['ci']:.6f} | "
            f"VAL_RM2={best_val_metrics['rm2']:.6f}"
        )

    print(f"SAVED OUTPUTS TO: {args.output_dir}")


if __name__ == "__main__":
    main()
