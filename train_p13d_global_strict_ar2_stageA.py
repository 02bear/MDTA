# -*- coding: utf-8 -*-
"""
Train Clean Global + Strict AR2 Stage A.

The independent Global checkpoint is frozen. Only the all-residue Strict AR2
branch and its bias-free residual head are trained.

Built-in controls:
1) strict Global checkpoint load
2) custom node-exposing Global path must numerically equal original Global path
3) INIT final == Global exactly
4) automatic Global bias-only / affine calibration control (fit TRAIN only)
5) mechanism watchdog
6) separate best-MSE and best-mechanism-valid checkpoints
7) automatic latent causal audit on best mechanism-valid checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d
from models.model_p13d_global_strict_ar2 import GlobalStrictAR2StageA


def set_seed(seed: int):
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


def _id_string(x: Any) -> str:
    if torch.is_tensor(x):
        if x.numel() == 1:
            return str(x.item())
        return str(x.detach().cpu().tolist())
    return str(x)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

class FenwickTree:
    def __init__(self, n: int):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def update(self, i: int):
        while i <= self.n:
            self.tree[i] += 1
            i += i & -i

    def query(self, i: int):
        s = 0
        while i > 0:
            s += int(self.tree[i])
            i -= i & -i
        return s


def cindex(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    if len(y) <= 1:
        return 0.0

    unique_p = np.unique(p)
    ranks = {v: i + 1 for i, v in enumerate(unique_p)}
    order = np.argsort(y, kind="mergesort")
    y = y[order]
    p = p[order]

    bit = FenwickTree(len(ranks))
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
            less = bit.query(rank - 1)
            equal = bit.query(rank) - less
            concordant += less + 0.5 * equal
            comparable += previous

        for k in range(start, end):
            bit.update(ranks[p[k]])
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
    e = p - y
    mse = float(np.mean(e ** 2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(e))),
        "ci": cindex(y, p),
        "rm2": rm2(y, p),
    }


def fmt(prefix, m):
    return (
        f"{prefix} MSE={m['mse']:.6f} | RMSE={m['rmse']:.6f} | "
        f"MAE={m['mae']:.6f} | CI={m['ci']:.6f} | RM2={m['rm2']:.6f}"
    )


# ---------------------------------------------------------------------
# Heavy-atom mapping
# ---------------------------------------------------------------------

class GlobalWithHeavyAtomMapping(Dataset):
    def __init__(self, base_dataset: Dataset, functional_group_dir: str):
        self.base_dataset = base_dataset
        root = Path(functional_group_dir)
        if not root.is_dir():
            raise NotADirectoryError(root)

        self.heavy_by_drug = {}
        self.num_atoms_by_drug = {}

        files = sorted(root.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files under {root}")

        for path in files:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if obj.get("version") != "drug_functional_groups_v2":
                raise ValueError(
                    f"{path}: expected drug_functional_groups_v2, "
                    f"got {obj.get('version')!r}"
                )
            did = str(obj.get("drug_id", path.stem))
            heavy = torch.as_tensor(
                obj["heavy_atom_indices"],
                dtype=torch.long,
            ).view(-1)
            if heavy.numel() == 0:
                raise ValueError(f"{path}: no heavy atoms")
            self.heavy_by_drug[did] = heavy
            self.num_atoms_by_drug[did] = int(obj["num_atoms_e2"])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        did = _id_string(item["drug_id"])
        if did not in self.heavy_by_drug:
            raise KeyError(f"Missing heavy-atom cache for drug_id={did}")
        return {
            "base": item,
            "heavy_local": self.heavy_by_drug[did],
            "expected_num_atoms": self.num_atoms_by_drug[did],
        }


def heavy_atom_collate(samples: List[Dict[str, Any]]):
    base_items = [x["base"] for x in samples]
    out = mdta_collate_fn_p13d(base_items)

    batch_size = len(samples)
    atom_batch = out["drug_3d"]["batch"].long()
    atom_counts = torch.bincount(atom_batch, minlength=batch_size)
    atom_offsets = torch.cumsum(atom_counts, 0) - atom_counts

    max_h = max(int(x["heavy_local"].numel()) for x in samples)
    heavy_global = torch.full(
        (batch_size, max_h),
        -1,
        dtype=torch.long,
    )
    heavy_mask = torch.zeros(
        (batch_size, max_h),
        dtype=torch.bool,
    )

    for b, x in enumerate(samples):
        local = x["heavy_local"].long().view(-1)
        n_atom = int(atom_counts[b].item())
        expected = int(x["expected_num_atoms"])

        if n_atom != expected:
            raise ValueError(
                f"drug_id={out['drug_id'][b]}: graph atoms={n_atom}, "
                f"cache atoms={expected}"
            )
        if int(local.min()) < 0 or int(local.max()) >= n_atom:
            raise IndexError(
                f"drug_id={out['drug_id'][b]}: invalid heavy atom index"
            )

        n = local.numel()
        heavy_global[b, :n] = local + atom_offsets[b]
        heavy_mask[b, :n] = True

    out["heavy_atom_indices"] = heavy_global
    out["heavy_atom_mask"] = heavy_mask
    return out


# ---------------------------------------------------------------------
# Dataset / model
# ---------------------------------------------------------------------

def build_dataset(args):
    base = DavisDatasetP13D(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.protein_3d_dir,
        drug_2d_dir=args.drug_2d_dir,
        use_drug_2d=False,
        drug_3d_dir=args.drug_3d_dir,
        use_drug_3d=True,
    )
    return GlobalWithHeavyAtomMapping(
        base,
        args.functional_group_dir,
    )


def build_loaders(args, dataset, split):
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=heavy_atom_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    train_loader = DataLoader(
        Subset(dataset, split["train_indices"]),
        shuffle=True,
        **common,
    )
    train_eval_loader = DataLoader(
        Subset(dataset, split["train_indices"]),
        shuffle=False,
        **common,
    )
    val_loader = DataLoader(
        Subset(dataset, split["val_indices"]),
        shuffle=False,
        **common,
    )
    return train_loader, train_eval_loader, val_loader


def ckpt_args_dict(ckpt):
    x = ckpt.get("args", {})
    return x if isinstance(x, dict) else vars(x)


def build_model(args, ckpt, device):
    a = ckpt_args_dict(ckpt)
    model = GlobalStrictAR2StageA(
        drug_1d_in_dim=int(a.get("drug_1d_in_dim", 768)),
        drug_3d_node_in_dim=int(a.get("drug_3d_node_in_dim", 10)),
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=int(a.get("hidden_dim", 128)),
        dropout=float(a.get("dropout", 0.1)),
        interaction_dim=args.interaction_dim,
        task="regression",
        freeze_global=True,
    ).to(device)

    model.load_global_checkpoint_state(
        ckpt["model_state_dict"]
    )
    model.set_global_frozen(True)
    return model


# ---------------------------------------------------------------------
# Eval / calibration
# ---------------------------------------------------------------------

@torch.no_grad()
def collect_global_predictions(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        p = model.global_model(batch).detach().view(-1).cpu()
        y = batch["label"].detach().view(-1).cpu()
        ys.append(y)
        ps.append(p)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def fit_calibration(y_train, p_train, y_val, p_val):
    b0 = float(np.mean(y_train - p_train))

    X = np.column_stack([p_train, np.ones_like(p_train)])
    coef, *_ = np.linalg.lstsq(X, y_train, rcond=None)
    a = float(coef[0])
    b = float(coef[1])

    return {
        "bias_only_b": b0,
        "affine_a": a,
        "affine_b": b,
        "train": {
            "global": metrics(y_train, p_train),
            "bias_only": metrics(y_train, p_train + b0),
            "affine": metrics(y_train, a * p_train + b),
        },
        "val": {
            "global": metrics(y_val, p_val),
            "bias_only": metrics(y_val, p_val + b0),
            "affine": metrics(y_val, a * p_val + b),
        },
    }


def mechanism_valid(result, args):
    return (
        result["mean_interaction_score_std"] >= args.pair_std_floor
        and result["mean_tv_from_uniform"] >= args.tv_floor
        and result["mean_z_norm"] >= args.z_norm_floor
        and result["delta_std"] >= args.delta_std_floor
    )


def run_epoch(model, loader, device, optimizer, args):
    training = optimizer is not None
    model.train(training)

    final_ps, global_ps, ys, deltas = [], [], [], []
    n_total = 0
    loss_sum = 0.0

    diag_sum = {
        "raw_std": 0.0,
        "int_std": 0.0,
        "ent": 0.0,
        "tv": 0.0,
        "znorm": 0.0,
        "atoms": 0.0,
        "residues": 0.0,
        "pairs": 0.0,
        "abs_delta": 0.0,
    }
    center_err_max = 0.0

    trainable = [
        p for name, p in model.named_parameters()
        if not name.startswith("global_model.") and p.requires_grad
    ]

    for step, batch in enumerate(loader, 1):
        batch = move_to_device(batch, device)
        y = batch["label"].float()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            d = model(batch, return_details=True)
            p = d["pred"].float()
            loss = F.mse_loss(p, y)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at step={step}: {loss.item()}"
                )

            if training:
                loss.backward()

                if step == 1:
                    bad = [
                        name for name, par in model.global_model.named_parameters()
                        if par.grad is not None
                    ]
                    if bad:
                        raise RuntimeError(
                            "Frozen Global received gradients: "
                            + ", ".join(bad[:10])
                        )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    args.grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            else:
                grad_norm = torch.zeros((), device=device)

        n = y.size(0)
        n_total += n
        loss_sum += float(loss.detach()) * n

        diag_sum["raw_std"] += float(
            d["ar2_raw_score_std"].detach().mean()
        ) * n
        diag_sum["int_std"] += float(
            d["ar2_interaction_score_std"].detach().mean()
        ) * n
        diag_sum["ent"] += float(
            d["ar2_normalized_entropy"].detach().mean()
        ) * n
        diag_sum["tv"] += float(
            d["ar2_tv_from_uniform"].detach().mean()
        ) * n
        diag_sum["znorm"] += float(
            d["ar2_z_norm"].detach().mean()
        ) * n
        diag_sum["atoms"] += float(
            d["ar2_num_atoms"].detach().mean()
        ) * n
        diag_sum["residues"] += float(
            d["ar2_num_residues"].detach().mean()
        ) * n
        diag_sum["pairs"] += float(
            d["ar2_num_pairs"].detach().mean()
        ) * n
        diag_sum["abs_delta"] += float(
            d["ar2_delta"].detach().abs().mean()
        ) * n
        center_err_max = max(
            center_err_max,
            float(d["ar2_center_err"].detach().max()),
        )

        final_ps.append(d["pred"].detach().view(-1).cpu())
        global_ps.append(d["global_pred"].detach().view(-1).cpu())
        deltas.append(d["ar2_delta"].detach().view(-1).cpu())
        ys.append(y.detach().view(-1).cpu())

        if training and args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"  STEP {step:05d}/{len(loader):05d} "
                f"| LOSS={loss.item():.6f} "
                f"| |DELTA|={d['ar2_delta'].detach().abs().mean().item():.5f} "
                f"| INT_STD={d['ar2_interaction_score_std'].detach().mean().item():.5f} "
                f"| ENT={d['ar2_normalized_entropy'].detach().mean().item():.5f} "
                f"| TV={d['ar2_tv_from_uniform'].detach().mean().item():.5f} "
                f"| Z={d['ar2_z_norm'].detach().mean().item():.5f} "
                f"| GRAD={float(grad_norm):.4f}",
                flush=True,
            )

        if training and args.max_train_batches > 0 and step >= args.max_train_batches:
            break
        if (not training) and args.max_eval_batches > 0 and step >= args.max_eval_batches:
            break

    y_np = torch.cat(ys).numpy()
    p_np = torch.cat(final_ps).numpy()
    g_np = torch.cat(global_ps).numpy()
    delta_np = torch.cat(deltas).numpy()

    result = {
        "final": metrics(y_np, p_np),
        "global": metrics(y_np, g_np),
        "loss": loss_sum / max(n_total, 1),
        "mean_abs_delta": diag_sum["abs_delta"] / max(n_total, 1),
        "delta_std": float(np.std(delta_np)),
        "mean_raw_score_std": diag_sum["raw_std"] / max(n_total, 1),
        "mean_interaction_score_std": diag_sum["int_std"] / max(n_total, 1),
        "mean_normalized_entropy": diag_sum["ent"] / max(n_total, 1),
        "mean_tv_from_uniform": diag_sum["tv"] / max(n_total, 1),
        "mean_z_norm": diag_sum["znorm"] / max(n_total, 1),
        "mean_heavy_atoms": diag_sum["atoms"] / max(n_total, 1),
        "mean_residues": diag_sum["residues"] / max(n_total, 1),
        "mean_pair_count": diag_sum["pairs"] / max(n_total, 1),
        "center_err_max": center_err_max,
    }
    result["mechanism_valid"] = mechanism_valid(result, args)
    return result, y_np, p_np


def print_epoch(prefix, r):
    print(fmt(f"{prefix} FINAL ", r["final"]), flush=True)
    print(fmt(f"{prefix} GLOBAL", r["global"]), flush=True)
    print(
        f"{prefix} DIAG | |DELTA|={r['mean_abs_delta']:.6f} "
        f"| DELTA_STD={r['delta_std']:.6f} "
        f"| RAW_STD={r['mean_raw_score_std']:.6f} "
        f"| INT_STD={r['mean_interaction_score_std']:.6f} "
        f"| ENT={r['mean_normalized_entropy']:.6f} "
        f"| TV={r['mean_tv_from_uniform']:.6f} "
        f"| Z={r['mean_z_norm']:.6f} "
        f"| CENTER_ERR={r['center_err_max']:.2e} "
        f"| A={r['mean_heavy_atoms']:.1f} "
        f"| R={r['mean_residues']:.1f} "
        f"| PAIRS={r['mean_pair_count']:.0f} "
        f"| MECH={'PASS' if r['mechanism_valid'] else 'FAIL'}",
        flush=True,
    )


def save_ckpt(path, model, optimizer, epoch, train_r, val_r, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_result": train_r,
            "val_result": val_r,
            "args": vars(args),
            "model_version": "CLEAN_GLOBAL_STRICT_AR2_ALL_RESIDUE",
        },
        path,
    )


# ---------------------------------------------------------------------
# Causal latent audit
# ---------------------------------------------------------------------

@torch.no_grad()
def collect_latents(model, loader, device):
    model.eval()
    y_all, g_all, z_all, drug_ids = [], [], [], []

    for batch_cpu in loader:
        drug_ids.extend([_id_string(x) for x in batch_cpu["drug_id"]])
        batch = move_to_device(batch_cpu, device)
        d = model(batch, return_details=True)

        y_all.append(batch["label"].detach().view(-1).cpu())
        g_all.append(d["global_pred"].detach().view(-1).cpu())
        z_all.append(d["ar2_feat"].detach().cpu())

    return (
        torch.cat(y_all),
        torch.cat(g_all),
        torch.cat(z_all),
        drug_ids,
    )


@torch.no_grad()
def head_on_latents(model, z_cpu, device, bs=512):
    out = []
    for s in range(0, z_cpu.size(0), bs):
        z = z_cpu[s:s+bs].to(device)
        out.append(model.ar2_delta_head(z).view(-1).cpu())
    return torch.cat(out)


def causal_audit(model, loader, device, seed):
    y, g, z, drug_ids = collect_latents(model, loader, device)

    d0 = head_on_latents(model, z, device)
    pred_orig = g + d0

    pred_zero = g + head_on_latents(
        model, torch.zeros_like(z), device
    )

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(z.size(0), generator=gen)
    pred_perm = g + head_on_latents(model, z[perm], device)

    z_drugmean = torch.empty_like(z)
    for did in sorted(set(drug_ids)):
        idx = [i for i, x in enumerate(drug_ids) if x == did]
        ii = torch.as_tensor(idx, dtype=torch.long)
        z_drugmean[ii] = z[ii].mean(0, keepdim=True)

    pred_drugmean = g + head_on_latents(
        model, z_drugmean, device
    )

    yn = y.numpy()
    po = pred_orig.numpy()
    return {
        "original": metrics(yn, po),
        "zero_z": metrics(yn, pred_zero.numpy()),
        "same_drug_mean_z": metrics(yn, pred_drugmean.numpy()),
        "global_permuted_z": metrics(yn, pred_perm.numpy()),
        "mean_abs_prediction_change": {
            "zero_z": float(np.mean(np.abs(pred_zero.numpy() - po))),
            "same_drug_mean_z": float(
                np.mean(np.abs(pred_drugmean.numpy() - po))
            ),
            "global_permuted_z": float(
                np.mean(np.abs(pred_perm.numpy() - po))
            ),
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser()

    p.add_argument("--pairs_csv", default="data/raw/davis/pairs.csv")
    p.add_argument("--drug_1d_dir", default="data/processed/davis/drug_1d_chemberta2")
    p.add_argument("--drug_2d_dir", default="data/processed/davis/drug_2d")
    p.add_argument("--drug_3d_dir", default="data/processed/davis/drug_3d")
    p.add_argument("--protein_1d_dir", default="data/processed/davis/protein_1d_esm2")
    p.add_argument("--protein_3d_dir", default="data/processed/davis/protein_3d_gvp")
    p.add_argument("--functional_group_dir", default="data/processed/davis/drug_functional_groups")

    p.add_argument("--global_checkpoint", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--ar_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--early_stop_patience", type=int, default=25)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--interaction_dim", type=int, default=128)

    p.add_argument("--mechanism_warmup", type=int, default=3)
    p.add_argument("--mechanism_patience", type=int, default=3)
    p.add_argument("--pair_std_floor", type=float, default=1e-4)
    p.add_argument("--tv_floor", type=float, default=1e-4)
    p.add_argument("--z_norm_floor", type=float, default=1e-5)
    p.add_argument("--delta_std_floor", type=float, default=1e-4)

    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_eval_batches", type=int, default=0)
    p.add_argument("--smoke_only", action="store_true")
    p.add_argument("--allow_overwrite", action="store_true")
    return p


def main():
    args = parser().parse_args()
    set_seed(args.seed)

    out = Path(args.output_dir)
    if out.exists() and not args.allow_overwrite:
        meaningful = [
            x for x in out.iterdir()
            if x.name not in {"train.log", "pid.txt", "launcher.log"}
        ]
        if meaningful:
            raise FileExistsError(
                f"Output directory already contains artifacts: {out}"
            )
    out.mkdir(parents=True, exist_ok=True)

    split = json.loads(
        Path(args.split_json).read_text(encoding="utf-8")
    )

    ckpt = torch.load(
        args.global_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in ckpt:
        raise KeyError("Global checkpoint has no model_state_dict")

    dataset = build_dataset(args)
    train_loader, train_eval_loader, val_loader = build_loaders(
        args, dataset, split
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = build_model(args, ckpt, device)

    global_trainable = sum(
        p.numel() for p in model.global_model.parameters()
        if p.requires_grad
    )
    ar_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("global_model.") and p.requires_grad
    ]

    print(f"GLOBAL CHECKPOINT: {args.global_checkpoint}", flush=True)
    print(f"GLOBAL EPOCH: {ckpt.get('epoch', 'N/A')}", flush=True)
    print(
        f"DEVICE={device} | TRAIN={len(train_loader.dataset)} "
        f"| VAL={len(val_loader.dataset)}",
        flush=True,
    )
    print(
        f"TRAINABLE | GLOBAL={global_trainable} (FROZEN) "
        f"| AR2={sum(p.numel() for p in ar_params):,}",
        flush=True,
    )
    print(
        "MODEL=CLEAN_GLOBAL_STRICT_AR2_ALL_RESIDUE "
        "| NO_E2 | NO_SELECTOR | HEAVY_ATOMS_X_ALL_RESIDUES "
        "| DOUBLE_CENTER | CENTERED_UNIFORM_NULL | BIAS_FREE_DELTA",
        flush=True,
    )

    if global_trainable != 0:
        raise RuntimeError("Global must be fully frozen.")

    # H(0) must be exactly zero.
    z0 = model.zero_input_delta(4, device)
    print(
        f"ZERO PROPERTY | max_abs(H(0))={z0.abs().max().item():.12g}",
        flush=True,
    )
    if float(z0.abs().max()) != 0.0:
        raise RuntimeError("Zero-preserving residual property failed.")

    # Verify custom node-exposing forward reproduces original Global exactly.
    first_cpu = next(iter(val_loader))
    first = move_to_device(first_cpu, device)
    model.eval()
    with torch.no_grad():
        original_global = model.global_model(first)
        details = model(first, return_details=True)

    forward_diff = float(
        (original_global - details["global_pred"]).abs().max()
    )
    init_delta_diff = float(
        (details["pred"] - details["global_pred"]).abs().max()
    )

    print(
        f"FORWARD SANITY | max_abs(original_global-nodepath_global)="
        f"{forward_diff:.12g}",
        flush=True,
    )
    print(
        f"INIT SANITY | max_abs(final-global)={init_delta_diff:.12g}",
        flush=True,
    )

    if forward_diff > 1e-6:
        raise RuntimeError(
            "Node-exposing Global path does not reproduce original Global."
        )
    if init_delta_diff > 1e-7:
        raise RuntimeError("INIT final != Global.")

    # Full calibration control unless smoke is explicitly truncated.
    if args.smoke_only:
        print("SMOKE MODE: skip full affine calibration.", flush=True)
        calibration = None
    else:
        ytr, ptr = collect_global_predictions(
            model, train_eval_loader, device
        )
        yva, pva = collect_global_predictions(
            model, val_loader, device
        )
        calibration = fit_calibration(
            ytr, ptr, yva, pva
        )

        print("\nGLOBAL CALIBRATION CONTROL", flush=True)
        print(
            f"bias-only b={calibration['bias_only_b']:.8f}",
            flush=True,
        )
        print(
            f"affine a={calibration['affine_a']:.8f} "
            f"b={calibration['affine_b']:.8f}",
            flush=True,
        )
        for key in ("global", "bias_only", "affine"):
            print(
                fmt(f"VAL {key.upper():9s}", calibration["val"][key]),
                flush=True,
            )
        (out / "global_calibration.json").write_text(
            json.dumps(calibration, indent=2),
            encoding="utf-8",
        )

    init_val, _, _ = run_epoch(
        model, val_loader, device, None, args
    )
    print_epoch("INIT-VAL", init_val)

    saved = ckpt.get("val_metrics", {})
    if (not args.smoke_only) and isinstance(saved, dict) and "mse" in saved:
        repro_diff = abs(
            init_val["global"]["mse"] - float(saved["mse"])
        )
        print(
            f"GLOBAL REPRO | saved={float(saved['mse']):.6f} "
            f"| current={init_val['global']['mse']:.6f} "
            f"| abs_diff={repro_diff:.6g}",
            flush=True,
        )
        if repro_diff > 1e-3:
            raise RuntimeError(
                "Global checkpoint reproduction differs by >1e-3."
            )

    if init_val["center_err_max"] > 1e-5:
        raise RuntimeError(
            f"Double-centering invariant failed: "
            f"{init_val['center_err_max']}"
        )

    optimizer = torch.optim.Adam(
        ar_params,
        lr=args.ar_lr,
        weight_decay=args.weight_decay,
    )

    if args.smoke_only:
        old = args.max_train_batches
        args.max_train_batches = 2
        smoke, _, _ = run_epoch(
            model, train_loader, device, optimizer, args
        )
        print_epoch("SMOKE-TRAIN", smoke)
        args.max_train_batches = old
        print("SMOKE_ONLY COMPLETE", flush=True)
        return

    history = []
    best_rmse = float("inf")
    best_epoch = -1
    best_val = None
    best_mech_rmse = float("inf")
    best_mech_epoch = -1
    best_mech_val = None
    stale = 0
    mech_stale = 0
    stop_reason = "max_epochs"

    for epoch in range(1, args.epochs + 1):
        print(
            f"\nEPOCH {epoch}/{args.epochs} "
            f"| GLOBAL=FROZEN | AR_LR={args.ar_lr:g}",
            flush=True,
        )

        tr, _, _ = run_epoch(
            model, train_loader, device, optimizer, args
        )
        va, yv, pv = run_epoch(
            model, val_loader, device, None, args
        )

        print_epoch("TRAIN", tr)
        print_epoch("VAL  ", va)

        history.append({"epoch": epoch, "train": tr, "val": va})
        (out / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        save_ckpt(
            out / "latest_model.pt",
            model, optimizer, epoch, tr, va, args
        )

        rmse_now = va["final"]["rmse"]
        if rmse_now < best_rmse - args.early_stop_min_delta:
            best_rmse = rmse_now
            best_epoch = epoch
            best_val = va
            stale = 0

            save_ckpt(
                out / "best_model.pt",
                model, optimizer, epoch, tr, va, args
            )
            np.savez(
                out / "best_val_predictions.npz",
                target=yv,
                prediction=pv,
            )

            print(
                f"SAVED BEST-MSE | EPOCH={epoch} "
                f"| FINAL_MSE={va['final']['mse']:.6f} "
                f"| GLOBAL_MSE={va['global']['mse']:.6f} "
                f"| DELTA={va['final']['mse']-va['global']['mse']:+.6f}",
                flush=True,
            )
        else:
            stale += 1

        if (
            va["mechanism_valid"]
            and rmse_now < best_mech_rmse - args.early_stop_min_delta
        ):
            best_mech_rmse = rmse_now
            best_mech_epoch = epoch
            best_mech_val = va
            save_ckpt(
                out / "best_mechanistic_model.pt",
                model, optimizer, epoch, tr, va, args
            )
            print(
                f"SAVED BEST-MECHANISTIC | EPOCH={epoch} "
                f"| MSE={va['final']['mse']:.6f} "
                f"| INT_STD={va['mean_interaction_score_std']:.6g} "
                f"| TV={va['mean_tv_from_uniform']:.6g} "
                f"| DELTA_STD={va['delta_std']:.6g}",
                flush=True,
            )

        collapsed = (
            va["mean_interaction_score_std"] < args.pair_std_floor
            and va["mean_tv_from_uniform"] < args.tv_floor
        )
        if epoch >= args.mechanism_warmup and collapsed:
            mech_stale += 1
            print(
                f"MECHANISM WARNING | {mech_stale}/{args.mechanism_patience} "
                f"| INT_STD={va['mean_interaction_score_std']:.3e} "
                f"| TV={va['mean_tv_from_uniform']:.3e}",
                flush=True,
            )
        else:
            mech_stale = 0

        if mech_stale >= args.mechanism_patience > 0:
            stop_reason = "mechanism_collapse"
            print(
                "STOP: interaction mechanism collapsed; stop wasting epochs.",
                flush=True,
            )
            break

        if stale >= args.early_stop_patience > 0:
            stop_reason = "validation_early_stop"
            print(
                f"EARLY STOP | BEST_EPOCH={best_epoch}",
                flush=True,
            )
            break

    summary = {
        "model_version": "CLEAN_GLOBAL_STRICT_AR2_ALL_RESIDUE",
        "global_checkpoint": args.global_checkpoint,
        "split_json": args.split_json,
        "stop_reason": stop_reason,
        "calibration": calibration,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_mechanistic_epoch": best_mech_epoch,
        "best_mechanistic_val": best_mech_val,
    }

    mech_path = out / "best_mechanistic_model.pt"
    if mech_path.exists():
        best_ckpt = torch.load(
            mech_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            best_ckpt["model_state_dict"],
            strict=True,
        )

        causal = causal_audit(
            model, val_loader, device, args.seed
        )
        summary["causal_audit"] = causal

        print("\nCAUSAL AUDIT", flush=True)
        for key in (
            "original",
            "zero_z",
            "same_drug_mean_z",
            "global_permuted_z",
        ):
            print(fmt(key.upper(), causal[key]), flush=True)
        print(
            "PRED CHANGE: "
            + json.dumps(causal["mean_abs_prediction_change"]),
            flush=True,
        )
    else:
        summary["causal_audit"] = None
        print(
            "\nNO MECHANISM-VALID CHECKPOINT. "
            "Do not claim AR benefit.",
            flush=True,
        )

    if calibration is not None and best_mech_val is not None:
        affine_mse = calibration["val"]["affine"]["mse"]
        ar_mse = best_mech_val["final"]["mse"]
        summary["beats_affine_calibration"] = bool(ar_mse < affine_mse)
        print(
            f"\nPERFORMANCE GATE | AR_MSE={ar_mse:.6f} "
            f"| GLOBAL_AFFINE_MSE={affine_mse:.6f} "
            f"| BEATS_AFFINE={ar_mse < affine_mse}",
            flush=True,
        )

    (out / "best_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"\nOUTPUT={out.resolve()} "
        f"| STOP={stop_reason} "
        f"| BEST_MSE_EPOCH={best_epoch} "
        f"| BEST_MECH_EPOCH={best_mech_epoch}",
        flush=True,
    )


if __name__ == "__main__":
    main()
