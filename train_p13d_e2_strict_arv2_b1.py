# -*- coding: utf-8 -*-
"""
Train AR2-B1: Frozen E2 + Double-Centered Interaction Bottleneck.

Key safeguards
--------------
- E2 is frozen and kept in eval mode.
- Old selector contributes indices only; B1 directly takes its first sorted Top-24.
- Heavy atoms only on drug side.
- No direct D/P/old-zAR shortcut.
- Pair score and pair embedding are double centered.
- Attention is centered against the uniform null.
- Bias-free residual head ensures:
      uniform / no-interaction signal -> z_AR2=0 -> delta_AR2=0
- Automatic mechanism watchdog stops early if the interaction branch collapses.
- Best MSE checkpoint and best mechanism-valid checkpoint are saved separately.
- A causal latent audit is run automatically on the best mechanism-valid model.
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

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = (
    _THIS_FILE.parent.parent
    if _THIS_FILE.parent.name == "scripts"
    else _THIS_FILE.parent
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d
from models.model_p13d_e2_strict_arv2_b1 import (
    E2StrictARv2B1StageA,
)


# ---------------------------------------------------------------------
# Repro / utility
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Heavy-atom mapping
# ---------------------------------------------------------------------

class E2WithHeavyAtomMapping(Dataset):
    """
    Original E2 sample + heavy atom local indices only.

    Source cache:
      data/processed/davis/drug_functional_groups/<drug_id>.pt

    This cache was built after validating SDF atom order == E2 drug_3d node order.
    Functional-group labels are NOT used in AR2-B1.
    """

    def __init__(
        self,
        e2_dataset: Dataset,
        functional_group_dir: str,
    ):
        self.e2_dataset = e2_dataset
        self.functional_group_dir = Path(functional_group_dir)

        if not self.functional_group_dir.is_dir():
            raise NotADirectoryError(self.functional_group_dir)

        self.heavy_by_drug: Dict[str, torch.Tensor] = {}
        self.num_atoms_by_drug: Dict[str, int] = {}

        files = sorted(self.functional_group_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(
                f"No .pt caches under {self.functional_group_dir}"
            )

        for path in files:
            obj = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
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
                raise ValueError(f"{path}: no heavy atoms.")

            self.heavy_by_drug[did] = heavy
            self.num_atoms_by_drug[did] = int(obj["num_atoms_e2"])

    def __len__(self):
        return len(self.e2_dataset)

    def __getitem__(self, index: int):
        item = self.e2_dataset[index]
        did = _id_string(item["drug_id"])

        if did not in self.heavy_by_drug:
            raise KeyError(
                f"drug_id={did} has no heavy-atom cache."
            )

        return {
            "e2": item,
            "heavy_atom_indices_local": self.heavy_by_drug[did],
            "expected_num_atoms": self.num_atoms_by_drug[did],
        }


def heavy_atom_collate(samples: List[Dict[str, Any]]):
    e2_items = [sample["e2"] for sample in samples]
    out = mdta_collate_fn_p13d(e2_items)

    batch_size = len(samples)
    atom_batch = out["drug_3d"]["batch"].long()

    atom_counts = torch.bincount(
        atom_batch,
        minlength=batch_size,
    )
    atom_offsets = torch.cumsum(atom_counts, dim=0) - atom_counts

    max_heavy = max(
        int(sample["heavy_atom_indices_local"].numel())
        for sample in samples
    )
    max_heavy = max(max_heavy, 1)

    heavy_global = torch.full(
        (batch_size, max_heavy),
        -1,
        dtype=torch.long,
    )
    heavy_mask = torch.zeros(
        (batch_size, max_heavy),
        dtype=torch.bool,
    )

    for b, sample in enumerate(samples):
        local = sample["heavy_atom_indices_local"].long().view(-1)

        n_atoms = int(atom_counts[b].item())
        expected = int(sample["expected_num_atoms"])

        if n_atoms != expected:
            raise ValueError(
                f"drug_id={out['drug_id'][b]}: E2 atom count={n_atoms} "
                f"!= cache count={expected}"
            )

        if int(local.min().item()) < 0 or int(local.max().item()) >= n_atoms:
            raise IndexError(
                f"drug_id={out['drug_id'][b]}: invalid heavy atom index."
            )

        n = int(local.numel())
        heavy_global[b, :n] = local + atom_offsets[b]
        heavy_mask[b, :n] = True

    out["heavy_atom_indices"] = heavy_global
    out["heavy_atom_mask"] = heavy_mask
    return out


# ---------------------------------------------------------------------
# Dataset / loader / model
# ---------------------------------------------------------------------

def build_dataset(args):
    e2_dataset = DavisDatasetP13D(
        pairs_csv=args.pairs_csv,
        drug_1d_dir=args.drug_1d_dir,
        protein_1d_dir=args.protein_1d_dir,
        protein_3d_dir=args.protein_3d_dir,
        drug_2d_dir=args.drug_2d_dir,
        use_drug_2d=False,
        drug_3d_dir=args.drug_3d_dir,
        use_drug_3d=True,
    )
    return E2WithHeavyAtomMapping(
        e2_dataset=e2_dataset,
        functional_group_dir=args.functional_group_dir,
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


def checkpoint_args_dict(checkpoint):
    value = checkpoint.get("args", {})
    return value if isinstance(value, dict) else vars(value)


def build_model_from_e2_checkpoint(
    args,
    checkpoint,
    device,
):
    ck = checkpoint_args_dict(checkpoint)

    e2_hidden = int(ck.get("hidden_dim", 128))
    e2_candidate_k = int(ck.get("pocket_top_k", 64))
    e2_heads = int(ck.get("interaction_heads", 4))

    if args.hidden_dim != e2_hidden:
        raise ValueError(
            f"--hidden_dim={args.hidden_dim}, checkpoint hidden_dim={e2_hidden}"
        )
    if args.candidate_top_k != e2_candidate_k:
        raise ValueError(
            f"--candidate_top_k={args.candidate_top_k}, "
            f"checkpoint pocket_top_k={e2_candidate_k}"
        )
    if args.ar_top_k > args.candidate_top_k:
        raise ValueError("ar_top_k cannot exceed candidate_top_k.")

    model = E2StrictARv2B1StageA(
        drug_1d_in_dim=int(ck.get("drug_1d_in_dim", 768)),
        drug_3d_node_in_dim=int(ck.get("drug_3d_node_in_dim", 10)),
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=e2_hidden,
        dropout=float(ck.get("dropout", 0.1)),
        task="regression",
        candidate_top_k=e2_candidate_k,
        old_interaction_heads=e2_heads,
        interaction_dim=args.interaction_dim,
        ar_top_k=args.ar_top_k,
        freeze_e2=True,
    ).to(device)

    model.load_e2_checkpoint_state(
        checkpoint["model_state_dict"]
    )
    model.set_e2_frozen(True)

    return model


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def mean_valid_std(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    out = []
    for b in range(values.size(0)):
        x = values[b][mask[b]]
        if x.numel() > 1:
            out.append(x.float().std(unbiased=False))
    if not out:
        return 0.0
    return float(torch.stack(out).mean().item())


def mechanism_valid(result, args) -> bool:
    return (
        result["mean_interaction_pair_score_std"] >= args.pair_std_floor
        and result["mean_tv_from_uniform"] >= args.tv_floor
        and result["mean_ar2_z_norm"] >= args.z_norm_floor
    )


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------

def run_epoch(
    model,
    loader,
    device,
    optimizer,
    args,
):
    training = optimizer is not None
    model.train(training)

    final_preds = []
    e2_preds = []
    base_preds = []
    deltas = []
    targets = []

    count = 0
    loss_sum = 0.0

    diag_sums = {
        "abs_old_delta": 0.0,
        "abs_ar2_delta": 0.0,
        "entropy": 0.0,
        "raw_score_std": 0.0,
        "interaction_score_std": 0.0,
        "tv": 0.0,
        "z_norm": 0.0,
        "selected_r": 0.0,
    }
    center_residual_max = 0.0

    trainable_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("e2.") and p.requires_grad
    ]

    for step, batch in enumerate(loader, 1):
        batch = move_to_device(batch, device)
        target = batch["label"].float()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            d = model(batch, return_details=True)
            pred = d["pred"].float()
            loss = F.mse_loss(pred, target)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at step={step}: {loss.item()}"
                )

            if training:
                loss.backward()

                if step == 1:
                    bad = [
                        name
                        for name, p in model.e2.named_parameters()
                        if p.grad is not None
                    ]
                    if bad:
                        raise RuntimeError(
                            "Frozen E2 received gradients: "
                            + ", ".join(bad[:10])
                        )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=args.grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            else:
                grad_norm = torch.zeros((), device=device)

        n = target.size(0)
        count += n
        loss_sum += float(loss.detach().item()) * n

        raw_std = mean_valid_std(
            d["ar2_raw_pair_scores"].detach(),
            d["ar2_pair_mask"].detach(),
        )
        int_std = mean_valid_std(
            d["ar2_interaction_pair_scores"].detach(),
            d["ar2_pair_mask"].detach(),
        )

        diag_sums["abs_old_delta"] += float(
            d["old_ar_delta"].detach().abs().mean().item()
        ) * n
        diag_sums["abs_ar2_delta"] += float(
            d["ar2_delta"].detach().abs().mean().item()
        ) * n
        diag_sums["entropy"] += float(
            d["ar2_normalized_pair_entropy"].detach().mean().item()
        ) * n
        diag_sums["raw_score_std"] += raw_std * n
        diag_sums["interaction_score_std"] += int_std * n
        diag_sums["tv"] += float(
            d["ar2_tv_from_uniform"].detach().mean().item()
        ) * n
        diag_sums["z_norm"] += float(
            d["ar2_z_norm"].detach().mean().item()
        ) * n
        diag_sums["selected_r"] += float(
            d["ar2_selected_residue_mask"]
            .detach()
            .sum(dim=1)
            .float()
            .mean()
            .item()
        ) * n

        center_residual_max = max(
            center_residual_max,
            float(d["ar2_center_residual_max"].detach().item()),
        )

        final_preds.append(pred.detach().view(-1).cpu())
        e2_preds.append(d["e2_pred"].detach().view(-1).cpu())
        base_preds.append(d["base_pred"].detach().view(-1).cpu())
        deltas.append(d["ar2_delta"].detach().view(-1).cpu())
        targets.append(target.detach().view(-1).cpu())

        if (
            training
            and args.log_interval > 0
            and step % args.log_interval == 0
        ):
            print(
                f"  STEP {step:05d}/{len(loader):05d} "
                f"| LOSS={loss.item():.6f} "
                f"| |AR2_DELTA|={d['ar2_delta'].detach().abs().mean().item():.5f} "
                f"| RAW_STD={raw_std:.5f} "
                f"| INT_STD={int_std:.5f} "
                f"| ENT={d['ar2_normalized_pair_entropy'].detach().mean().item():.4f} "
                f"| TV={d['ar2_tv_from_uniform'].detach().mean().item():.5f} "
                f"| ZNORM={d['ar2_z_norm'].detach().mean().item():.5f} "
                f"| GRAD={float(grad_norm):.4f}",
                flush=True,
            )

        if args.max_train_batches > 0 and training:
            if step >= args.max_train_batches:
                break
        if args.max_eval_batches > 0 and not training:
            if step >= args.max_eval_batches:
                break

    y = torch.cat(targets).numpy()
    final_p = torch.cat(final_preds).numpy()
    e2_p = torch.cat(e2_preds).numpy()
    base_p = torch.cat(base_preds).numpy()
    delta_p = torch.cat(deltas).numpy()

    result = {
        "final": metrics(y, final_p),
        "e2": metrics(y, e2_p),
        "base": metrics(y, base_p),
        "loss": loss_sum / max(count, 1),
        "mean_abs_old_ar_delta": (
            diag_sums["abs_old_delta"] / max(count, 1)
        ),
        "mean_abs_ar2_delta": (
            diag_sums["abs_ar2_delta"] / max(count, 1)
        ),
        "ar2_delta_std": float(np.std(delta_p)),
        "mean_normalized_pair_entropy": (
            diag_sums["entropy"] / max(count, 1)
        ),
        "mean_raw_pair_score_std": (
            diag_sums["raw_score_std"] / max(count, 1)
        ),
        "mean_interaction_pair_score_std": (
            diag_sums["interaction_score_std"] / max(count, 1)
        ),
        "mean_tv_from_uniform": (
            diag_sums["tv"] / max(count, 1)
        ),
        "mean_ar2_z_norm": (
            diag_sums["z_norm"] / max(count, 1)
        ),
        "mean_selected_residue_count": (
            diag_sums["selected_r"] / max(count, 1)
        ),
        "center_residual_max": center_residual_max,
    }
    result["mechanism_valid"] = mechanism_valid(result, args)

    return result, y, final_p


def print_epoch(prefix, result):
    print(fmt(f"{prefix} FINAL", result["final"]), flush=True)
    print(fmt(f"{prefix} E2   ", result["e2"]), flush=True)
    print(fmt(f"{prefix} BASE ", result["base"]), flush=True)

    print(
        f"{prefix} DIAG | "
        f"|AR2_DELTA|={result['mean_abs_ar2_delta']:.6f} "
        f"| DELTA_STD={result['ar2_delta_std']:.6f} "
        f"| RAW_STD={result['mean_raw_pair_score_std']:.6f} "
        f"| INT_STD={result['mean_interaction_pair_score_std']:.6f} "
        f"| ENT={result['mean_normalized_pair_entropy']:.6f} "
        f"| TV={result['mean_tv_from_uniform']:.6f} "
        f"| ZNORM={result['mean_ar2_z_norm']:.6f} "
        f"| CENTER_ERR={result['center_residual_max']:.2e} "
        f"| R={result['mean_selected_residue_count']:.2f} "
        f"| MECH={'PASS' if result['mechanism_valid'] else 'FAIL'}",
        flush=True,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_result,
    val_result,
    args,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_result": train_result,
            "val_result": val_result,
            "args": vars(args),
            "model_version": "E2_STRICT_ARV2_B1",
        },
        path,
    )


# ---------------------------------------------------------------------
# Causal latent audit
# ---------------------------------------------------------------------

@torch.no_grad()
def collect_val_latents(
    model,
    loader,
    device,
):
    model.eval()

    y_all = []
    e2_all = []
    z_all = []
    drug_ids = []

    for batch_cpu in loader:
        drug_ids.extend(
            [_id_string(x) for x in batch_cpu["drug_id"]]
        )
        batch = move_to_device(batch_cpu, device)
        d = model(batch, return_details=True)

        y_all.append(batch["label"].detach().view(-1).cpu())
        e2_all.append(d["e2_pred"].detach().view(-1).cpu())
        z_all.append(d["ar2_feat"].detach().cpu())

    return (
        torch.cat(y_all),
        torch.cat(e2_all),
        torch.cat(z_all),
        drug_ids,
    )


@torch.no_grad()
def head_on_cpu_latents(
    model,
    z_cpu,
    device,
    batch_size=512,
):
    out = []
    for s in range(0, z_cpu.size(0), batch_size):
        e = min(s + batch_size, z_cpu.size(0))
        z = z_cpu[s:e].to(device)
        d = model.ar2_delta_head(z).view(-1).cpu()
        out.append(d)
    return torch.cat(out)


def run_causal_audit(
    model,
    loader,
    device,
    seed,
):
    y, e2, z, drug_ids = collect_val_latents(
        model,
        loader,
        device,
    )

    original_delta = head_on_cpu_latents(
        model,
        z,
        device,
    )
    pred_original = e2 + original_delta

    z_zero = torch.zeros_like(z)
    pred_zero = e2 + head_on_cpu_latents(
        model,
        z_zero,
        device,
    )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(z.size(0), generator=g)
    pred_perm = e2 + head_on_cpu_latents(
        model,
        z[perm],
        device,
    )

    z_drugmean = torch.empty_like(z)
    for did in sorted(set(drug_ids)):
        idx = [
            i for i, x in enumerate(drug_ids)
            if x == did
        ]
        ii = torch.as_tensor(idx, dtype=torch.long)
        mean_z = z[ii].mean(0, keepdim=True)
        z_drugmean[ii] = mean_z

    pred_drugmean = e2 + head_on_cpu_latents(
        model,
        z_drugmean,
        device,
    )

    yn = y.numpy()
    original_np = pred_original.numpy()

    result = {
        "original": metrics(yn, original_np),
        "zero_z": metrics(yn, pred_zero.numpy()),
        "same_drug_mean_z": metrics(
            yn,
            pred_drugmean.numpy(),
        ),
        "global_permuted_z": metrics(
            yn,
            pred_perm.numpy(),
        ),
        "mean_abs_prediction_change": {
            "zero_z": float(
                np.mean(
                    np.abs(
                        pred_zero.numpy()
                        - original_np
                    )
                )
            ),
            "same_drug_mean_z": float(
                np.mean(
                    np.abs(
                        pred_drugmean.numpy()
                        - original_np
                    )
                )
            ),
            "global_permuted_z": float(
                np.mean(
                    np.abs(
                        pred_perm.numpy()
                        - original_np
                    )
                )
            ),
        },
    }
    return result


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pairs_csv",
        default="data/raw/davis/pairs.csv",
    )
    p.add_argument(
        "--drug_1d_dir",
        default="data/processed/davis/drug_1d_chemberta2",
    )
    p.add_argument(
        "--drug_2d_dir",
        default="data/processed/davis/drug_2d",
    )
    p.add_argument(
        "--drug_3d_dir",
        default="data/processed/davis/drug_3d",
    )
    p.add_argument(
        "--protein_1d_dir",
        default="data/processed/davis/protein_1d_esm2",
    )
    p.add_argument(
        "--protein_3d_dir",
        default="data/processed/davis/protein_3d_gvp",
    )
    p.add_argument(
        "--functional_group_dir",
        default="data/processed/davis/drug_functional_groups",
    )

    p.add_argument("--e2_checkpoint", required=True)
    p.add_argument("--split_json", default=None)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--ar2_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--early_stop_patience", type=int, default=25)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--log_interval", type=int, default=100)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--interaction_dim", type=int, default=128)
    p.add_argument("--candidate_top_k", type=int, default=64)
    p.add_argument("--ar_top_k", type=int, default=24)

    # Mechanism watchdog.
    p.add_argument("--mechanism_warmup", type=int, default=3)
    p.add_argument("--mechanism_patience", type=int, default=3)
    p.add_argument("--pair_std_floor", type=float, default=1e-4)
    p.add_argument("--tv_floor", type=float, default=1e-4)
    p.add_argument("--z_norm_floor", type=float, default=1e-5)

    # Fast smoke/debug controls. 0 means full.
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_eval_batches", type=int, default=0)
    p.add_argument("--smoke_only", action="store_true")

    p.add_argument("--allow_overwrite", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    e2_path = Path(args.e2_checkpoint)
    if not e2_path.exists():
        raise FileNotFoundError(e2_path)

    if args.split_json is None:
        inferred = e2_path.parent / "split_indices.json"
        if not inferred.exists():
            raise FileNotFoundError(
                f"No split_indices.json next to E2 checkpoint: {inferred}"
            )
        args.split_json = str(inferred)

    output = Path(args.output_dir)
    if output.exists() and not args.allow_overwrite:
        meaningful = [
            x for x in output.iterdir()
            if x.name not in {"train.log", "pid.txt", "launcher.log"}
        ]
        if meaningful:
            raise FileExistsError(
                f"Output directory already contains artifacts: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)

    split = json.loads(
        Path(args.split_json).read_text(encoding="utf-8")
    )
    if "train_indices" not in split or "val_indices" not in split:
        raise KeyError(
            "split_json must contain train_indices and val_indices."
        )

    checkpoint = torch.load(
        e2_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError("E2 checkpoint has no model_state_dict.")

    print(f"E2 CHECKPOINT: {e2_path}", flush=True)
    print(f"E2 EPOCH: {checkpoint.get('epoch', 'N/A')}", flush=True)

    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dataset = build_dataset(args)
    train_loader, val_loader, test_loader = build_loaders(
        args,
        dataset,
        split,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = build_model_from_e2_checkpoint(
        args,
        checkpoint,
        device,
    )

    e2_trainable = sum(
        p.numel()
        for p in model.e2.parameters()
        if p.requires_grad
    )
    ar2_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("e2.") and p.requires_grad
    ]
    ar2_trainable = sum(p.numel() for p in ar2_params)

    if e2_trainable != 0:
        raise RuntimeError(
            f"E2 must be frozen, but {e2_trainable} params are trainable."
        )
    if ar2_trainable == 0:
        raise RuntimeError("No trainable AR2-B1 parameters.")

    print(
        f"DEVICE={device} | TRAIN={len(train_loader.dataset)} "
        f"| VAL={len(val_loader.dataset)}",
        flush=True,
    )
    print(
        f"TRAINABLE | E2=0 | AR2-B1={ar2_trainable:,}",
        flush=True,
    )
    print(
        "MODEL=AR2-B1 | TOP24_FIXED_FROM_OLD_SELECTOR "
        "| HEAVY_ATOMS_ONLY | MULTIPLICATIVE_PAIR_ONLY "
        "| DOUBLE_CENTER_SCORE | CENTERED_UNIFORM_ATTENTION "
        "| DOUBLE_CENTER_EDGE | BIAS_FREE_DELTA_HEAD",
        flush=True,
    )

    # Exact zero-preserving property.
    zero_delta = model.zero_input_delta(
        batch_size=4,
        device=device,
    )
    zero_delta_max = float(zero_delta.abs().max().item())
    print(
        f"ZERO PROPERTY | max_abs(H(0))={zero_delta_max:.12g}",
        flush=True,
    )
    if zero_delta_max != 0.0:
        raise RuntimeError("Bias-free zero-preserving property failed.")

    optimizer = torch.optim.Adam(
        ar2_params,
        lr=args.ar2_lr,
        weight_decay=args.weight_decay,
    )

    # Initial val sanity.
    init_val, _, _ = run_epoch(
        model,
        val_loader,
        device,
        None,
        args,
    )
    print_epoch("INIT-VAL", init_val)

    max_init_diff = abs(
        init_val["final"]["mse"] - init_val["e2"]["mse"]
    )
    if max_init_diff > 1e-7:
        raise RuntimeError(
            f"INIT final != E2: MSE difference={max_init_diff}"
        )

    if init_val["center_residual_max"] > 1e-5:
        raise RuntimeError(
            "Double-centering numerical invariant failed: "
            f"{init_val['center_residual_max']}"
        )

    if args.smoke_only:
        # One short train pass to verify backward/optimizer path.
        old_max = args.max_train_batches
        args.max_train_batches = 2
        smoke_train, _, _ = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args,
        )
        print_epoch("SMOKE-TRAIN", smoke_train)
        args.max_train_batches = old_max

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
    mechanism_stale = 0
    stop_reason = "max_epochs"

    for epoch in range(1, args.epochs + 1):
        print(
            f"\nEPOCH {epoch}/{args.epochs} "
            f"| E2=FROZEN | AR2_B1_LR={args.ar2_lr:g}",
            flush=True,
        )

        train_result, _, _ = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args,
        )
        val_result, val_y, val_p = run_epoch(
            model,
            val_loader,
            device,
            None,
            args,
        )

        print_epoch("TRAIN", train_result)
        print_epoch("VAL  ", val_result)

        history.append(
            {
                "epoch": epoch,
                "train": train_result,
                "val": val_result,
            }
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

        # Best-by-MSE, regardless of mechanism.
        if current_rmse < best_rmse - args.early_stop_min_delta:
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
                f"SAVED BEST-MSE | EPOCH={epoch} "
                f"| FINAL_MSE={val_result['final']['mse']:.6f} "
                f"| E2_MSE={val_result['e2']['mse']:.6f} "
                f"| DELTA="
                f"{val_result['final']['mse'] - val_result['e2']['mse']:+.6f}",
                flush=True,
            )
        else:
            stale += 1
            print(
                f"NO MSE IMPROVEMENT | STALE="
                f"{stale}/{args.early_stop_patience}",
                flush=True,
            )

        # Best mechanism-valid checkpoint.
        if (
            val_result["mechanism_valid"]
            and current_rmse
            < best_mech_rmse - args.early_stop_min_delta
        ):
            best_mech_rmse = current_rmse
            best_mech_epoch = epoch
            best_mech_val = val_result

            save_checkpoint(
                output / "best_mechanistic_model.pt",
                model,
                optimizer,
                epoch,
                train_result,
                val_result,
                args,
            )

            print(
                f"SAVED BEST-MECHANISTIC | EPOCH={epoch} "
                f"| MSE={val_result['final']['mse']:.6f} "
                f"| INT_STD="
                f"{val_result['mean_interaction_pair_score_std']:.6g} "
                f"| TV={val_result['mean_tv_from_uniform']:.6g}",
                flush=True,
            )

        # Mechanism watchdog.
        collapsed = (
            val_result["mean_interaction_pair_score_std"]
            < args.pair_std_floor
            and val_result["mean_tv_from_uniform"] < args.tv_floor
        )

        if epoch >= args.mechanism_warmup and collapsed:
            mechanism_stale += 1
            print(
                f"MECHANISM WARNING | collapse_count="
                f"{mechanism_stale}/{args.mechanism_patience} "
                f"| INT_STD="
                f"{val_result['mean_interaction_pair_score_std']:.3e} "
                f"| TV={val_result['mean_tv_from_uniform']:.3e}",
                flush=True,
            )
        else:
            mechanism_stale = 0

        if (
            args.mechanism_patience > 0
            and mechanism_stale >= args.mechanism_patience
        ):
            stop_reason = "mechanism_collapse"
            print(
                "STOP: interaction mechanism collapsed. "
                "No reason to burn more epochs.",
                flush=True,
            )
            break

        if (
            args.early_stop_patience > 0
            and stale >= args.early_stop_patience
        ):
            stop_reason = "validation_early_stop"
            print(
                f"EARLY STOP | BEST_EPOCH={best_epoch}",
                flush=True,
            )
            break

    summary = {
        "model_version": "E2_STRICT_ARV2_B1",
        "stop_reason": stop_reason,
        "e2_checkpoint": str(e2_path),
        "split_json": args.split_json,
        "design": {
            "e2_frozen": True,
            "old_selector": "first sorted Top24 indices only",
            "heavy_atoms_only": True,
            "multiplicative_pair_only": True,
            "double_center_score": True,
            "center_uniform_attention": True,
            "double_center_pair_edge": True,
            "bias_free_delta_head": True,
        },
        "initial_val": init_val,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_mechanistic_epoch": best_mech_epoch,
        "best_mechanistic_val": best_mech_val,
    }

    # Automatic causal audit only on a mechanism-valid checkpoint.
    mech_path = output / "best_mechanistic_model.pt"
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

        causal = run_causal_audit(
            model,
            val_loader,
            device,
            args.seed,
        )
        summary["causal_audit"] = causal

        print("\nCAUSAL AUDIT (BEST MECHANISTIC)", flush=True)
        for key in (
            "original",
            "zero_z",
            "same_drug_mean_z",
            "global_permuted_z",
        ):
            print(fmt(key.upper(), causal[key]), flush=True)

        print(
            "CAUSAL PRED CHANGE: "
            + json.dumps(
                causal["mean_abs_prediction_change"],
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        summary["causal_audit"] = None
        print(
            "\nNO MECHANISM-VALID CHECKPOINT. "
            "Do not interpret any MSE-only gain as AR evidence.",
            flush=True,
        )

    # Optional test only for mechanism-valid checkpoint.
    if test_loader is not None and mech_path.exists():
        test_result, _, _ = run_epoch(
            model,
            test_loader,
            device,
            None,
            args,
        )
        summary["test_mechanistic"] = test_result
        print_epoch("TEST-MECH", test_result)

    (output / "best_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nOUTPUT_DIR={output.resolve()}", flush=True)
    print(
        f"STOP_REASON={stop_reason} "
        f"| BEST_MSE_EPOCH={best_epoch} "
        f"| BEST_MECH_EPOCH={best_mech_epoch}",
        flush=True,
    )


if __name__ == "__main__":
    main()
