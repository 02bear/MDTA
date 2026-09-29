# -*- coding: utf-8 -*-
"""
Train E2-guided three-granularity DTA V1.

Training policy
---------------
- Load the fold-specific trained E2 best_model.pt.
- Freeze the entire pretrained E2 branch.
- Train ONLY the new FP branch and fp_delta_head.
- Use the original E2 data pipeline for:
      drug_1d, drug_3d, protein_1d, protein_3d, label
- Read BRICS mapping directly from multiscale/drug_brics_fragments.pt.
- Read CAVIAR mapping directly from protein_subpockets_caviar_v1/*.pt.
- drug_atom_features_v2 / protein_residue_features_v2 are NOT used by this Stage-A run.
- Pure final MSE loss only.
- Early stopping follows E2: validation RMSE.
- Log base / E2 / final metrics separately.

The first evaluation before training must reproduce the E2 checkpoint because
E2 is frozen/eval and fp_delta_head is zero initialized.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d

from models.model_p13d_e2_guided_three_granularity_v1_stageA_frozen import (
    E2GuidedThreeGranularityDTA,
)


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


def _id_string(x: Any) -> str:
    if torch.is_tensor(x):
        if x.numel() == 1:
            return str(x.item())
        return str(x.detach().cpu().tolist())
    return str(x)


def _scalar_label(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().view(-1)[0].cpu().item())
    return float(x)



def _to_index_tensor(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().long().view(-1)
    return torch.as_tensor(value, dtype=torch.long).view(-1)


def _lookup_brics_entry(cache: Any, drug_id: Any) -> Dict[str, Any]:
    """Robust lookup for drug_brics_fragments.pt across simple/nested dict layouts."""
    candidates = [drug_id, str(drug_id)]
    try:
        candidates.append(int(str(drug_id)))
    except Exception:
        pass

    containers = [cache]
    if isinstance(cache, dict):
        for key in ("drugs", "data", "entries", "fragments", "drug_fragments"):
            value = cache.get(key)
            if isinstance(value, dict):
                containers.append(value)

    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in candidates:
            if key in container:
                entry = container[key]
                if not isinstance(entry, dict):
                    raise TypeError(
                        f"BRICS entry for drug_id={drug_id!r} is not a dict: "
                        f"{type(entry)}"
                    )
                return entry

    sample_keys = []
    if isinstance(cache, dict):
        sample_keys = list(cache.keys())[:10]
    raise KeyError(
        f"drug_id={drug_id!r} was not found in BRICS cache. "
        f"Top-level sample keys={sample_keys}"
    )


class E2WithStructuralMappings(Dataset):
    """
    Original E2 sample + mapping-only structural metadata.

    Drug mapping:
      data/processed/davis/multiscale/drug_brics_fragments.pt
      fragment_atom_indices are SDF graph-node indices, so they index the
      original E2 drug_3d graph (which keeps explicit hydrogens).

    Protein mapping:
      data/processed/davis/protein_subpockets_caviar_v1/<protein_id>.pt
      each subpocket stores protein_graph_node_indices aligned to protein_3d_gvp.

    No drug_atom_features_v2 or protein_residue_features_v2 tensors enter E2.
    """

    def __init__(
        self,
        e2_dataset: Dataset,
        drug_brics_cache: str,
        protein_subpocket_dir: str,
    ):
        self.e2_dataset = e2_dataset
        self.drug_brics_cache_path = Path(drug_brics_cache)
        self.protein_subpocket_dir = Path(protein_subpocket_dir)

        if not self.drug_brics_cache_path.exists():
            raise FileNotFoundError(self.drug_brics_cache_path)
        if not self.protein_subpocket_dir.is_dir():
            raise NotADirectoryError(self.protein_subpocket_dir)

        self.drug_brics_cache = torch.load(
            self.drug_brics_cache_path,
            map_location="cpu",
            weights_only=False,
        )

        self.subpocket_files = {
            path.stem: path
            for path in self.protein_subpocket_dir.glob("*.pt")
        }
        if not self.subpocket_files:
            raise FileNotFoundError(
                f"No .pt files found under {self.protein_subpocket_dir}"
            )

    def __len__(self):
        return len(self.e2_dataset)

    def __getitem__(self, index: int):
        item = self.e2_dataset[index]
        drug_id = item["drug_id"]
        protein_id = item["protein_id"]

        brics = _lookup_brics_entry(self.drug_brics_cache, drug_id)
        if "fragment_atom_indices" not in brics:
            raise KeyError(
                f"BRICS cache entry for drug_id={drug_id!r} does not contain "
                "'fragment_atom_indices'."
            )

        pid = str(protein_id)
        subpocket_path = self.subpocket_files.get(pid)
        if subpocket_path is None:
            # Some datasets may carry scalar tensors/integers as IDs.
            pid_alt = _id_string(protein_id)
            subpocket_path = self.subpocket_files.get(pid_alt)
        if subpocket_path is None:
            raise FileNotFoundError(
                f"No CAVIAR cache file for protein_id={protein_id!r} "
                f"under {self.protein_subpocket_dir}"
            )

        pocket_cache = torch.load(
            subpocket_path,
            map_location="cpu",
            weights_only=False,
        )
        subpockets = pocket_cache.get("subpockets", [])
        if not isinstance(subpockets, (list, tuple)):
            raise TypeError(
                f"{subpocket_path}: 'subpockets' must be a list/tuple."
            )

        fragment_lists = [
            _to_index_tensor(fragment)
            for fragment in brics["fragment_atom_indices"]
        ]
        pocket_lists = []
        for pocket_index, pocket in enumerate(subpockets):
            if "protein_graph_node_indices" not in pocket:
                raise KeyError(
                    f"{subpocket_path}: subpocket[{pocket_index}] is missing "
                    "'protein_graph_node_indices'."
                )
            pocket_lists.append(
                _to_index_tensor(pocket["protein_graph_node_indices"])
            )

        return {
            "e2": item,
            "fragment_atom_indices_local": fragment_lists,
            "pocket_residue_indices_local": pocket_lists,
        }


def _cap_or_full(n: int, cap: int) -> int:
    return min(n, cap) if cap and cap > 0 else n


def mapping_collate(
    samples: List[Dict[str, Any]],
    max_fragments: int,
    max_atoms_per_fragment: int,
    max_subpockets: int,
    max_residues_per_subpocket: int,
):
    e2_items = [sample["e2"] for sample in samples]
    out = mdta_collate_fn_p13d(e2_items)

    batch_size = len(samples)
    atom_counts = torch.bincount(
        out["drug_3d"]["batch"].long(), minlength=batch_size
    )
    residue_counts = torch.bincount(
        out["protein_3d"]["batch"].long(), minlength=batch_size
    )
    atom_offsets = torch.cumsum(atom_counts, dim=0) - atom_counts
    residue_offsets = torch.cumsum(residue_counts, dim=0) - residue_counts

    # Determine padded batch sizes. Default caps of 0 mean "do not truncate".
    fragment_count = max(
        (len(sample["fragment_atom_indices_local"]) for sample in samples),
        default=0,
    )
    pocket_count = max(
        (len(sample["pocket_residue_indices_local"]) for sample in samples),
        default=0,
    )
    F = _cap_or_full(fragment_count, max_fragments)
    P = _cap_or_full(pocket_count, max_subpockets)

    atom_per_fragment = 0
    residue_per_pocket = 0
    for sample in samples:
        for fragment in sample["fragment_atom_indices_local"][:F]:
            atom_per_fragment = max(atom_per_fragment, int(fragment.numel()))
        for pocket in sample["pocket_residue_indices_local"][:P]:
            residue_per_pocket = max(residue_per_pocket, int(pocket.numel()))

    A = _cap_or_full(atom_per_fragment, max_atoms_per_fragment)
    R = _cap_or_full(residue_per_pocket, max_residues_per_subpocket)

    # Model expects non-empty padded dimensions.
    F = max(F, 1)
    P = max(P, 1)
    A = max(A, 1)
    R = max(R, 1)

    fragment_atom_indices = torch.full(
        (batch_size, F, A), -1, dtype=torch.long
    )
    fragment_atom_mask = torch.zeros(
        (batch_size, F, A), dtype=torch.bool
    )
    fragment_mask = torch.zeros((batch_size, F), dtype=torch.bool)

    pocket_residue_indices = torch.full(
        (batch_size, P, R), -1, dtype=torch.long
    )
    pocket_residue_mask = torch.zeros(
        (batch_size, P, R), dtype=torch.bool
    )
    pocket_mask = torch.zeros((batch_size, P), dtype=torch.bool)

    for b, sample in enumerate(samples):
        atom_n = int(atom_counts[b].item())
        residue_n = int(residue_counts[b].item())

        for f, local_index in enumerate(
            sample["fragment_atom_indices_local"][:F]
        ):
            local_index = local_index.long()
            local_index = local_index[
                (local_index >= 0) & (local_index < atom_n)
            ]
            if local_index.numel() == 0:
                continue
            local_index = local_index[:A]
            n = local_index.numel()
            fragment_atom_indices[b, f, :n] = (
                local_index + atom_offsets[b]
            )
            fragment_atom_mask[b, f, :n] = True
            fragment_mask[b, f] = True

        for p, local_index in enumerate(
            sample["pocket_residue_indices_local"][:P]
        ):
            local_index = local_index.long()
            local_index = local_index[
                (local_index >= 0) & (local_index < residue_n)
            ]
            if local_index.numel() == 0:
                continue
            local_index = local_index[:R]
            n = local_index.numel()
            pocket_residue_indices[b, p, :n] = (
                local_index + residue_offsets[b]
            )
            pocket_residue_mask[b, p, :n] = True
            pocket_mask[b, p] = True

        if not fragment_mask[b].any():
            raise ValueError(
                f"Sample b={b}, drug_id={out['drug_id'][b]} has no valid "
                "BRICS fragment after mapping to the original E2 drug_3d graph."
            )
        if not pocket_mask[b].any():
            raise ValueError(
                f"Sample b={b}, protein_id={out['protein_id'][b]} has no valid "
                "CAVIAR subpocket after mapping to protein_3d_gvp."
            )

    out["fragment_atom_indices"] = fragment_atom_indices
    out["fragment_atom_mask"] = fragment_atom_mask
    out["fragment_mask"] = fragment_mask
    out["pocket_residue_indices"] = pocket_residue_indices
    out["pocket_residue_mask"] = pocket_residue_mask
    out["pocket_mask"] = pocket_mask
    return out


def build_datasets(args):
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

    return E2WithStructuralMappings(
        e2_dataset=e2_dataset,
        drug_brics_cache=args.drug_brics_cache,
        protein_subpocket_dir=args.protein_subpocket_dir,
    )


def build_loaders(args, dataset, split):
    collate = partial(
        mapping_collate,
        max_fragments=args.max_fragments,
        max_atoms_per_fragment=args.max_atoms_per_fragment,
        max_subpockets=args.max_subpockets,
        max_residues_per_subpocket=args.max_residues_per_subpocket,
    )
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    train_indices = split["train_indices"]
    val_indices = split["val_indices"]
    test_indices = split.get("test_indices", [])

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        shuffle=False,
        **common,
    )
    test_loader = (
        DataLoader(Subset(dataset, test_indices), shuffle=False, **common)
        if test_indices
        else None
    )
    return train_loader, val_loader, test_loader


def load_e2_checkpoint(model, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} does not contain 'model_state_dict'."
        )

    model.load_e2_checkpoint_state(checkpoint["model_state_dict"])

    print(f"LOADED E2 CHECKPOINT: {checkpoint_path}", flush=True)
    if "epoch" in checkpoint:
        print(f"E2 CHECKPOINT EPOCH: {checkpoint['epoch']}", flush=True)
    if "val_metrics" in checkpoint:
        print(
            "E2 SAVED VAL: "
            + json.dumps(checkpoint["val_metrics"], ensure_ascii=False),
            flush=True,
        )
    if "args" in checkpoint:
        ckpt_args = checkpoint["args"]
        for key in ("hidden_dim", "dropout", "pocket_top_k", "interaction_heads"):
            if key in ckpt_args:
                print(f"E2 CKPT ARG {key}={ckpt_args[key]}", flush=True)

    return checkpoint


def run_epoch(model, loader, device, optimizer, args):
    training = optimizer is not None
    model.train(training)

    final_preds = []
    e2_preds = []
    base_preds = []
    targets = []

    sum_loss = 0.0
    sum_abs_ar_delta = 0.0
    sum_abs_fp_delta = 0.0
    sum_fp_entropy = 0.0
    sum_null_weight = 0.0
    sum_fragment_ar_mass = 0.0
    count = 0

    for step, batch in enumerate(loader, 1):
        batch = move_to_device(batch, device)
        target = batch["label"].float()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            details = model(batch, return_details=True)
            prediction = details["pred"].float()
            loss = F.mse_loss(prediction, target)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"NON_FINITE loss at step={step}: {loss.item()}"
                )

            if training:
                loss.backward()

                # Stage-A invariant: frozen E2 must never receive gradients.
                if step == 1:
                    bad_e2_grads = [
                        name for name, parameter in model.e2.named_parameters()
                        if parameter.grad is not None
                    ]
                    if bad_e2_grads:
                        raise RuntimeError(
                            "Frozen E2 unexpectedly received gradients: "
                            + ", ".join(bad_e2_grads[:10])
                        )

                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=args.grad_clip,
                        error_if_nonfinite=True,
                    )
                else:
                    grad_norm = torch.zeros((), device=device)
                optimizer.step()
            else:
                grad_norm = torch.zeros((), device=device)

        n = target.size(0)
        count += n
        sum_loss += float(loss.detach().item()) * n
        sum_abs_ar_delta += float(details["ar_delta"].detach().abs().mean().item()) * n
        sum_abs_fp_delta += float(details["fp_delta"].detach().abs().mean().item()) * n
        sum_fp_entropy += float(details["mean_fp_entropy"].detach().item()) * n
        sum_null_weight += float(details["mean_null_weight"].detach().item()) * n
        sum_fragment_ar_mass += float(
            details["mean_fragment_ar_mass"].detach().item()
        ) * n

        final_preds.append(details["pred"].detach().float().view(-1).cpu())
        e2_preds.append(details["e2_pred"].detach().float().view(-1).cpu())
        base_preds.append(details["base_pred"].detach().float().view(-1).cpu())
        targets.append(target.detach().float().view(-1).cpu())

        if training and args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"  STEP {step:05d}/{len(loader):05d} "
                f"| LOSS={loss.item():.6f} "
                f"| |AR_DELTA|={details['ar_delta'].detach().abs().mean().item():.4f} "
                f"| |FP_DELTA|={details['fp_delta'].detach().abs().mean().item():.4f} "
                f"| NULL={details['mean_null_weight'].detach().item():.3f} "
                f"| FP_ENT={details['mean_fp_entropy'].detach().item():.3f} "
                f"| GRAD={float(grad_norm):.3f}",
                flush=True,
            )

    y = torch.cat(targets).numpy()
    final_p = torch.cat(final_preds).numpy()
    e2_p = torch.cat(e2_preds).numpy()
    base_p = torch.cat(base_preds).numpy()

    result = {
        "final": metrics(y, final_p),
        "e2": metrics(y, e2_p),
        "base": metrics(y, base_p),
        "loss": sum_loss / max(count, 1),
        "mean_abs_ar_delta": sum_abs_ar_delta / max(count, 1),
        "mean_abs_fp_delta": sum_abs_fp_delta / max(count, 1),
        "mean_fp_entropy": sum_fp_entropy / max(count, 1),
        "mean_null_weight": sum_null_weight / max(count, 1),
        "mean_fragment_ar_mass": sum_fragment_ar_mass / max(count, 1),
    }
    return result, y, final_p


def save_checkpoint(path, model, optimizer, epoch, train_result, val_result, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_result": train_result,
            "val_result": val_result,
            "args": vars(args),
            "model_version": "E2_GUIDED_THREE_GRANULARITY_V1_STAGE_A",
        },
        path,
    )


def print_epoch(prefix, result):
    print(fmt(f"{prefix} FINAL", result["final"]), flush=True)
    print(fmt(f"{prefix} E2   ", result["e2"]), flush=True)
    print(fmt(f"{prefix} BASE ", result["base"]), flush=True)
    print(
        f"{prefix} DIAG | |AR_DELTA|={result['mean_abs_ar_delta']:.6f} "
        f"| |FP_DELTA|={result['mean_abs_fp_delta']:.6f} "
        f"| FP_ENT={result['mean_fp_entropy']:.6f} "
        f"| NULL={result['mean_null_weight']:.6f} "
        f"| FRAG_AR_MASS={result['mean_fragment_ar_mass']:.6f}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser()

    # Original E2 inputs.
    parser.add_argument("--pairs_csv", default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", default="data/processed/davis/drug_1d_chemberta2")
    parser.add_argument("--drug_2d_dir", default="data/processed/davis/drug_2d")
    parser.add_argument("--drug_3d_dir", default="data/processed/davis/drug_3d")
    parser.add_argument("--protein_1d_dir", default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_3d_dir", default="data/processed/davis/protein_3d_gvp")

    # Mapping-only structural caches. These do NOT replace E2 inputs.
    parser.add_argument(
        "--drug_brics_cache",
        default="data/processed/davis/multiscale/drug_brics_fragments.pt",
    )
    parser.add_argument(
        "--protein_subpocket_dir",
        default="data/processed/davis/protein_subpockets_caviar_v1",
    )

    # Exact fold initialization and split.
    parser.add_argument("--e2_checkpoint", required=True)
    parser.add_argument(
        "--split_json",
        default=None,
        help=(
            "If omitted, use split_indices.json next to --e2_checkpoint. "
            "This is recommended to guarantee the exact E2 fold split."
        ),
    )
    parser.add_argument("--output_dir", required=True)

    # Training.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    # Stage A: E2 is frozen; only the new FP branch has an optimizer LR.
    parser.add_argument("--fp_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
        help="0 disables gradient clipping, matching original E2 training.",
    )

    parser.add_argument("--early_stop_patience", type=int, default=60)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=100)

    # E2 model hyperparameters: must match checkpoint.
    parser.add_argument("--drug_1d_in_dim", type=int, default=768)
    parser.add_argument("--drug_3d_node_in_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pocket_top_k", type=int, default=64)
    parser.add_argument("--interaction_heads", type=int, default=4)

    # Structural mapping truncation, following previous CAVIAR experiments.
    parser.add_argument("--max_fragments", type=int, default=0, help="0 = no extra truncation")
    parser.add_argument("--max_atoms_per_fragment", type=int, default=0, help="0 = no extra truncation")
    parser.add_argument("--max_subpockets", type=int, default=30)
    parser.add_argument("--max_residues_per_subpocket", type=int, default=48)

    parser.add_argument("--allow_overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    e2_checkpoint_path = Path(args.e2_checkpoint)
    if not e2_checkpoint_path.exists():
        raise FileNotFoundError(e2_checkpoint_path)

    if args.split_json is None:
        inferred = e2_checkpoint_path.parent / "split_indices.json"
        if not inferred.exists():
            raise FileNotFoundError(
                "No --split_json was given and the E2 fold directory does not "
                f"contain split_indices.json: {inferred}"
            )
        args.split_json = str(inferred)

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. "
            "Use a new directory or pass --allow_overwrite."
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    if "train_indices" not in split or "val_indices" not in split:
        raise KeyError("split_json must contain train_indices and val_indices.")

    dataset = build_datasets(args)
    train_loader, val_loader, test_loader = build_loaders(args, dataset, split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = E2GuidedThreeGranularityDTA(
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
        freeze_e2=True,
    ).to(device)

    checkpoint = load_e2_checkpoint(model, e2_checkpoint_path, device)

    # Stage A: the complete pretrained E2 branch MUST stay frozen.
    model.set_e2_frozen(True)

    e2_trainable = sum(
        p.numel() for p in model.e2.parameters() if p.requires_grad
    )
    fp_parameters = [
        p for name, p in model.named_parameters()
        if not name.startswith("e2.") and p.requires_grad
    ]
    fp_trainable = sum(p.numel() for p in fp_parameters)

    if e2_trainable != 0:
        raise RuntimeError(
            f"Stage A requires frozen E2, but {e2_trainable:,} E2 parameters "
            "are still trainable."
        )
    if fp_trainable == 0:
        raise RuntimeError("No trainable FP parameters were found.")

    print(
        f"DEVICE={device} | TRAIN={len(train_loader.dataset)} "
        f"| VAL={len(val_loader.dataset)} "
        f"| TEST={len(test_loader.dataset) if test_loader else 0}",
        flush=True,
    )
    print(
        f"TRAINABLE PARAMS | E2={e2_trainable:,} (FROZEN) "
        f"| FP={fp_trainable:,}",
        flush=True,
    )
    print(
        "MODEL=E2_GUIDED_THREE_GRANULARITY_V1_STAGE_A "
        "| E2(Global+AR) FROZEN/EVAL "
        "| BRICS/CAVIAR FP residual TRAINABLE "
        "| FINAL LOSS=MSE only",
        flush=True,
    )

    # Optimizer receives ONLY FP-side parameters.
    optimizer = torch.optim.Adam(
        fp_parameters,
        lr=args.fp_lr,
        weight_decay=args.weight_decay,
    )

    # Before any optimizer step, final must equal E2 exactly because fp_delta=0.
    model.eval()
    first_batch = move_to_device(next(iter(val_loader)), device)
    with torch.no_grad():
        sanity = model(first_batch, return_details=True)
    max_initial_diff = float(
        (sanity["pred"] - sanity["e2_pred"]).abs().max().item()
    )
    print(
        f"SANITY | max_abs(final-e2) before training = {max_initial_diff:.12g}",
        flush=True,
    )
    if max_initial_diff > 1e-7:
        raise RuntimeError(
            "FP zero-initialization failed: initial final prediction is not "
            "identical to E2."
        )

    # Full pre-training validation: should reproduce the fold E2 result.
    init_val, _, _ = run_epoch(model, val_loader, device, None, args)
    print_epoch("INIT-VAL", init_val)

    saved_val = checkpoint.get("val_metrics")
    if isinstance(saved_val, dict) and "mse" in saved_val:
        diff = abs(init_val["e2"]["mse"] - float(saved_val["mse"]))
        print(
            f"E2 REPRO CHECK | checkpoint MSE={float(saved_val['mse']):.6f} "
            f"| reloaded E2 MSE={init_val['e2']['mse']:.6f} "
            f"| abs_diff={diff:.6g}",
            flush=True,
        )
        if diff > 1e-3:
            print(
                "WARNING: reloaded E2 validation MSE differs from checkpoint by "
                ">1e-3. Check split/data/preprocessing before interpreting FP gains.",
                flush=True,
            )

    history = []
    best_rmse = float("inf")
    best_epoch = -1
    best_val = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        print(
            f"\nEPOCH {epoch}/{args.epochs} "
            f"| E2=FROZEN | FP_LR={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )

        train_result, _, _ = run_epoch(
            model, train_loader, device, optimizer, args
        )
        val_result, val_y, val_p = run_epoch(
            model, val_loader, device, None, args
        )

        print_epoch("TRAIN", train_result)
        print_epoch("VAL  ", val_result)

        record = {
            "epoch": epoch,
            "train": train_result,
            "val": val_result,
        }
        history.append(record)
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
        improved = current_rmse < best_rmse - args.early_stop_min_delta

        if improved:
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
                f"SAVED BEST | EPOCH={epoch} "
                f"| FINAL_MSE={val_result['final']['mse']:.6f} "
                f"| E2_MSE={val_result['e2']['mse']:.6f} "
                f"| BASE_MSE={val_result['base']['mse']:.6f}",
                flush=True,
            )
        else:
            stale += 1
            print(
                f"NO IMPROVEMENT | STALE={stale}/{args.early_stop_patience} "
                f"| BEST_EPOCH={best_epoch} | BEST_RMSE={best_rmse:.6f}",
                flush=True,
            )

        if (
            args.early_stop_patience > 0
            and stale >= args.early_stop_patience
        ):
            print(
                f"EARLY STOP | EPOCH={epoch} | BEST_EPOCH={best_epoch}",
                flush=True,
            )
            break

    summary = {
        "model_version": "E2_GUIDED_THREE_GRANULARITY_V1_STAGE_A",
        "e2_checkpoint": str(e2_checkpoint_path),
        "split_json": args.split_json,
        "e2_frozen": True,
        "loss": "final_mse_only",
        "best_epoch": best_epoch,
        "best_val": best_val,
        "initial_val": init_val,
    }

    # Optional test split if present in the supplied split JSON.
    if test_loader is not None and (output / "best_model.pt").exists():
        best_checkpoint = torch.load(
            output / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        test_result, test_y, test_p = run_epoch(
            model, test_loader, device, None, args
        )
        summary["test"] = test_result
        np.savez(
            output / "test_predictions_best_model.npz",
            target=test_y,
            prediction=test_p,
        )
        print_epoch("TEST ", test_result)

    (output / "best_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nOUTPUT_DIR={output.resolve()}", flush=True)
    print(
        f"BEST_EPOCH={best_epoch} | "
        f"BEST_VAL_FINAL_MSE="
        f"{best_val['final']['mse'] if best_val else float('nan'):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
