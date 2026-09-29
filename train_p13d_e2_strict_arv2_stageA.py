# -*- coding: utf-8 -*-
"""
Train AR2-A1:
    frozen fold-specific E2 + trainable Strict Cross-Only AR2 residual.

Experiment question
-------------------
Can a genuinely pair-conditioned atom-residue interaction provide stable
incremental affinity information beyond the current E2 predictor, when all
global D/P shortcuts are blocked from the new residual branch?

Fixed design
------------
- Original E2: frozen + eval.
- Old E2 selector: reused ONLY as Top-64 candidate residue indices.
- Drug AR2 nodes: heavy-atom Drug3D node tokens only.
- Protein AR2 nodes: raw Protein3D residue node tokens.
- New cross-only reranker: 64 candidates -> Top-24 AR residues.
- Bidirectional atom->residue and residue->atom conditional attention.
- delta_AR2 = H(z_AR2) only. No D/P/old-zAR direct input.
- Last AR2 residual layer zero initialized => INIT FINAL == E2.
- Loss: final MSE only.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d
from models.model_p13d_e2_strict_arv2 import E2StrictARv2StageA


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
    Add ONLY heavy-atom local indices to the original E2 sample.

    Source:
      data/processed/davis/drug_functional_groups/<drug_id>.pt

    The functional-group v2 preprocessing already verified:
      SDF atom order == drug_3d graph-node order.

    AR2 uses only `heavy_atom_indices`; functional-group definitions themselves
    are NOT used in this experiment.
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
                f"No .pt functional-group caches under "
                f"{self.functional_group_dir}"
            )

        for path in files:
            obj = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            if obj.get("version") != "drug_functional_groups_v2":
                raise ValueError(
                    f"{path}: expected version='drug_functional_groups_v2', "
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
                f"drug_id={did} has no functional-group v2 cache under "
                f"{self.functional_group_dir}"
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
                f"drug_id={out['drug_id'][b]}: collated E2 atom count "
                f"{n_atoms} != functional-group cache count {expected}"
            )

        if local.numel() == 0:
            raise ValueError(
                f"drug_id={out['drug_id'][b]} has no heavy atoms."
            )

        if int(local.min().item()) < 0 or int(local.max().item()) >= n_atoms:
            raise IndexError(
                f"drug_id={out['drug_id'][b]}: heavy-atom local index "
                f"outside [0,{n_atoms-1}]"
            )

        n = int(local.numel())
        heavy_global[b, :n] = local + atom_offsets[b]
        heavy_mask[b, :n] = True

    out["heavy_atom_indices"] = heavy_global
    out["heavy_atom_mask"] = heavy_mask
    return out


# ---------------------------------------------------------------------
# Dataset / loader / checkpoint
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
    if isinstance(value, dict):
        return value
    return vars(value)


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
            f"--hidden_dim={args.hidden_dim} but E2 checkpoint hidden_dim="
            f"{e2_hidden}. AR2-A1 requires the same node-token dimension."
        )
    if args.candidate_top_k != e2_candidate_k:
        raise ValueError(
            f"--candidate_top_k={args.candidate_top_k} but E2 checkpoint "
            f"pocket_top_k={e2_candidate_k}. A1 reuses the exact trained selector."
        )
    if args.ar_top_k > args.candidate_top_k:
        raise ValueError(
            f"ar_top_k={args.ar_top_k} cannot exceed candidate_top_k="
            f"{args.candidate_top_k}."
        )

    model = E2StrictARv2StageA(
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
# Train / eval
# ---------------------------------------------------------------------

def mean_valid_pair_score_std(
    scores: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    values = []
    for b in range(scores.size(0)):
        x = scores[b][mask[b]]
        if x.numel() > 1:
            values.append(x.float().std(unbiased=False))
    if not values:
        return 0.0
    return float(torch.stack(values).mean().item())


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
    targets = []

    count = 0
    sum_loss = 0.0
    sum_abs_old_ar_delta = 0.0
    sum_abs_ar2_delta = 0.0
    sum_joint_entropy = 0.0
    sum_pair_score_std = 0.0
    sum_selected_residues = 0.0

    ar2_parameters = [
        p for name, p in model.named_parameters()
        if not name.startswith("e2.") and p.requires_grad
    ]

    for step, batch in enumerate(loader, 1):
        batch = move_to_device(batch, device)
        target = batch["label"].float()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            details = model(
                batch,
                return_details=True,
            )
            prediction = details["pred"].float()
            loss = F.mse_loss(prediction, target)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"NON_FINITE loss at step={step}: {loss.item()}"
                )

            if training:
                loss.backward()

                # Frozen E2 is an invariant of AR2-A1.
                if step == 1:
                    bad = [
                        name
                        for name, p in model.e2.named_parameters()
                        if p.grad is not None
                    ]
                    if bad:
                        raise RuntimeError(
                            "Frozen E2 unexpectedly received gradients: "
                            + ", ".join(bad[:10])
                        )

                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        ar2_parameters,
                        max_norm=args.grad_clip,
                        error_if_nonfinite=True,
                    )
                else:
                    grad_sq = torch.zeros((), device=device)
                    for p in ar2_parameters:
                        if p.grad is not None:
                            grad_sq = (
                                grad_sq
                                + p.grad.detach().float().pow(2).sum()
                            )
                    grad_norm = grad_sq.sqrt()

                optimizer.step()
            else:
                grad_norm = torch.zeros((), device=device)

        n = target.size(0)
        count += n
        sum_loss += float(loss.detach().item()) * n
        sum_abs_old_ar_delta += float(
            details["old_ar_delta"].detach().abs().mean().item()
        ) * n
        sum_abs_ar2_delta += float(
            details["ar2_delta"].detach().abs().mean().item()
        ) * n
        sum_joint_entropy += float(
            details[
                "ar2_normalized_joint_entropy"
            ].detach().mean().item()
        ) * n
        sum_pair_score_std += mean_valid_pair_score_std(
            details["ar2_pair_scores"].detach(),
            details["ar2_pair_mask"].detach(),
        ) * n
        sum_selected_residues += float(
            details[
                "ar2_selected_residue_mask"
            ].detach().sum(dim=1).float().mean().item()
        ) * n

        final_preds.append(
            details["pred"].detach().float().view(-1).cpu()
        )
        e2_preds.append(
            details["e2_pred"].detach().float().view(-1).cpu()
        )
        base_preds.append(
            details["base_pred"].detach().float().view(-1).cpu()
        )
        targets.append(
            target.detach().float().view(-1).cpu()
        )

        if training and args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"  STEP {step:05d}/{len(loader):05d} "
                f"| LOSS={loss.item():.6f} "
                f"| |OLD_AR_DELTA|="
                f"{details['old_ar_delta'].detach().abs().mean().item():.4f} "
                f"| |AR2_DELTA|="
                f"{details['ar2_delta'].detach().abs().mean().item():.4f} "
                f"| AR2_ENT="
                f"{details['ar2_normalized_joint_entropy'].detach().mean().item():.3f} "
                f"| PAIR_STD="
                f"{mean_valid_pair_score_std(details['ar2_pair_scores'].detach(), details['ar2_pair_mask'].detach()):.4f} "
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
        "mean_abs_old_ar_delta": (
            sum_abs_old_ar_delta / max(count, 1)
        ),
        "mean_abs_ar2_delta": (
            sum_abs_ar2_delta / max(count, 1)
        ),
        "mean_ar2_normalized_joint_entropy": (
            sum_joint_entropy / max(count, 1)
        ),
        "mean_ar2_pair_score_std_within_pair": (
            sum_pair_score_std / max(count, 1)
        ),
        "mean_selected_residue_count": (
            sum_selected_residues / max(count, 1)
        ),
    }
    return result, y, final_p


def print_epoch(prefix, result):
    print(fmt(f"{prefix} FINAL", result["final"]), flush=True)
    print(fmt(f"{prefix} E2   ", result["e2"]), flush=True)
    print(fmt(f"{prefix} BASE ", result["base"]), flush=True)
    print(
        f"{prefix} DIAG | "
        f"|OLD_AR_DELTA|={result['mean_abs_old_ar_delta']:.6f} "
        f"| |AR2_DELTA|={result['mean_abs_ar2_delta']:.6f} "
        f"| AR2_ENT={result['mean_ar2_normalized_joint_entropy']:.6f} "
        f"| PAIR_STD={result['mean_ar2_pair_score_std_within_pair']:.6f} "
        f"| SELECTED_R={result['mean_selected_residue_count']:.2f}",
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
            "model_version": "E2_STRICT_ARV2_STAGE_A1",
        },
        path,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()

    # Original E2 inputs.
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

    # Mapping-only cache used solely to identify heavy atom indices.
    p.add_argument(
        "--functional_group_dir",
        default="data/processed/davis/drug_functional_groups",
    )

    # Fold-specific E2 and split.
    p.add_argument("--e2_checkpoint", required=True)
    p.add_argument(
        "--split_json",
        default=None,
        help=(
            "If omitted, use split_indices.json next to the E2 checkpoint."
        ),
    )
    p.add_argument("--output_dir", required=True)

    # Training.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--ar2_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--early_stop_patience", type=int, default=60)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--log_interval", type=int, default=100)

    # AR2-A1 fixed design.
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--interaction_dim", type=int, default=128)
    p.add_argument(
        "--candidate_top_k",
        type=int,
        default=64,
        help="Old E2 selector high-recall candidate pool; must match checkpoint.",
    )
    p.add_argument(
        "--ar_top_k",
        type=int,
        default=24,
        help="Actual residues entering Strict AR2.",
    )

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
                "No --split_json was given and E2 fold directory has no "
                f"split_indices.json: {inferred}"
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
                f"Output directory already contains artifacts: {output}. "
                "Use another output_dir or --allow_overwrite."
            )
    output.mkdir(parents=True, exist_ok=True)

    split = json.loads(
        Path(args.split_json).read_text(encoding="utf-8")
    )
    if "train_indices" not in split or "val_indices" not in split:
        raise KeyError(
            "split_json must contain train_indices and val_indices."
        )

    # Load E2 checkpoint FIRST so the exact old architecture can be enforced.
    checkpoint = torch.load(
        e2_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{e2_path} does not contain model_state_dict."
        )

    ck = checkpoint_args_dict(checkpoint)
    print(f"E2 CHECKPOINT: {e2_path}", flush=True)
    print(f"E2 EPOCH: {checkpoint.get('epoch', 'N/A')}", flush=True)
    if "val_metrics" in checkpoint:
        print(
            "E2 SAVED VAL: "
            + json.dumps(
                checkpoint["val_metrics"],
                ensure_ascii=False,
            ),
            flush=True,
        )
    for key in (
        "hidden_dim",
        "dropout",
        "pocket_top_k",
        "interaction_heads",
    ):
        if key in ck:
            print(
                f"E2 CKPT ARG {key}={ck[key]}",
                flush=True,
            )

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
            f"AR2-A1 requires frozen E2, but {e2_trainable:,} E2 "
            "parameters are trainable."
        )
    if ar2_trainable == 0:
        raise RuntimeError("No trainable AR2 parameters.")

    print(
        f"DEVICE={device} | TRAIN={len(train_loader.dataset)} "
        f"| VAL={len(val_loader.dataset)} "
        f"| TEST={len(test_loader.dataset) if test_loader else 0}",
        flush=True,
    )
    print(
        f"TRAINABLE PARAMS | E2={e2_trainable:,} (FROZEN) "
        f"| AR2={ar2_trainable:,}",
        flush=True,
    )
    print(
        "MODEL=E2_STRICT_ARV2_STAGE_A1 "
        "| OLD E2 FROZEN/EVAL "
        "| OLD SELECTOR INDICES ONLY "
        f"| CANDIDATE_K={args.candidate_top_k} "
        f"| AR_K={args.ar_top_k} "
        "| HEAVY_ATOMS_ONLY "
        "| STRICT_CROSS_ONLY "
        "| BIDIRECTIONAL_CONDITIONAL_ATTN "
        "| DELTA_HEAD_INPUT=z_AR2_ONLY "
        "| NO D/P/OLD-zAR SHORTCUT "
        "| LOSS=MSE",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        ar2_params,
        lr=args.ar2_lr,
        weight_decay=args.weight_decay,
    )

    # Zero-init sanity: before any training, final must equal E2.
    model.eval()
    first_batch = move_to_device(
        next(iter(val_loader)),
        device,
    )
    with torch.no_grad():
        sanity = model(first_batch, return_details=True)

    max_initial_diff = float(
        (
            sanity["pred"] - sanity["e2_pred"]
        ).abs().max().item()
    )
    print(
        f"SANITY | max_abs(final-e2) before training="
        f"{max_initial_diff:.12g}",
        flush=True,
    )
    if max_initial_diff > 1e-7:
        raise RuntimeError(
            "AR2 zero initialization failed: INIT final != E2."
        )

    # E2 reproduction sanity.
    init_val, _, _ = run_epoch(
        model,
        val_loader,
        device,
        None,
        args,
    )
    print_epoch("INIT-VAL", init_val)

    saved_val = checkpoint.get("val_metrics")
    if isinstance(saved_val, dict) and "mse" in saved_val:
        diff = abs(
            init_val["e2"]["mse"] - float(saved_val["mse"])
        )
        print(
            f"E2 REPRO CHECK | checkpoint MSE="
            f"{float(saved_val['mse']):.6f} "
            f"| reloaded={init_val['e2']['mse']:.6f} "
            f"| abs_diff={diff:.6g}",
            flush=True,
        )
        if diff > 1e-3:
            print(
                "WARNING: E2 validation reproduction differs by >1e-3. "
                "Do not interpret AR2 gains until split/data are checked.",
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
            f"| E2=FROZEN | AR2_LR="
            f"{optimizer.param_groups[0]['lr']:.3g}",
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
        improved = (
            current_rmse
            < best_rmse - args.early_stop_min_delta
        )

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
                f"| DELTA="
                f"{val_result['final']['mse'] - val_result['e2']['mse']:+.6f}",
                flush=True,
            )
        else:
            stale += 1
            print(
                f"NO IMPROVEMENT | STALE="
                f"{stale}/{args.early_stop_patience} "
                f"| BEST_EPOCH={best_epoch} "
                f"| BEST_RMSE={best_rmse:.6f}",
                flush=True,
            )

        if (
            args.early_stop_patience > 0
            and stale >= args.early_stop_patience
        ):
            print(
                f"EARLY STOP | EPOCH={epoch} "
                f"| BEST_EPOCH={best_epoch}",
                flush=True,
            )
            break

    summary = {
        "model_version": "E2_STRICT_ARV2_STAGE_A1",
        "e2_checkpoint": str(e2_path),
        "split_json": args.split_json,
        "e2_frozen": True,
        "old_selector_role": "candidate_indices_only",
        "candidate_top_k": args.candidate_top_k,
        "ar_top_k": args.ar_top_k,
        "heavy_atoms_only": True,
        "strict_cross_only": True,
        "bidirectional_conditional_attention": True,
        "delta_head_input": "z_ar2_only",
        "direct_global_shortcut": False,
        "loss": "final_mse_only",
        "initial_val": init_val,
        "best_epoch": best_epoch,
        "best_val": best_val,
    }

    if (
        test_loader is not None
        and (output / "best_model.pt").exists()
    ):
        best_ckpt = torch.load(
            output / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            best_ckpt["model_state_dict"],
            strict=True,
        )
        test_result, test_y, test_p = run_epoch(
            model,
            test_loader,
            device,
            None,
            args,
        )
        summary["test"] = test_result
        np.savez(
            output / "test_predictions_best_model.npz",
            target=test_y,
            prediction=test_p,
        )
        print_epoch("TEST ", test_result)

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
        f"BEST_EPOCH={best_epoch} | BEST_VAL_FINAL_MSE="
        f"{best_val['final']['mse'] if best_val else float('nan'):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
