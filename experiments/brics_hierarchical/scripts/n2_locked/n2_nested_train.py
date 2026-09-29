#!/usr/bin/env python3
"""Locked N2 inner epoch selection and fresh outer refit for BRICS graph models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_for_drugs(global_data, drugs) -> list[int]:
    wanted = {str(x) for x in drugs}
    rows = [i for i, value in enumerate(global_data["drug_id"]) if str(value) in wanted]
    observed = {str(global_data["drug_id"][i]) for i in rows}
    if observed != wanted:
        raise RuntimeError(f"drug row mapping mismatch: missing={sorted(wanted-observed)}")
    return rows


def load_selected_labels(pairs_csv: Path, allowed_rows: list[int], total_rows: int) -> torch.Tensor:
    """Parse the label field only for explicitly allowed rows; all others remain NaN."""
    allowed = set(map(int, allowed_rows))
    labels = torch.full((total_rows,), float("nan"), dtype=torch.float32)
    seen = set()
    with pairs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "label" not in (reader.fieldnames or []):
            raise RuntimeError("pairs CSV has no label column")
        for index, row in enumerate(reader):
            if index in allowed:
                labels[index] = float(row["label"])
                seen.add(index)
    if seen != allowed:
        raise RuntimeError(f"allowed label rows missing: {len(allowed-seen)}")
    if not torch.isfinite(labels[allowed_rows]).all():
        raise RuntimeError("non-finite selected labels")
    return labels


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "align": 0.0, "reconstruct": 0.0, "count": 0}
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction, debug = model(batch, use_mask=True, return_debug=True)
        mse = F.mse_loss(prediction, batch["label"])
        delta_penalty = debug["delta"].pow(2).mean()
        loss = (mse + args.lambda_align * debug["alignment_loss"]
                + args.lambda_reconstruct * debug["reconstruction_loss"]
                + args.lambda_delta * delta_penalty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        count = len(prediction)
        totals["loss"] += float(loss.detach()) * count
        totals["mse"] += float(mse.detach()) * count
        totals["align"] += float(debug["alignment_loss"].detach()) * count
        totals["reconstruct"] += float(debug["reconstruction_loss"].detach()) * count
        totals["count"] += count
    return {key: totals[key] / totals["count"] for key in ("loss", "mse", "align", "reconstruct")}


def trainable_state(model):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--phase", choices=["inner", "refit"], required=True)
    parser.add_argument("--fold", type=int, choices=range(2, 6), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--drug-cache", type=Path, required=True)
    parser.add_argument("--outer-split", type=Path, required=True)
    parser.add_argument("--inner-split", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--condition", choices=["real", "no_fragment_edges"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selected-epoch-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-align", type=float, default=0.05)
    parser.add_argument("--lambda-reconstruct", type=float, default=0.02)
    parser.add_argument("--lambda-delta", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts = args.project / "experiments/brics_hierarchical/scripts"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(args.project))
    import train_brics_graph_stage1 as graph
    from experiments.klifs85_interaction.train_klifs_interact import metrics

    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    if "label" in global_data or not global_data.get("label_independent", False):
        raise RuntimeError("N2 requires a label-free global cache")
    drug_data = torch.load(args.drug_cache, map_location="cpu", weights_only=False)
    outer = json.loads(args.outer_split.read_text(encoding="utf-8"))
    inner = json.loads(args.inner_split.read_text(encoding="utf-8"))
    outer_train = {str(x) for x in outer["train_drugs"]}
    inner_train = {str(x) for x in inner["inner_train_drugs"]}
    inner_val = {str(x) for x in inner["inner_val_drugs"]}
    if inner_train & inner_val or inner_train | inner_val != outer_train:
        raise RuntimeError("inner split is not an exact partition of outer train")

    if args.phase == "inner":
        train_rows = rows_for_drugs(global_data, inner["inner_train_drugs"])
        val_rows = rows_for_drugs(global_data, inner["inner_val_drugs"])
        allowed_rows = train_rows + val_rows
        selected_epochs = None
    else:
        if args.selected_epoch_json is None:
            raise ValueError("refit requires --selected-epoch-json")
        selection = json.loads(args.selected_epoch_json.read_text(encoding="utf-8"))
        selected_epochs = int(selection["best_epoch"])
        if not 0 <= selected_epochs <= args.epochs:
            raise RuntimeError(f"invalid selected epoch: {selected_epochs}")
        train_rows = rows_for_drugs(global_data, outer["train_drugs"])
        val_rows = []
        allowed_rows = train_rows

    labels = load_selected_labels(args.pairs_csv, allowed_rows, len(global_data["prediction"]))
    gated_data = dict(global_data)
    gated_data["label"] = labels
    store = graph.ChemStore(gated_data, drug_data, args.condition, args.seed)

    # Fresh process + explicit reset + new model + fresh optimizer.  No checkpoint
    # argument exists for inner BRICS weights; only the integer E* JSON is read.
    set_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(graph.base.IndexDataset(train_rows), batch_size=args.batch_size,
                              shuffle=True, generator=generator, num_workers=0, collate_fn=store.collate)
    val_loader = None
    if val_rows:
        val_loader = DataLoader(graph.base.IndexDataset(val_rows), batch_size=args.batch_size,
                                shuffle=False, num_workers=0, collate_fn=store.collate)
    model = graph.BRICSChemGraphP13D(args.project, args.checkpoint).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    print("INNER_WEIGHTS_REUSED = False", flush=True)
    print("RANDOM_SEED_RESET = True", flush=True)
    print("MODEL_REINSTANTIATED = True", flush=True)
    print("FRESH_OPTIMIZER = True", flush=True)

    epoch0_loader = val_loader if val_loader is not None else DataLoader(
        graph.base.IndexDataset(train_rows), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=store.collate)
    epoch0 = graph.base.evaluate(model, epoch0_loader, args.device)
    epoch0_rows = val_rows if val_rows else train_rows
    baseline0 = store.prediction[epoch0_rows]
    max_abs0 = float(torch.max(torch.abs(epoch0["prediction"] - baseline0)))
    mse0 = metrics(epoch0["label"].numpy(), epoch0["prediction"].numpy())["mse"]

    history = []
    if args.phase == "inner":
        baseline_metrics = metrics(epoch0["label"].numpy(), baseline0.numpy())
        best_mse, best_epoch, no_improve = baseline_metrics["mse"], 0, 0
        best_state = trainable_state(model)
        for epoch in range(1, args.epochs + 1):
            training = train_epoch(model, train_loader, optimizer, args.device, args)
            validation = graph.base.evaluate(model, val_loader, args.device)
            val_metrics = metrics(validation["label"].numpy(), validation["prediction"].numpy())
            row = {"epoch": epoch, "train": training, "inner_val": val_metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_metrics["mse"] < best_mse - 1e-6:
                best_mse, best_epoch, no_improve = val_metrics["mse"], epoch, 0
                best_state = trainable_state(model)
            else:
                no_improve += 1
            if no_improve >= args.patience:
                break
        result = {
            "phase": "inner_epoch_selection", "fold": args.fold, "condition": args.condition,
            "seed": args.seed, "best_epoch": best_epoch, "best_inner_val_mse": best_mse,
            "epoch0_max_abs_pred_vs_p13d": max_abs0, "epoch0_mse": mse0,
            "outer_validation_labels_accessed": False, "inner_weights_reused_by_refit": False,
            "history": history,
        }
        filename, state = "best.pt", best_state
    else:
        for epoch in range(1, selected_epochs + 1):
            training = train_epoch(model, train_loader, optimizer, args.device, args)
            row = {"epoch": epoch, "outer_train": training}
            history.append(row)
            print(json.dumps(row), flush=True)
        result = {
            "phase": "outer_refit", "fold": args.fold, "condition": args.condition,
            "seed": args.seed, "selected_epoch": selected_epochs, "trained_epochs": selected_epochs,
            "epoch0_scope": "outer_train", "epoch0_max_abs_pred_vs_p13d": max_abs0,
            "epoch0_mse": mse0,
            "epoch0_prediction_equivalent_to_p13d": bool(max_abs0 <= 1e-5),
            "epoch0_name": ("P13D baseline" if selected_epochs == 0 and max_abs0 <= 1e-5
                            else "no-training initialized BRICS model" if selected_epochs == 0 else "not_applicable"),
            "INNER_WEIGHTS_REUSED": False, "RANDOM_SEED_RESET": True,
            "MODEL_REINSTANTIATED": True, "FRESH_OPTIMIZER": True,
            "outer_validation_labels_accessed": False, "outer_validation_metrics_computed": False,
            "source_selected_epoch_json": str(args.selected_epoch_json), "history": history,
        }
        filename, state = "final.pt", trainable_state(model)

    result["provenance"] = {
        "p13d_checkpoint": str(args.checkpoint), "p13d_checkpoint_sha256": sha256(args.checkpoint),
        "global_cache": str(args.global_cache), "global_cache_sha256": sha256(args.global_cache),
        "drug_cache": str(args.drug_cache), "drug_cache_sha256": sha256(args.drug_cache),
        "outer_split": str(args.outer_split), "outer_split_sha256": sha256(args.outer_split),
        "inner_split": str(args.inner_split), "inner_split_sha256": sha256(args.inner_split),
    }
    result["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"hierarchy_state": state, "result": result}, args.output_dir / filename)
    print(json.dumps({"saved": str(args.output_dir / filename), **{k: result[k] for k in result if k in ("best_epoch", "selected_epoch", "INNER_WEIGHTS_REUSED")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
