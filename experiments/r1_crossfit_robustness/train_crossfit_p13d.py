#!/usr/bin/env python3
"""Locked two-stage P13D training for one N3 outer-fold/cross-fit run."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from n3_common import (
    BATCH_SIZE,
    LR,
    MAX_EPOCHS,
    MIN_DELTA,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
    SelectiveLabelP13DDataset,
    build_model,
    build_n3_split,
    collate_with_index,
    load_selected_labels,
    locked_training_config,
    read_pair_entities,
    rows_for_drugs,
    set_seed,
    sha256_file,
    write_json,
)


def loader(dataset, rows, shuffle):
    return DataLoader(
        Subset(dataset, rows), batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=0, collate_fn=collate_with_index, pin_memory=True,
    )


def move(batch, device):
    from datasets.collate_p13d import move_batch_to_device
    return move_batch_to_device(batch, device)


def simple_metrics(prediction, label):
    prediction = prediction.view(-1).detach().cpu()
    label = label.view(-1).detach().cpu()
    mse = float(torch.mean((prediction - label) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(torch.mean(torch.abs(prediction-label)))}


def train_epoch(model, data_loader, optimizer, device):
    model.train()
    preds, labels, total, count = [], [], 0.0, 0
    for batch in data_loader:
        batch = move(batch, device)
        if not torch.isfinite(batch["label"]).all():
            raise RuntimeError("non-finite training label")
        optimizer.zero_grad()
        prediction = model(batch)
        loss = torch.nn.functional.mse_loss(prediction, batch["label"])
        loss.backward()
        optimizer.step()
        n = int(batch["label"].shape[0])
        total += float(loss.detach()) * n
        count += n
        preds.append(prediction.detach().cpu())
        labels.append(batch["label"].detach().cpu())
    metrics = simple_metrics(torch.cat(preds), torch.cat(labels))
    metrics["loss"] = total / count
    return metrics


@torch.no_grad()
def evaluate(model, data_loader, device, require_labels=True):
    model.eval()
    indices, preds, labels = [], [], []
    for batch in data_loader:
        indices.append(batch["index"].clone())
        batch = move(batch, device)
        if require_labels and not torch.isfinite(batch["label"]).all():
            raise RuntimeError("non-finite evaluation label")
        prediction = model(batch)
        preds.append(prediction.detach().cpu())
        labels.append(batch["label"].detach().cpu())
    result = {"index": torch.cat(indices), "prediction": torch.cat(preds), "label": torch.cat(labels)}
    if require_labels:
        result["metrics"] = simple_metrics(result["prediction"], result["label"])
    return result


def save_training_checkpoint(path, model, optimizer, epoch, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata,
    }, path)


def validate_materialized_split(args, expected):
    path = args.output_root / "audit/splits" / f"fold_{args.fold}" / f"cf_{args.cf}.json"
    observed = json.loads(path.read_text(encoding="utf-8"))
    for key in ("outer_train_drugs", "holdout_drugs", "T_j", "epoch_train_drugs", "epoch_val_drugs"):
        if observed[key] != expected[key]:
            raise RuntimeError(f"materialized split mismatch for {key}")
    return path


def assert_reproduction_gate(args):
    path = args.output_root / "audit/historical_reproduction.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("all_passed") is not True:
        raise RuntimeError("historical reproduction gate not passed")
    for fold in range(1, 6):
        if result["folds"][str(fold)].get("passed") is not True:
            raise RuntimeError(f"historical reproduction fold {fold} failed")
    return path


def stage_a(args, split, split_path, reproduction_path, entities, device):
    output = args.output_root / f"fold_{args.fold}/cf_{args.cf}/stage_a_epoch_selection"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_model.pt"
    result_path = output / "result.json"
    et_rows = rows_for_drugs(entities, split["epoch_train_drugs"])
    ev_rows = rows_for_drugs(entities, split["epoch_val_drugs"])
    allowed = et_rows + ev_rows
    labels = load_selected_labels(args.pairs_csv, allowed, len(entities))
    dataset = SelectiveLabelP13DDataset(args.project, entities, labels)
    train_loader = loader(dataset, et_rows, True)
    val_loader = loader(dataset, ev_rows, False)

    set_seed(SEED)
    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    epoch0 = evaluate(model, val_loader, device)["metrics"]
    best_epoch, best_rmse, no_improve = 0, epoch0["rmse"], 0
    metadata = {
        "phase": "stage_a_epoch_selection", "outer_fold": args.fold, "cf_fold": args.cf,
        "seed": SEED, "epoch": 0, "validation": epoch0,
        "HOLDOUT_USED_FOR_TRAINING": False,
        "HOLDOUT_USED_FOR_EPOCH_SELECTION": False,
    }
    save_training_checkpoint(checkpoint, model, optimizer, 0, metadata)
    history = [{"epoch": 0, "validation": epoch0, "name": "NO_TRAINING_INITIALIZED_MODEL"}]
    print(json.dumps(history[-1]), flush=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        training = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, val_loader, device)["metrics"]
        row = {"epoch": epoch, "training": training, "validation": validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["rmse"] < best_rmse - MIN_DELTA:
            best_epoch, best_rmse, no_improve = epoch, validation["rmse"], 0
            save_training_checkpoint(checkpoint, model, optimizer, epoch, {
                **metadata, "epoch": epoch, "training": training, "validation": validation,
            })
        else:
            no_improve += 1
        write_json(output / "progress.json", {
            "last_epoch": epoch, "best_epoch": best_epoch, "best_val_rmse": best_rmse,
            "epochs_no_improve": no_improve,
        })
        if no_improve >= PATIENCE:
            break

    result = {
        "phase": "stage_a_epoch_selection", "outer_fold": args.fold, "cf_fold": args.cf,
        "seed": SEED, "best_epoch": best_epoch, "E_star": best_epoch,
        "best_val_rmse": best_rmse, "epoch0_metrics": epoch0,
        "epoch0_name": "NO_TRAINING_INITIALIZED_MODEL",
        "epochs_completed": history[-1]["epoch"], "history": history,
        "training_config": locked_training_config(), "split": split,
        "HOLDOUT_USED_FOR_TRAINING": False,
        "HOLDOUT_USED_FOR_EPOCH_SELECTION": False,
        "outer_validation_labels_accessed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "materialized_split": str(split_path),
        "reproduction_gate": str(reproduction_path),
    }
    write_json(result_path, result)
    print(f"STAGE_A_COMPLETE fold={args.fold} cf={args.cf} E*={best_epoch}", flush=True)


def stage_b(args, split, split_path, reproduction_path, entities, device):
    stage_a_dir = args.output_root / f"fold_{args.fold}/cf_{args.cf}/stage_a_epoch_selection"
    selection = json.loads((stage_a_dir / "result.json").read_text(encoding="utf-8"))
    selected_epoch = int(selection["E_star"])
    if not 0 <= selected_epoch <= MAX_EPOCHS:
        raise RuntimeError(f"invalid E*: {selected_epoch}")

    output = args.output_root / f"fold_{args.fold}/cf_{args.cf}/stage_b_strict_refit"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "final_model.pt"
    result_path = output / "result.json"
    t_rows = rows_for_drugs(entities, split["T_j"])
    h_rows = rows_for_drugs(entities, split["holdout_drugs"])
    if set(t_rows) & set(h_rows):
        raise RuntimeError("holdout rows overlap refit rows")
    labels = load_selected_labels(args.pairs_csv, t_rows, len(entities))
    dataset = SelectiveLabelP13DDataset(args.project, entities, labels)

    # This is a fresh process invocation. Only the integer E* above is read from
    # Stage A; no Stage-A state_dict or optimizer state is loaded.
    del selection
    gc.collect()
    set_seed(SEED)
    train_loader = loader(dataset, t_rows, True)
    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print("INNER_WEIGHTS_REUSED = False", flush=True)
    print("RANDOM_SEED_RESET = True", flush=True)
    print("MODEL_REINSTANTIATED = True", flush=True)
    print("FRESH_OPTIMIZER = True", flush=True)
    print("HOLDOUT_USED_FOR_TRAINING = False", flush=True)
    print("HOLDOUT_USED_FOR_EPOCH_SELECTION = False", flush=True)
    history = []
    for epoch in range(1, selected_epoch + 1):
        training = train_epoch(model, train_loader, optimizer, device)
        row = {"epoch": epoch, "training": training}
        history.append(row)
        print(json.dumps(row), flush=True)

    checkpoint_metadata = {
        "phase": "stage_b_strict_refit", "outer_fold": args.fold, "cf_fold": args.cf,
        "seed": SEED, "selected_epoch": selected_epoch, "trained_epochs": selected_epoch,
        "INNER_WEIGHTS_REUSED": False, "RANDOM_SEED_RESET": True,
        "MODEL_REINSTANTIATED": True, "FRESH_OPTIMIZER": True,
        "HOLDOUT_USED_FOR_TRAINING": False,
        "HOLDOUT_USED_FOR_EPOCH_SELECTION": False,
        "outer_validation_labels_accessed": False,
    }
    save_training_checkpoint(checkpoint, model, optimizer, selected_epoch, checkpoint_metadata)
    frozen_sha = sha256_file(checkpoint)

    # Holdout samples carry NaN labels here. The model is frozen before this
    # loader is entered, so no holdout label has been parsed or used yet.
    holdout_loader = loader(dataset, h_rows, False)
    observed = evaluate(model, holdout_loader, device, require_labels=False)
    predictions = observed["prediction"].view(-1).numpy().astype(np.float64)
    observed_rows = observed["index"].numpy().astype(int)
    if not np.array_equal(observed_rows, np.asarray(h_rows, dtype=int)):
        raise RuntimeError("holdout inference row order mismatch")
    if not np.isfinite(predictions).all():
        raise RuntimeError("strict OOF predictions contain NaN/Inf")

    # Only after checkpoint freezing and label-free OOF inference may H labels
    # be parsed to form residuals.
    h_labels_all = load_selected_labels(args.pairs_csv, h_rows, len(entities))
    h_labels = h_labels_all[h_rows].numpy().astype(np.float64)
    cache_path = args.project / f"experiments/pdbbind_to_davis_transfer/data/global_predictions/fold{args.fold}.pt"
    historical = torch.load(cache_path, map_location="cpu", weights_only=False)
    for i, row in enumerate(h_rows):
        if str(historical["drug_id"][row]) != entities[row][0] or str(historical["protein_id"][row]) != entities[row][1]:
            raise RuntimeError("historical cache row mapping mismatch")
    historical_pred = historical["prediction"].numpy()[h_rows].astype(np.float64)
    historical_labels = historical["label"].numpy()[h_rows].astype(np.float64)
    if np.max(np.abs(historical_labels - h_labels)) > 1e-6:
        raise RuntimeError("selected holdout labels differ from historical cache")

    rows = []
    for local, global_row in enumerate(h_rows):
        drug_id, protein_id = entities[global_row]
        rows.append({
            "outer_fold": args.fold, "cf_fold": args.cf, "row_index": global_row,
            "drug_id": drug_id, "protein_id": protein_id, "label": h_labels[local],
            "historical_insample_pred": historical_pred[local],
            "strict_oof_pred": predictions[local],
            "residual_insample": h_labels[local] - historical_pred[local],
            "residual_oof": h_labels[local] - predictions[local],
        })
    oof_path = output / "oof_predictions.csv"
    pd.DataFrame(rows).to_csv(oof_path, index=False)
    drug_counts = pd.Series([row["drug_id"] for row in rows]).value_counts()
    if len(rows) != len(split["holdout_drugs"]) * 442 or not (drug_counts == 442).all():
        raise RuntimeError("holdout completeness failure")

    epoch0_audit = None
    if selected_epoch == 0:
        max_abs = float(np.max(np.abs(predictions - historical_pred)))
        epoch0_mse = float(np.mean((predictions - h_labels) ** 2))
        epoch0_audit = {
            "max_abs_pred_epoch0_vs_historical_P13D": max_abs,
            "MSE_epoch0": epoch0_mse,
            "prediction_equivalent": bool(max_abs <= 1e-5),
            "name": "P13D baseline" if max_abs <= 1e-5 else "NO_TRAINING_INITIALIZED_MODEL",
        }
    result = {
        **checkpoint_metadata,
        "E_star": selected_epoch,
        "history": history,
        "training_config": locked_training_config(), "split": split,
        "strict_oof_pair_count": len(rows), "holdout_drug_count": len(drug_counts),
        "holdout_labels_parsed_before_checkpoint_freeze": False,
        "holdout_labels_parsed_before_final_oof_inference": False,
        "checkpoint": str(checkpoint), "checkpoint_sha256": frozen_sha,
        "oof_predictions": str(oof_path), "epoch0_audit": epoch0_audit,
        "materialized_split": str(split_path), "reproduction_gate": str(reproduction_path),
    }
    write_json(result_path, result)
    print(f"STAGE_B_COMPLETE fold={args.fold} cf={args.cf} E*={selected_epoch} pairs={len(rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--cf", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--phase", choices=["stage-a", "stage-b"], required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("formal N3 training requires an explicit CUDA device")
    split = build_n3_split(args.project, args.fold, args.cf)
    split_path = validate_materialized_split(args, split)
    reproduction_path = assert_reproduction_gate(args)
    entities = read_pair_entities(args.pairs_csv)
    if args.phase == "stage-a":
        stage_a(args, split, split_path, reproduction_path, entities, args.device)
    else:
        stage_b(args, split, split_path, reproduction_path, entities, args.device)


if __name__ == "__main__":
    main()
