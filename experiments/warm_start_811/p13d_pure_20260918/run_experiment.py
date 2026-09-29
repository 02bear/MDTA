"""Leakage-safe pure P13D baseline on the fixed DAVIS 8:1:1 split."""

import fcntl
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
PROJECT = Path("/data1/ztx/MyModel-MDTA")
SOURCE = HERE / "source"
OUT = PROJECT / "outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/p13d_pure_20260918"
SPLIT = PROJECT / "data/splits/davis_fixed_split_811_full.json"

sys.path.insert(0, str(SOURCE))
import train_p13d_earlystop as base  # noqa: E402


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def arguments():
    return SimpleNamespace(
        pairs_csv="data/raw/davis/pairs.csv",
        drug_1d_dir="data/processed/davis/drug_1d_chemberta2",
        drug_2d_dir="data/processed/davis/drug_2d",
        drug_3d_dir="data/processed/davis/drug_3d",
        protein_1d_dir="data/processed/davis/protein_1d_esm2",
        protein_3d_dir="data/processed/davis/protein_3d_gvp",
        split_json=str(SPLIT),
        output_dir=str(OUT),
        seed=42,
        train_ratio=0.8,
        batch_size=16,
        num_workers=0,
        epochs=500,
        lr=3e-4,
        weight_decay=1e-5,
        early_stop_patience=60,
        early_stop_min_delta=1e-4,
        drug_1d_in_dim=768,
        drug_3d_node_in_dim=10,
        hidden_dim=128,
        dropout=0.1,
    )


def checkpoint(model, optimizer, epoch, history, best_rmse, best_epoch,
               stale, best_train, best_val, args):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_rmse": best_rmse,
        "best_epoch": best_epoch,
        "stale": stale,
        "best_train": best_train,
        "best_val": best_val,
        "rng": rng_state(),
        "args": vars(args),
        "protocol": {
            "model": "pure_p13d_original_mlp_head",
            "split": str(SPLIT),
            "test_policy": "sealed_until_best_validation_checkpoint_is_fixed",
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock_handle = (OUT / "worker.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ANOTHER_WORKER_IS_ACTIVE", flush=True)
        return

    args = arguments()
    seed_all(args.seed)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    device = torch.device("cuda")

    dataset, train_set, val_set, train_loader, val_loader = base.build_dataloaders(args)
    test_set = Subset(dataset, split["test_indices"])
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=base.mdta_collate_fn_p13d,
        pin_memory=True,
    )
    assert len(dataset) == 30056
    assert len(train_set) == 24044 and len(val_set) == 3005 and len(test_set) == 3007
    assert list(train_set.indices) == split["train_indices"]
    assert list(val_set.indices) == split["val_indices"]
    assert not set(train_set.indices) & set(val_set.indices)
    assert not set(train_set.indices) & set(test_set.indices)
    assert not set(val_set.indices) & set(test_set.indices)

    model = base.build_model(args, device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_rmse = float("inf")
    best_epoch = -1
    stale = 0
    best_train = None
    best_val = None
    start_epoch = 1
    latest_path = OUT / "latest_model.pt"
    best_path = OUT / "best_model.pt"

    if (OUT / "result.json").exists():
        print("RESULT_ALREADY_COMPLETE", flush=True)
        return
    if latest_path.exists():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        best_rmse = state["best_val_rmse"]
        best_epoch = state["best_epoch"]
        stale = state["stale"]
        best_train = state["best_train"]
        best_val = state["best_val"]
        start_epoch = state["epoch"] + 1
        restore_rng(state["rng"])
        print(f"RESUME_FROM_EPOCH={start_epoch}", flush=True)

    preflight = {
        "model": "pure_p13d_original_mlp_head",
        "train": len(train_set),
        "val": len(val_set),
        "test_sealed": len(test_set),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.early_stop_patience,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    dump_json(OUT / "preflight.json", preflight)
    dump_json(OUT / "status.json", {"state": "training", "start_epoch": start_epoch, **preflight})
    print("P13D_811_START " + json.dumps(preflight), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.early_stop_patience:
            break
        began = time.time()
        train_metrics = base.train_one_epoch(
            model, train_loader, criterion, optimizer, device, log_interval=200
        )
        val_metrics = base.evaluate(model, val_loader, criterion, device)
        improved = val_metrics["rmse"] < best_rmse - args.early_stop_min_delta
        if improved:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            stale = 0
            best_train = dict(train_metrics)
            best_val = dict(val_metrics)
        else:
            stale += 1

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "best_epoch": best_epoch,
            "stale": stale,
            "seconds": time.time() - began,
        }
        history.append(record)
        state = checkpoint(
            model, optimizer, epoch, history, best_rmse, best_epoch,
            stale, best_train, best_val, args
        )
        if improved:
            save_torch(best_path, state)
        save_torch(latest_path, state)
        dump_json(OUT / "history.json", history)
        dump_json(OUT / "status.json", {"state": "training", **record, "updated": time.time()})
        print("P13D_811_EPOCH " + json.dumps(record), flush=True)

    if best_epoch < 1 or not best_path.exists():
        raise RuntimeError("No valid best checkpoint was produced")

    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model_state_dict"], strict=True)
    test_metrics = base.evaluate(model, test_loader, criterion, device)
    result = {
        "complete": True,
        "model": "pure_p13d_original_mlp_head",
        "best_epoch": best_epoch,
        "last_epoch": history[-1]["epoch"],
        "best_train_metrics": best_train,
        "best_val_metrics": best_val,
        "test_metrics": test_metrics,
        "test_policy": "test evaluated once after validation selection",
        "finished": time.time(),
    }
    dump_json(OUT / "result.json", result)
    dump_json(OUT / "status.json", {"state": "complete", **result})
    print("P13D_811_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
