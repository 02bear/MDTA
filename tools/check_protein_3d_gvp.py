# -*- coding: utf-8 -*-
import json
import math
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


REQUIRED_KEYS = [
    "node_s",
    "node_v",
    "coords",
    "edge_index",
    "edge_s",
    "edge_v",
]


def is_float_tensor(x: torch.Tensor) -> bool:
    return x.dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    }


def check_one_file(pt_path: Path):
    errors = []
    example_shapes = {}

    try:
        obj = torch.load(pt_path, weights_only=False)
    except Exception as e:
        return {
            "ok": False,
            "errors": [f"load_error: {e}"],
            "example_shapes": {},
        }

    # 1) required keys
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        errors.append(f"missing_keys: {missing}")
        return {
            "ok": False,
            "errors": errors,
            "example_shapes": {},
        }

    node_s = obj["node_s"]
    node_v = obj["node_v"]
    coords = obj["coords"]
    edge_index = obj["edge_index"]
    edge_s = obj["edge_s"]
    edge_v = obj["edge_v"]

    # tensor type check
    for name, x in [
        ("node_s", node_s),
        ("node_v", node_v),
        ("coords", coords),
        ("edge_index", edge_index),
        ("edge_s", edge_s),
        ("edge_v", edge_v),
    ]:
        if not isinstance(x, torch.Tensor):
            errors.append(f"{name} is not torch.Tensor, got {type(x)}")

    if errors:
        return {
            "ok": False,
            "errors": errors,
            "example_shapes": {},
        }

    # collect shapes
    example_shapes = {
        "node_s": list(node_s.shape),
        "node_v": list(node_v.shape),
        "coords": list(coords.shape),
        "edge_index": list(edge_index.shape),
        "edge_s": list(edge_s.shape),
        "edge_v": list(edge_v.shape),
    }

    # 2) shape checks
    if node_s.ndim != 2 or node_s.shape[1] != 6:
        errors.append(f"node_s shape invalid: expected [L,6], got {tuple(node_s.shape)}")

    if node_v.ndim != 3 or node_v.shape[1] != 3 or node_v.shape[2] != 3:
        errors.append(f"node_v shape invalid: expected [L,3,3], got {tuple(node_v.shape)}")

    if coords.ndim != 2 or coords.shape[1] != 3:
        errors.append(f"coords shape invalid: expected [L,3], got {tuple(coords.shape)}")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        errors.append(f"edge_index shape invalid: expected [2,E], got {tuple(edge_index.shape)}")

    if edge_s.ndim != 2 or edge_s.shape[1] != 32:
        errors.append(f"edge_s shape invalid: expected [E,32], got {tuple(edge_s.shape)}")

    if edge_v.ndim != 3 or edge_v.shape[1] != 1 or edge_v.shape[2] != 3:
        errors.append(f"edge_v shape invalid: expected [E,1,3], got {tuple(edge_v.shape)}")

    # check L / E consistency when possible
    if node_s.ndim == 2 and node_v.ndim == 3 and coords.ndim == 2:
        L = node_s.shape[0]
        if node_v.shape[0] != L:
            errors.append(f"L mismatch: node_v.shape[0]={node_v.shape[0]} vs node_s.shape[0]={L}")
        if coords.shape[0] != L:
            errors.append(f"L mismatch: coords.shape[0]={coords.shape[0]} vs node_s.shape[0]={L}")
    else:
        L = None

    if edge_index.ndim == 2 and edge_s.ndim == 2 and edge_v.ndim == 3:
        E = edge_index.shape[1]
        if edge_s.shape[0] != E:
            errors.append(f"E mismatch: edge_s.shape[0]={edge_s.shape[0]} vs edge_index.shape[1]={E}")
        if edge_v.shape[0] != E:
            errors.append(f"E mismatch: edge_v.shape[0]={edge_v.shape[0]} vs edge_index.shape[1]={E}")

    # 3) dtype checks
    if not is_float_tensor(node_s):
        errors.append(f"node_s dtype invalid: expected float, got {node_s.dtype}")
    if not is_float_tensor(node_v):
        errors.append(f"node_v dtype invalid: expected float, got {node_v.dtype}")
    if not is_float_tensor(coords):
        errors.append(f"coords dtype invalid: expected float, got {coords.dtype}")
    if edge_index.dtype != torch.int64:
        errors.append(f"edge_index dtype invalid: expected torch.int64/long, got {edge_index.dtype}")
    if not is_float_tensor(edge_s):
        errors.append(f"edge_s dtype invalid: expected float, got {edge_s.dtype}")
    if not is_float_tensor(edge_v):
        errors.append(f"edge_v dtype invalid: expected float, got {edge_v.dtype}")

    # 4) edge index range checks
    if L is not None and edge_index.numel() > 0:
        try:
            ei_min = int(edge_index.min().item())
            ei_max = int(edge_index.max().item())
            if ei_min < 0:
                errors.append(f"edge_index min invalid: {ei_min} < 0")
            if ei_max >= L:
                errors.append(f"edge_index max invalid: {ei_max} >= L({L})")
        except Exception as e:
            errors.append(f"edge_index range check failed: {e}")

    # 5) NaN / Inf checks
    float_tensors = {
        "node_s": node_s,
        "node_v": node_v,
        "coords": coords,
        "edge_s": edge_s,
        "edge_v": edge_v,
    }
    for name, t in float_tensors.items():
        if not is_float_tensor(t):
            continue
        if not torch.isfinite(t).all().item():
            num_nan = int(torch.isnan(t).sum().item())
            num_inf = int(torch.isinf(t).sum().item())
            errors.append(f"{name} has non-finite values: nan={num_nan}, inf={num_inf}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "example_shapes": example_shapes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protein_3d_dir",
        type=str,
        default="data/processed/davis/protein_3d_gvp",
    )
    args = parser.parse_args()

    protein_3d_dir = Path(args.protein_3d_dir)
    if not protein_3d_dir.exists() or not protein_3d_dir.is_dir():
        raise FileNotFoundError(f"protein_3d_dir not found or not a directory: {protein_3d_dir}")

    pt_files = sorted(protein_3d_dir.glob("*.pt"))
    num_files = len(pt_files)

    failures = []
    num_ok = 0
    example_shapes = None

    for p in tqdm(pt_files, total=num_files, desc="Checking protein_3d_gvp .pt files"):
        ret = check_one_file(p)
        if ret["ok"]:
            num_ok += 1
            if example_shapes is None:
                example_shapes = ret["example_shapes"]
        else:
            failures.append(
                {
                    "file": str(p),
                    "errors": " | ".join(ret["errors"]),
                    "shapes": json.dumps(ret["example_shapes"], ensure_ascii=False),
                }
            )

    num_failed = len(failures)
    failure_csv_path = protein_3d_dir / "check_failures.csv"
    pd.DataFrame(failures).to_csv(failure_csv_path, index=False)

    summary = {
        "num_files": int(num_files),
        "num_ok": int(num_ok),
        "num_failed": int(num_failed),
        "failure_csv_path": str(failure_csv_path),
        "example_shapes": example_shapes,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # non-zero exit when any failure exists
    if num_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
