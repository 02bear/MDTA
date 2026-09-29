#!/usr/bin/env python3
"""Extract compact KLIFS-85 sequence and geometry tensors for training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--klifs-dir", type=Path, required=True)
    parser.add_argument("--prott5-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, klifs_dir = args.project_root.resolve(), args.klifs_dir.resolve()
    audit = pd.read_csv(klifs_dir / "mapping_audit.csv", dtype={"protein_id": str}).set_index("protein_id")
    accepted = {"ok", "ok_sequence_fallback"}
    manifest = json.loads((args.prott5_dir / "manifest.json").read_text())["manifest"]
    protein_ids = pd.read_csv(root / "data/raw/davis/proteins.csv", dtype=str)["protein_id"].tolist()
    output = {}
    for protein_id in protein_ids:
        status = audit.loc[protein_id, "status"]
        if status not in accepted:
            output[protein_id] = {
                "mask": torch.zeros(85, dtype=torch.bool),
                "sequence": torch.zeros((85, 1024), dtype=torch.float16),
                "geometry": torch.zeros((85, 18), dtype=torch.float32),
                "mapping_status": status,
            }
            continue
        mapping = torch.load(klifs_dir / "by_protein" / f"{protein_id}.pt", map_location="cpu", weights_only=False)
        indices, mask = mapping["sequence_indices"].long(), mapping["mask"].bool()
        prott5_path = root / manifest[protein_id]["cache"]
        prott5 = torch.load(prott5_path, map_location="cpu", weights_only=False)["per_tok"]
        gvp = torch.load(root / "data/processed/davis/protein_3d_gvp" / f"{protein_id}.pt", map_location="cpu", weights_only=False)
        valid = mask & (indices >= 0) & (indices < len(prott5)) & (indices < len(gvp["node_s"]))
        sequence = torch.zeros((85, 1024), dtype=torch.float16)
        sequence[valid] = prott5[indices[valid]].to(torch.float16)
        coords = torch.zeros((85, 3), dtype=torch.float32)
        coords[valid] = gvp["coords"][indices[valid]].float()
        if valid.any():
            coords[valid] -= coords[valid].mean(0, keepdim=True)
        node_s = torch.zeros((85, 6), dtype=torch.float32)
        node_v = torch.zeros((85, 9), dtype=torch.float32)
        node_s[valid] = gvp["node_s"][indices[valid]].float()
        node_v[valid] = gvp["node_v"][indices[valid]].float().reshape(-1, 9)
        geometry = torch.cat([node_s, node_v, coords / 10.0], dim=-1)
        output[protein_id] = {
            "mask": valid,
            "sequence": sequence,
            "geometry": geometry,
            "coords": coords,
            "mapping_status": status,
            "mapping_method": mapping.get("mapping_method", "structure_chain"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"proteins": output, "accepted_statuses": sorted(accepted)}, args.output)
    print({
        "proteins": len(output),
        "accepted": sum(bool(item["mask"].any()) for item in output.values()),
        "all_shapes_valid": all(item["sequence"].shape == (85, 1024) for item in output.values()),
    })


if __name__ == "__main__":
    main()
