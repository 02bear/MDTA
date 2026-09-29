#!/usr/bin/env python3
"""Cache frozen P13D drug encoder states for BRICS hierarchy screening."""

import argparse
import json
import sys
from pathlib import Path

import torch


def move_dict(data, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--mapping-cache", type=Path, required=True)
    parser.add_argument("--drug-1d-dir", type=Path, required=True)
    parser.add_argument("--drug-3d-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.project.resolve()))
    from models.model_p13d import MyModelMDTAP13D

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    model = MyModelMDTAP13D(
        drug_1d_in_dim=int(ckpt_args.get("drug_1d_in_dim", 768)),
        drug_3d_node_in_dim=int(ckpt_args.get("drug_3d_node_in_dim", 10)),
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=int(ckpt_args.get("hidden_dim", 128)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        task="regression",
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(args.device).eval()

    mapping_data = torch.load(args.mapping_cache, map_location="cpu", weights_only=False)
    global_data = torch.load(args.global_cache, map_location="cpu", weights_only=False)
    first_pair = {}
    for row, drug_id in enumerate(global_data["drug_id"]):
        first_pair.setdefault(str(drug_id), row)

    output = {}
    max_fusion_error = 0.0
    with torch.no_grad():
        for drug_id, mapping in mapping_data["drugs"].items():
            one_d = torch.load(args.drug_1d_dir / f"{drug_id}.pt", map_location="cpu", weights_only=False)["mean"].float()
            raw = torch.load(args.drug_3d_dir / f"{drug_id}.pt", map_location="cpu", weights_only=False)
            graph = {
                "x": raw["x"].float(),
                "pos": raw["pos"].float(),
                "edge_index": raw["edge_index"].long(),
                "batch": torch.zeros(raw["x"].shape[0], dtype=torch.long),
            }
            encoded = model.drug_3d_encoder(move_dict(graph, args.device), return_node=True)
            drug_1d = model.drug_1d_encoder(one_d.unsqueeze(0).to(args.device))
            drug_fused = model.drug_fusion([drug_1d, encoded["graph_feat"]])
            expected = global_data["pair_feature"][first_pair[drug_id], :128].to(args.device)
            error = float(torch.max(torch.abs(drug_fused.squeeze(0) - expected)))
            max_fusion_error = max(max_fusion_error, error)
            if encoded["node_feat"].shape[0] != mapping["atom_to_fragment"].numel():
                raise ValueError(
                    f"{drug_id}: encoder atoms={encoded['node_feat'].shape[0]} "
                    f"mapping atoms={mapping['atom_to_fragment'].numel()}"
                )
            output[drug_id] = {
                "atom_node_feat": encoded["node_feat"].cpu(),
                "drug_3d_graph_feat": encoded["graph_feat"].squeeze(0).cpu(),
                "drug_1d_feat": drug_1d.squeeze(0).cpu(),
                "drug_fused": drug_fused.squeeze(0).cpu(),
                "atom_to_fragment": mapping["atom_to_fragment"],
                "fragment_edge_index": mapping["edge_index"],
                "fragment_edge_attr": mapping["edge_attr"],
                "n_fragments": mapping["n_fragments"],
            }

    if max_fusion_error > 1e-5:
        raise RuntimeError(f"cached drug fusion mismatch: {max_fusion_error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "drugs": output,
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "max_drug_fusion_error": max_fusion_error,
            "description": "Frozen P13D drug atom/global states plus BRICS mapping",
        },
    }
    torch.save(payload, args.output)
    audit = {
        "drugs": len(output),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "max_drug_fusion_error": max_fusion_error,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
