#!/usr/bin/env python3
"""Cache the exact inputs to layer 3 of the frozen P13D drug EGNN."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch_geometric.nn import global_mean_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--drug-cache", type=Path, required=True)
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
    encoder = model.drug_3d_encoder

    drug_cache = torch.load(args.drug_cache, map_location="cpu", weights_only=False)
    output = {}
    max_graph_error = 0.0
    with torch.no_grad():
        for drug_id in sorted(drug_cache["drugs"]):
            raw = torch.load(args.drug_3d_dir / f"{drug_id}.pt", map_location="cpu", weights_only=False)
            h = encoder.input_proj(raw["x"].float().to(args.device))
            pos = raw["pos"].float().to(args.device)
            edge_index = raw["edge_index"].long().to(args.device)
            for layer in encoder.layers[:2]:
                h, pos = layer(h, pos, edge_index)

            h3, _ = encoder.layers[2](h, pos, edge_index)
            batch = torch.zeros(h3.shape[0], dtype=torch.long, device=args.device)
            graph = encoder.out_proj(global_mean_pool(h3, batch)).squeeze(0)
            expected = drug_cache["drugs"][drug_id]["drug_3d_graph_feat"].to(args.device)
            max_graph_error = max(max_graph_error, float(torch.max(torch.abs(graph - expected))))
            output[drug_id] = {
                "h": h.cpu(),
                "pos": pos.cpu(),
                "edge_index": edge_index.cpu(),
            }

    if max_graph_error > 1e-5:
        raise RuntimeError(f"layer-2 cache does not reproduce baseline graph embeddings: {max_graph_error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "drugs": output,
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "drugs": len(output),
            "max_graph_error": max_graph_error,
            "description": "Unlabeled structural cache after input_proj and EGNN layers 1-2",
        },
    }
    torch.save(payload, args.output)
    audit = {**payload["metadata"], "output": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

