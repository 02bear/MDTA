#!/usr/bin/env python3
"""Extract the minimal BRICS hierarchy needed by the drug-only experiment."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(args.entity_cache, map_location="cpu", weights_only=False)
    mappings = {}
    audit = []
    for drug_id, graph in source["drugs"].items():
        atom_to_fragment = graph["atom_to_fragment"].long().cpu()
        edge_index = graph["edge_index"].long().cpu()
        edge_attr = graph["edge_attr"].float().cpu()
        n_fragments = int(atom_to_fragment.max().item()) + 1
        if edge_index.numel() and int(edge_index.max().item()) >= n_fragments:
            raise ValueError(f"{drug_id}: fragment edge index out of range")
        mappings[str(drug_id)] = {
            "atom_to_fragment": atom_to_fragment,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "n_fragments": n_fragments,
            "n_brics_cuts": int(graph["n_brics_cuts"]),
        }
        audit.append({
            "drug_id": str(drug_id),
            "atoms": int(atom_to_fragment.numel()),
            "fragments": n_fragments,
            "directed_fragment_edges": int(edge_index.shape[1]),
            "brics_cuts": int(graph["n_brics_cuts"]),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "drugs": mappings,
        "metadata": {
            "source": str(args.entity_cache),
            "description": "Explicit-H atom-to-BRICS-fragment mapping and fragment graph only",
        },
    }, args.output)
    summary = {
        "drugs": len(audit),
        "atoms_min": min(x["atoms"] for x in audit),
        "atoms_max": max(x["atoms"] for x in audit),
        "fragments_min": min(x["fragments"] for x in audit),
        "fragments_mean": sum(x["fragments"] for x in audit) / len(audit),
        "fragments_max": max(x["fragments"] for x in audit),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
