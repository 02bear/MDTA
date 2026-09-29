#!/usr/bin/env python3
"""Verify extracted PLIP pairs against DTBind ligand/protein graph indexing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.pairs.read_text())
    rows = []
    for record in source["records"]:
        pdb_id = record["pdb_id"]
        protein = torch.load(
            args.site_dir / "protein_graph" / f"{pdb_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        ligand = torch.load(
            args.site_dir / "ligand_graph" / f"{pdb_id}_ligand.pt",
            map_location="cpu",
            weights_only=False,
        )
        residue_indices = {pair["residue_index"] for pair in record["pairs"]}
        residue_positive = {
            index for index, value in enumerate(protein.y.reshape(-1).tolist()) if value > 0
        }
        rows.append(
            {
                "pdb_id": pdb_id,
                "atom_count_plip": record["atom_count"],
                "atom_count_graph": int(ligand.x.shape[0]),
                "residue_count_plip": record["residue_count"],
                "residue_count_graph": int(protein.x.shape[0]),
                "pair_count": len(record["pairs"]),
                "pair_residue_count": len(residue_indices),
                "dtbind_positive_count": len(residue_positive),
                "pair_residue_subset_of_dtbind": residue_indices <= residue_positive,
                "pair_residue_overlap": len(residue_indices & residue_positive),
            }
        )
    summary = {
        "sample_count": len(rows),
        "atom_count_all_match": all(
            row["atom_count_plip"] == row["atom_count_graph"] for row in rows
        ),
        "residue_count_all_match": all(
            row["residue_count_plip"] == row["residue_count_graph"] for row in rows
        ),
        "pair_residue_subset_all": all(
            row["pair_residue_subset_of_dtbind"] for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
