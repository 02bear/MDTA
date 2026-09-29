#!/usr/bin/env python3
"""Inspect 20 cached DTBind samples and compare labels, graph nodes and raw PDB residues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


def parse_labels(path: Path):
    lines = [line.strip() for line in path.open() if line.strip()]
    return {
        lines[i][1:].lower(): {"sequence": lines[i + 1], "labels": lines[i + 2]}
        for i in range(0, len(lines), 3)
    }


def tensor_shapes(obj):
    result = {}
    keys = obj.keys() if callable(getattr(obj, "keys", None)) else []
    for key in keys:
        value = getattr(obj, key)
        if torch.is_tensor(value):
            result[key] = list(value.shape)
        elif isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = {"type": type(value).__name__, "length": len(value)}
        else:
            result[key] = {"type": type(value).__name__}
    return result


def raw_residue_summary(pdb_path: Path):
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, pdb_path)
    model = next(structure.get_models())
    chains = {}
    total_standard = 0
    total_polymer = 0
    for chain in model:
        standard = 0
        polymer = 0
        for residue in chain:
            if is_aa(residue, standard=False):
                polymer += 1
            if is_aa(residue, standard=True):
                standard += 1
        if polymer:
            chains[chain.id] = {"polymer_aa": polymer, "standard_aa": standard}
            total_polymer += polymer
            total_standard += standard
    return {
        "chains": chains,
        "total_polymer_aa": total_polymer,
        "total_standard_aa": total_standard,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtbind-root", type=Path, required=True)
    parser.add_argument("--pdbbind-refined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    site = args.dtbind_root / "Data" / "site"
    records = parse_labels(site / "site_labels.txt")
    protein_ids = {path.stem.lower() for path in (site / "protein_graph").glob("*.pt")}
    ligand_ids = {
        path.name.lower().removesuffix("_ligand.pt")
        for path in (site / "ligand_graph").glob("*_ligand.pt")
    }
    refined_ids = {
        path.parent.name.lower()
        for path in args.pdbbind_refined.glob("*/*_protein.pdb")
    }
    eligible = sorted(set(records) & protein_ids & ligand_ids & refined_ids)[: args.limit]
    rows = []
    for pdb_id in eligible:
        protein_graph = torch.load(
            site / "protein_graph" / f"{pdb_id}.pt", map_location="cpu", weights_only=False
        )
        ligand_graph = torch.load(
            site / "ligand_graph" / f"{pdb_id}_ligand.pt",
            map_location="cpu",
            weights_only=False,
        )
        raw_pdb = args.pdbbind_refined / pdb_id / f"{pdb_id}_protein.pdb"
        record = records[pdb_id]
        rows.append(
            {
                "pdb_id": pdb_id,
                "sequence_len": len(record["sequence"]),
                "positive_count": record["labels"].count("1"),
                "protein_graph": tensor_shapes(protein_graph),
                "ligand_graph": tensor_shapes(ligand_graph),
                "raw_pdb": raw_residue_summary(raw_pdb),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
