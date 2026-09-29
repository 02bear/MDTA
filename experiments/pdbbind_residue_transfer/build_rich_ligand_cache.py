#!/usr/bin/env python3
"""Build an atom-index-audited rich ligand cache without touching DTBind graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rich_ligand_features import ELEMENTS, load_heavy_molecule, molecule_to_graph


def cached_elements(graph):
    indices = graph.x[:, : len(ELEMENTS)].argmax(dim=1).tolist()
    return [ELEMENTS[index] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtbind-root", type=Path, required=True)
    parser.add_argument("--pdbbind-refined", type=Path, required=True)
    parser.add_argument("--pair-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    site = args.dtbind_root / "Data" / "site"
    all_records = json.loads(args.pair_labels.read_text())["records"]
    records = all_records[: args.limit] if args.limit else all_records
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_records": len(all_records),
        "requested": len(records),
        "written": 0,
        "already_present": 0,
        "failures": [],
    }
    aromatic_atoms = charged_atoms = donors = acceptors = ring_atoms = 0

    for record in records:
        pdb_id = record["pdb_id"]
        output = args.output_dir / f"{pdb_id}_ligand_rich.pt"
        try:
            old_graph = torch.load(
                site / "ligand_graph" / f"{pdb_id}_ligand.pt",
                map_location="cpu",
                weights_only=False,
            )
            molecule = load_heavy_molecule(
                args.pdbbind_refined / pdb_id / f"{pdb_id}_ligand.sdf"
            )
            molecule_elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            old_elements = cached_elements(old_graph)
            if molecule.GetNumAtoms() != record["atom_count"]:
                raise ValueError(
                    f"label/SDF atom count mismatch {record['atom_count']} != {molecule.GetNumAtoms()}"
                )
            if old_elements != molecule_elements:
                mismatch = next(
                    index for index, (old, new) in enumerate(zip(old_elements, molecule_elements))
                    if old != new
                )
                raise ValueError(
                    f"atom order mismatch at {mismatch}: cached={old_elements[mismatch]} sdf={molecule_elements[mismatch]}"
                )
            if output.exists():
                graph = torch.load(output, map_location="cpu", weights_only=False)
                summary["already_present"] += 1
            else:
                graph = molecule_to_graph(molecule)
                torch.save(graph, output)
                summary["written"] += 1
            aromatic_atoms += int(graph.x[:, 81].sum())
            charged_atoms += int((graph.x[:, 84] < 0.5).sum())
            donors += int(graph.x[:, 94].sum())
            acceptors += int(graph.x[:, 95].sum())
            ring_atoms += int(graph.x[:, 96].sum())
        except Exception as error:
            summary["failures"].append({"pdb_id": pdb_id, "error": repr(error)})

    summary.update({
        "success": summary["written"] + summary["already_present"],
        "failure_count": len(summary["failures"]),
        "feature_totals": {
            "aromatic_atoms": aromatic_atoms,
            "charged_atoms": charged_atoms,
            "donor_atoms": donors,
            "acceptor_atoms": acceptors,
            "ring_atoms": ring_atoms,
        },
    })
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
