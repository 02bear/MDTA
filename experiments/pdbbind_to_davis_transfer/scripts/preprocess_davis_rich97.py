#!/usr/bin/env python3
"""Build Davis ligand graphs with the exact PDBbind 97d/6d featurizer."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from rdkit import Chem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    source_dir = args.project / "experiments/pdbbind_residue_transfer"
    sys.path.insert(0, str(source_dir))
    from rich_ligand_features import ATOM_DIM, EDGE_DIM, load_heavy_molecule, molecule_to_graph

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v2_dir = args.project / "data/processed/davis/drug_atom_features_v2"
    sdf_dir = args.project / "data/raw/davis/pubchem_sdf"
    rows, failures = [], []
    with (args.project / "data/raw/davis/drugs.csv").open(newline="") as handle:
        drugs = list(csv.DictReader(handle))

    for item in drugs:
        drug_id = str(item["drug_id"])
        try:
            mol = load_heavy_molecule(sdf_dir / f"{drug_id}.sdf")
            graph = molecule_to_graph(mol)
            conf = mol.GetConformer()
            graph.pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
            graph.atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            graph.drug_id = drug_id
            graph.source_sdf = str(sdf_dir / f"{drug_id}.sdf")

            v2 = torch.load(v2_dir / f"{drug_id}.pt", map_location="cpu", weights_only=False)
            v2_count = int(v2["num_atoms"])
            coord_match = graph.pos.shape == v2["structure_pos"].shape
            coord_max_abs = (
                float((graph.pos - v2["structure_pos"].float()).abs().max())
                if coord_match else None
            )
            smiles_mol = Chem.MolFromSmiles(item["smiles"])
            smiles_heavy = smiles_mol.GetNumHeavyAtoms() if smiles_mol is not None else None
            row = {
                "drug_id": drug_id,
                "atoms": int(graph.x.shape[0]),
                "directed_edges": int(graph.edge_index.shape[1]),
                "atom_dim": int(graph.x.shape[1]),
                "edge_dim": int(graph.edge_attr.shape[1]),
                "v2_atom_count": v2_count,
                "v2_count_match": int(graph.x.shape[0]) == v2_count,
                "v2_coordinate_shape_match": coord_match,
                "v2_coordinate_max_abs_difference": coord_max_abs,
                "csv_smiles_heavy_atoms": smiles_heavy,
                "csv_smiles_count_match": smiles_heavy == int(graph.x.shape[0]),
            }
            if graph.x.shape[1] != ATOM_DIM or graph.edge_attr.shape[1] != EDGE_DIM:
                raise AssertionError(f"feature dimensions differ: {graph.x.shape}, {graph.edge_attr.shape}")
            torch.save(graph, args.output_dir / f"{drug_id}.pt")
            rows.append(row)
        except Exception as exc:
            failures.append({"drug_id": drug_id, "error": str(exc)})

    report = {
        "purpose": "Feature-compatible Davis ligand input for the frozen PDBbind atom encoder.",
        "input": str(sdf_dir),
        "source_featurizer": str(source_dir / "rich_ligand_features.py"),
        "output": str(args.output_dir),
        "requested": len(drugs),
        "success": len(rows),
        "failed": len(failures),
        "atom_dim": ATOM_DIM,
        "edge_dim": EDGE_DIM,
        "all_v2_atom_counts_match": all(r["v2_count_match"] for r in rows),
        "all_v2_coordinate_shapes_match": all(r["v2_coordinate_shape_match"] for r in rows),
        "max_v2_coordinate_abs_difference": max(
            (r["v2_coordinate_max_abs_difference"] or 0.0) for r in rows
        ),
        "csv_smiles_count_match": sum(r["csv_smiles_count_match"] for r in rows),
        "failures": failures,
        "records": rows,
    }
    (args.output_dir / "preprocessing_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    if failures or len(rows) != 68 or not report["all_v2_atom_counts_match"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
