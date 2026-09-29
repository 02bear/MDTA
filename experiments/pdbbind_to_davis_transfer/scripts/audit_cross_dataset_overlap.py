#!/usr/bin/env python3
"""Audit ligand leakage from the PDBbind source-training split into Davis folds."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def identifiers(mol):
    mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    inchikey = Chem.MolToInchiKey(mol)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=False)
    # Acyclic molecules have an empty Murcko scaffold. Keep them separate by
    # canonical structure instead of collapsing every acyclic ligand together.
    scaffold_key = scaffold if scaffold else f"ACYCLIC::{smiles}"
    return {"canonical_smiles": smiles, "inchikey": inchikey, "scaffold": scaffold_key}


def load_sdf(path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"cannot read molecule: {path}")
    return mol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    p.add_argument("--pdbbind-root", type=Path,
                   default=Path("/data1/ztx/DTBind/Data_raw/pdbbind_v2020/refined-set"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    project = args.project
    split_path = project / "experiments/pdbbind_residue_transfer/data/dual_cold_homology30_cov80_scaffold_seed42.json"
    source_ids = json.loads(split_path.read_text())["split_ids"]["train"]

    source_rows, failures = [], []
    for pdb_id in source_ids:
        path = args.pdbbind_root / pdb_id / f"{pdb_id}_ligand.sdf"
        try:
            source_rows.append({"pdb_id": pdb_id, **identifiers(load_sdf(path))})
        except Exception as exc:
            failures.append({"dataset": "pdbbind", "id": pdb_id, "error": str(exc)})

    drugs_csv = project / "data/raw/davis/drugs.csv"
    davis_sdf = project / "data/raw/davis/pubchem_sdf"
    davis_rows = []
    with drugs_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            drug_id = str(row["drug_id"])
            try:
                sdf_ids = identifiers(load_sdf(davis_sdf / f"{drug_id}.sdf"))
                smiles_mol = Chem.MolFromSmiles(row["smiles"])
                smiles_ids = identifiers(smiles_mol) if smiles_mol is not None else None
                davis_rows.append({
                    "drug_id": drug_id,
                    "input_smiles": row["smiles"],
                    **sdf_ids,
                    "csv_sdf_inchikey_match": bool(smiles_ids and smiles_ids["inchikey"] == sdf_ids["inchikey"]),
                    "csv_sdf_scaffold_match": bool(smiles_ids and smiles_ids["scaffold"] == sdf_ids["scaffold"]),
                })
            except Exception as exc:
                failures.append({"dataset": "davis", "id": drug_id, "error": str(exc)})

    by_inchikey, by_smiles, by_scaffold = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in source_rows:
        by_inchikey[row["inchikey"]].append(row["pdb_id"])
        by_smiles[row["canonical_smiles"]].append(row["pdb_id"])
        by_scaffold[row["scaffold"]].append(row["pdb_id"])

    for row in davis_rows:
        row["pdbbind_exact_inchikey_ids"] = by_inchikey[row["inchikey"]]
        row["pdbbind_exact_smiles_ids"] = by_smiles[row["canonical_smiles"]]
        row["pdbbind_scaffold_ids"] = by_scaffold[row["scaffold"]]
        row["exact_overlap"] = bool(row["pdbbind_exact_inchikey_ids"] or row["pdbbind_exact_smiles_ids"])
        row["scaffold_overlap"] = bool(row["pdbbind_scaffold_ids"])

    fold_reports = {}
    for fold in range(1, 6):
        fold_json = project / f"data/splits/davis_drug_cold_5fold_seed42/fold_{fold}/split.json"
        split = json.loads(fold_json.read_text())
        fold_report = {}
        for partition in ("train", "val", "test"):
            ids = set(map(str, split[f"{partition}_drugs"]))
            rows = [r for r in davis_rows if r["drug_id"] in ids]
            exact = [r["drug_id"] for r in rows if r["exact_overlap"]]
            scaffold = [r["drug_id"] for r in rows if r["scaffold_overlap"]]
            fold_report[partition] = {
                "drug_count": len(rows),
                "exact_overlap_count": len(exact),
                "exact_overlap_drugs": exact,
                "scaffold_overlap_count": len(scaffold),
                "scaffold_overlap_drugs": scaffold,
            }
        fold_reports[str(fold)] = fold_report

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "ligand_overlap_details.json"
    report_path = args.output_dir / "ligand_overlap_report.json"
    detail_path.write_text(json.dumps(davis_rows, indent=2))
    report = {
        "source_split": str(split_path),
        "pdbbind_source_train_requested": len(source_ids),
        "pdbbind_source_train_parsed": len(source_rows),
        "davis_requested": 68,
        "davis_parsed": len(davis_rows),
        "davis_csv_sdf_inchikey_match": sum(r["csv_sdf_inchikey_match"] for r in davis_rows),
        "davis_csv_sdf_scaffold_match": sum(r["csv_sdf_scaffold_match"] for r in davis_rows),
        "failures": failures,
        "folds": fold_reports,
        "policy": "Any val/test scaffold overlap requires a fold-specific source split excluding that scaffold.",
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
