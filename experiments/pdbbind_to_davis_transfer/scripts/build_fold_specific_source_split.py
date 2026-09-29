#!/usr/bin/env python3
"""Remove Davis validation/test chemistry from every PDBbind source partition."""

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


def load_sdf(path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"cannot parse {path}")
    return Chem.RemoveHs(mol)


def keys(mol):
    mol = Chem.RemoveHs(mol)
    variants = [mol]
    try:
        fragment = rdMolStandardize.FragmentParent(mol)
        variants.extend([fragment, rdMolStandardize.TautomerParent(fragment)])
    except Exception:
        pass
    exact, scaffold = set(), set()
    for variant in variants:
        # Some rdMolStandardize outputs do not have RingInfo initialized in
        # RDKit 2024.03. Reparse a canonical representation before Murcko.
        variant = Chem.MolFromSmiles(
            Chem.MolToSmiles(variant, canonical=True, isomericSmiles=True)
        )
        if variant is None:
            continue
        try:
            exact.add(Chem.MolToInchiKey(variant).split("-")[0])
        except Exception:
            exact.add(Chem.MolToSmiles(variant, canonical=True, isomericSmiles=False))
        scaf = MurckoScaffold.GetScaffoldForMol(variant)
        value = Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=False)
        if value:
            scaffold.add(value)
    return exact, scaffold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("/data1/ztx/MyModel-MDTA"))
    p.add_argument("--pdbbind-root", type=Path,
                   default=Path("/data1/ztx/DTBind/Data_raw/pdbbind_v2020/refined-set"))
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    source_path = args.project / "experiments/pdbbind_residue_transfer/data/dual_cold_homology30_cov80_scaffold_seed42.json"
    source = json.loads(source_path.read_text())
    davis_split_path = args.project / f"data/splits/davis_drug_cold_5fold_seed42/fold_{args.fold}/split.json"
    davis_split = json.loads(davis_split_path.read_text())
    heldout = set(map(str, davis_split["val_drugs"] + davis_split["test_drugs"]))
    smiles = {}
    with (args.project / "data/raw/davis/drugs.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            smiles[str(row["drug_id"])] = row["smiles"]

    heldout_exact, heldout_scaffold = set(), set()
    heldout_key_sources = {}
    for drug_id in sorted(heldout):
        molecule_variants = [
            ("sdf", load_sdf(args.project / f"data/raw/davis/pubchem_sdf/{drug_id}.sdf")),
            ("csv", Chem.MolFromSmiles(smiles[drug_id])),
        ]
        heldout_key_sources[drug_id] = {"exact": [], "scaffold": []}
        for representation, mol in molecule_variants:
            exact, scaffold = keys(mol)
            heldout_exact.update(exact)
            heldout_scaffold.update(scaffold)
            heldout_key_sources[drug_id]["exact"].extend(sorted(exact))
            heldout_key_sources[drug_id]["scaffold"].extend(sorted(scaffold))

    cleaned, removed = {}, {}
    for partition, pdb_ids in source["split_ids"].items():
        cleaned[partition], removed[partition] = [], []
        for pdb_id in pdb_ids:
            mol = load_sdf(args.pdbbind_root / pdb_id / f"{pdb_id}_ligand.sdf")
            exact, scaffold = keys(mol)
            exact_hit = sorted(exact & heldout_exact)
            scaffold_hit = sorted(scaffold & heldout_scaffold)
            if exact_hit or scaffold_hit:
                removed[partition].append({
                    "pdb_id": pdb_id, "exact_connectivity_hits": exact_hit,
                    "scaffold_hits": scaffold_hit,
                })
            else:
                cleaned[partition].append(pdb_id)

    output = {
        "parameters": {
            "fold": args.fold,
            "policy": "Exclude from all PDBbind partitions if either Davis CSV or SDF heldout representation matches by connectivity InChIKey block or standardized Murcko scaffold.",
            "source_split": str(source_path),
            "davis_split": str(davis_split_path),
            "heldout_davis_drugs": sorted(heldout),
        },
        "split_ids": cleaned,
        "split_counts": {k: len(v) for k, v in cleaned.items()},
        "removed_counts": {k: len(v) for k, v in removed.items()},
        "removed": removed,
        "heldout_key_sources": heldout_key_sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k not in ("split_ids", "removed", "heldout_key_sources")}, indent=2))


if __name__ == "__main__":
    main()
