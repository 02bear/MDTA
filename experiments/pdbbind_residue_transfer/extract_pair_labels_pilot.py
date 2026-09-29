#!/usr/bin/env python3
"""Extract unambiguous PLIP atom-residue pairs for a small audited sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
from pathlib import Path

from plip.structure.preparation import PDBComplex

from inspect_plip_atom_labels import build_complex, public_fields


COORD_RE = re.compile(r"\(([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\)")
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def atom_coord(atom):
    if hasattr(atom, "coords"):
        return [float(value) for value in atom.coords]
    match = COORD_RE.search(str(atom))
    if not match:
        raise ValueError(f"Cannot parse atom coordinate: {atom}")
    return [float(match.group(i)) for i in range(1, 4)]


def nearest_ligand_atom(coord, atom_map, tolerance=0.05):
    eligible = [item for item in atom_map if item["graph_atom_index"] is not None]
    distances = [
        math.sqrt(sum((a - b) ** 2 for a, b in zip(coord, item["coord"])))
        for item in eligible
    ]
    index = min(range(len(distances)), key=distances.__getitem__)
    if distances[index] > tolerance:
        raise ValueError(f"Ligand coordinate mismatch: nearest distance {distances[index]:.4f}")
    return eligible[index]["graph_atom_index"], distances[index]


def residue_map(protein_path: Path):
    mapping, residues, seen = {}, [], set()
    for line in protein_path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM  "):
            continue
        resname = line[17:20].strip()
        if resname not in STANDARD_AA:
            continue
        chain = line[21].strip() or " "
        residue_number = int(line[22:26])
        insertion_code = line[26].strip()
        key = (chain, residue_number, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        index = len(residues)
        residues.append(
            {
                "index": index,
                "chain": chain,
                "resnr": residue_number,
                "icode": insertion_code,
                "resname": resname,
            }
        )
        mapping.setdefault((chain, residue_number), []).append(index)
    return mapping, residues


def extract_one(pdb_id, refined_root, obabel):
    sample_dir = refined_root / pdb_id
    protein = sample_dir / f"{pdb_id}_protein.pdb"
    ligand = sample_dir / f"{pdb_id}_ligand.sdf"
    residue_lookup, residues = residue_map(protein)
    with tempfile.TemporaryDirectory() as temp_dir:
        complex_path = Path(temp_dir) / "complex.pdb"
        atom_map = build_complex(protein, ligand, obabel, complex_path)
        mol = PDBComplex()
        mol.load_pdb(str(complex_path))
        mol.analyze()
        pairs, skipped = [], []
        for interaction_set in mol.interaction_sets.values():
            for interaction in interaction_set.all_itypes:
                fields = public_fields(interaction)
                interaction_class = type(interaction).__name__
                ligand_atom = None
                ligand_atoms = None
                label_type = None
                if interaction_class == "hydroph_interaction":
                    ligand_atom = fields["ligatom"]
                    label_type = "hydrophobic"
                elif interaction_class == "hbond":
                    ligand_atom = fields["a"] if fields["protisdon"] else fields["d"]
                    label_type = "hbond"
                elif interaction_class == "waterbridge":
                    ligand_atom = fields["a"] if fields["protisdon"] else fields["d"]
                    label_type = "waterbridge"
                elif interaction_class == "halogenbond":
                    ligand_atom = fields["don"][0]
                    label_type = "halogenbond"
                else:
                    if interaction_class == "saltbridge":
                        ligand_group = fields["negative"] if fields["protispos"] else fields["positive"]
                        ligand_atoms = list(ligand_group[0])
                        label_type = "saltbridge"
                    elif interaction_class == "pistack":
                        ligand_atoms = list(fields["ligandring"][0])
                        label_type = "pistacking"
                    elif interaction_class == "pication":
                        ligand_group = fields["ring"] if fields["protcharged"] else fields["charge"]
                        ligand_atoms = list(ligand_group[0])
                        label_type = "pication"
                    else:
                        skipped.append(interaction_class)
                        continue
                residue_candidates = residue_lookup.get(
                    (str(fields["reschain"]), int(fields["resnr"])), []
                )
                if len(residue_candidates) != 1:
                    skipped.append(f"ambiguous_residue:{interaction_class}")
                    continue
                atoms_to_map = ligand_atoms if ligand_atoms is not None else [ligand_atom]
                for atom in atoms_to_map:
                    atom_index, distance = nearest_ligand_atom(atom_coord(atom), atom_map)
                    pairs.append(
                        {
                            "atom_index": atom_index,
                            "residue_index": residue_candidates[0],
                            "type": label_type,
                            "coord_match_distance": distance,
                        }
                    )
    unique = {
        (pair["atom_index"], pair["residue_index"], pair["type"]): pair for pair in pairs
    }
    return {
        "pdb_id": pdb_id,
        "atom_count": sum(item["graph_atom_index"] is not None for item in atom_map),
        "residue_count": len(residues),
        "pairs": list(unique.values()),
        "skipped_interaction_classes": sorted(set(skipped)),
        "max_coord_match_distance": max(
            [pair["coord_match_distance"] for pair in unique.values()] or [0.0]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--ids", default="", help="Comma-separated PDB IDs")
    parser.add_argument("--ids-json", type=Path)
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--obabel", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    records, failures = [], []
    ids = [item.strip().lower() for item in args.ids.split(",") if item.strip()]
    if args.ids_json:
        payload = json.loads(args.ids_json.read_text())
        split_ids = payload.get("split_ids", payload)
        ids = []
        for split in ("train", "val", "test"):
            ids.extend(split_ids.get(split, []))
        ids = list(dict.fromkeys(item.lower() for item in ids))
    if args.manifest_csv:
        with args.manifest_csv.open(newline="") as handle:
            ids = [
                row["pdb_id"].lower()
                for row in csv.DictReader(handle)
                if row["has_protein_graph"] == "True"
                and row["has_ligand_graph"] == "True"
                and row["in_refined_set"] == "True"
            ]
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    ids = ids[args.shard_index :: args.num_shards]
    for pdb_id in ids:
        try:
            record = extract_one(pdb_id, args.refined_root, args.obabel)
            records.append(record)
            print(pdb_id, len(record["pairs"]), record["max_coord_match_distance"], flush=True)
        except Exception as error:
            failures.append({"pdb_id": pdb_id, "error": repr(error)})
            print(pdb_id, "FAILED", repr(error), flush=True)
    report = {"records": records, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"success": len(records), "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
