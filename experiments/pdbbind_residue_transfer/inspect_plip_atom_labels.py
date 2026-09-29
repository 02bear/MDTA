#!/usr/bin/env python3
"""Build one temporary complex and inspect PLIP atom/residue interaction fields."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from plip.structure.preparation import PDBComplex


def ligand_pdb_lines(obabel: str, ligand_sdf: Path):
    result = subprocess.run(
        [obabel, str(ligand_sdf), "-opdb"], check=True, text=True, capture_output=True
    )
    return result.stdout.splitlines()


def build_complex(protein_pdb: Path, ligand_sdf: Path, obabel: str, output: Path):
    protein_lines = protein_pdb.read_text(errors="replace").splitlines()
    kept = [line for line in protein_lines if not line.startswith(("END", "CONECT"))]
    max_serial = max(
        [int(line[6:11]) for line in kept if line.startswith(("ATOM  ", "HETATM"))]
        or [0]
    )
    ligand_lines = ligand_pdb_lines(obabel, ligand_sdf)
    serial_map = {}
    next_serial = max_serial + 1
    rewritten = []
    ligand_atom_order = []
    heavy_atom_index = 0
    for line in ligand_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            old_serial = int(line[6:11])
            serial_map[old_serial] = next_serial
            atom_name = line[12:16]
            element = line[76:78] if len(line) >= 78 else atom_name.strip()[:1]
            element = element.strip().upper()
            new_line = (
                f"HETATM{next_serial:5d} {atom_name} LIG Z 999    "
                f"{line[30:54]}  1.00  0.00          {element:>2}  "
            )
            rewritten.append(new_line)
            ligand_atom_order.append(
                {
                    "sdf_zero_based": len(ligand_atom_order),
                    "pdb_serial": next_serial,
                    "coord": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    "element": element,
                    "graph_atom_index": None if element == "H" else heavy_atom_index,
                }
            )
            if element != "H":
                heavy_atom_index += 1
            next_serial += 1
        elif line.startswith("CONECT"):
            values = [int(value) for value in line.split()[1:]]
            mapped = [serial_map[value] for value in values if value in serial_map]
            if mapped:
                rewritten.append("CONECT" + "".join(f"{value:5d}" for value in mapped))
    output.write_text("\n".join(kept + rewritten + ["END"]) + "\n")
    return ligand_atom_order


def jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


def public_fields(interaction):
    if hasattr(interaction, "_asdict"):
        return interaction._asdict()
    if hasattr(interaction, "__dict__"):
        return vars(interaction)
    return {
        name: getattr(interaction, name)
        for name in dir(interaction)
        if not name.startswith("_") and not callable(getattr(interaction, name))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--obabel", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        complex_path = Path(temp_dir) / "complex.pdb"
        atom_map = build_complex(args.protein, args.ligand, args.obabel, complex_path)
        mol = PDBComplex()
        mol.load_pdb(str(complex_path))
        mol.analyze()
        interaction_sets = {}
        for binding_site, interaction_set in mol.interaction_sets.items():
            interaction_sets[str(binding_site)] = {
                "counts": {
                    name: len(getattr(interaction_set, name, []))
                    for name in (
                        "hydrophobic_contacts",
                        "hbonds_pdon",
                        "hbonds_ldon",
                        "pistacking",
                        "pication_paro",
                        "pication_laro",
                        "saltbridge_lneg",
                        "saltbridge_pneg",
                        "halogen_bonds",
                        "water_bridges",
                    )
                },
                "interactions": [
                    {
                        "class": type(interaction).__name__,
                        "fields": {
                            key: jsonable(value)
                            for key, value in public_fields(interaction).items()
                            if not key.startswith("_")
                        },
                    }
                    for interaction in interaction_set.all_itypes
                ],
            }
        report = {"ligand_atom_map": atom_map, "interaction_sets": interaction_sets}
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
