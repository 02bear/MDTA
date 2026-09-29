#!/usr/bin/env python3
"""Print compact PLIP field schemas for representative interaction classes."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from plip.structure.preparation import PDBComplex

from inspect_plip_atom_labels import build_complex, jsonable, public_fields


def compact(value):
    converted = jsonable(value)
    text = json.dumps(converted)
    return text if len(text) <= 800 else text[:800] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--obabel", required=True)
    args = parser.parse_args()
    seen = set()
    for pdb_id in args.ids.split(","):
        sample = args.refined_root / pdb_id
        with tempfile.TemporaryDirectory() as temp_dir:
            complex_path = Path(temp_dir) / "complex.pdb"
            build_complex(
                sample / f"{pdb_id}_protein.pdb",
                sample / f"{pdb_id}_ligand.sdf",
                args.obabel,
                complex_path,
            )
            mol = PDBComplex()
            mol.load_pdb(str(complex_path))
            mol.analyze()
            for interaction_set in mol.interaction_sets.values():
                for interaction in interaction_set.all_itypes:
                    name = type(interaction).__name__
                    if name in seen:
                        continue
                    seen.add(name)
                    print("CLASS", name)
                    for key, value in public_fields(interaction).items():
                        print(" ", key, type(value).__name__, compact(value))


if __name__ == "__main__":
    main()
