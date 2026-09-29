#!/usr/bin/env python3
"""Add explicit fragment chemistry to the frozen P13D/BRICS cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, rdFingerprintGenerator


BRICS_LABELS = ["1", "3", "4", "5", "6", "7a", "7b", "8", "9", "10", "11", "12", "13", "14", "15", "16"]


def fragment_features(smiles, atom_to_fragment, n_fragments):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    heavy_assignment = atom_to_fragment[: mol.GetNumAtoms()].cpu().numpy()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    # Treat the audited atom_to_fragment mapping as authoritative. Morgan
    # centers are restricted to atoms in the fragment while the intact
    # molecule supplies chemically valid aromatic/attachment context.
    fingerprints, descriptors = [], []
    for fragment in range(n_fragments):
        atom_indices = np.flatnonzero(heavy_assignment == fragment).tolist()
        if not atom_indices:
            raise ValueError(f"fragment {fragment} has no heavy atom")
        fp = generator.GetFingerprint(mol, fromAtoms=atom_indices)
        fp_array = np.zeros(256, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        fingerprints.append(fp_array)
        atoms = [mol.GetAtomWithIdx(index) for index in atom_indices]
        descriptors.append([
            sum(atom.GetMass() for atom in atoms),
            sum(atom.GetFormalCharge() for atom in atoms),
            sum(atom.GetIsAromatic() for atom in atoms),
            sum(atom.IsInRing() for atom in atoms),
            sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms),
            sum(atom.GetAtomicNum() == 6 for atom in atoms),
            sum(atom.GetAtomicNum() == 7 for atom in atoms),
            sum(atom.GetAtomicNum() == 8 for atom in atoms),
            sum(atom.GetAtomicNum() == 16 for atom in atoms),
            len(atoms),
        ])

    environment = np.zeros((n_fragments, len(BRICS_LABELS)), dtype=np.float32)
    label_to_index = {label: index for index, label in enumerate(BRICS_LABELS)}
    for (left, right), (left_label, right_label) in BRICS.FindBRICSBonds(mol):
        lf, rf = int(heavy_assignment[left]), int(heavy_assignment[right])
        if left_label in label_to_index:
            environment[lf, label_to_index[left_label]] = 1.0
        if right_label in label_to_index:
            environment[rf, label_to_index[right_label]] = 1.0
    return np.stack(fingerprints), environment, np.asarray(descriptors, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache = torch.load(args.input_cache, map_location="cpu", weights_only=False)
    pairs = pd.read_csv(args.pairs_csv, dtype={"drug_id": str})
    smiles = pairs.drop_duplicates("drug_id").set_index("drug_id")["smiles"].to_dict()
    raw = {}
    all_descriptors = []
    for drug_id, item in cache["drugs"].items():
        fp, env, desc = fragment_features(
            smiles[drug_id], item["atom_to_fragment"], int(item["n_fragments"])
        )
        raw[drug_id] = (fp, env, desc)
        all_descriptors.append(desc)
    all_descriptors = np.concatenate(all_descriptors, axis=0)
    mean = all_descriptors.mean(axis=0)
    std = all_descriptors.std(axis=0)
    std[std < 1e-6] = 1.0

    output = {}
    for drug_id, item in cache["drugs"].items():
        fp, env, desc = raw[drug_id]
        item = dict(item)
        item["fragment_chem_feat"] = torch.from_numpy(
            np.concatenate([fp, env, (desc - mean) / std], axis=1)
        ).float()
        output[drug_id] = item
    payload = {
        "drugs": output,
        "metadata": dict(cache.get("metadata", {}), **{
            "chemistry": "fragment-centered Morgan-r2-256 + BRICS-environment-16 + normalized-atom-descriptors-10",
            "fragment_chem_dim": 282,
            "descriptor_mean": mean.tolist(),
            "descriptor_std": std.tolist(),
            "source": str(args.input_cache),
        }),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    audit = {
        "drugs": len(output),
        "fragments": int(sum(item["n_fragments"] for item in output.values())),
        "fragment_chem_dim": 282,
        "fingerprint_density": float(np.mean(np.concatenate([x[0] for x in raw.values()]))),
        "brics_environment_density": float(np.mean(np.concatenate([x[1] for x in raw.values()]))),
        "descriptor_mean": mean.tolist(),
        "descriptor_std": std.tolist(),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
