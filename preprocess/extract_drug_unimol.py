# -*- coding: utf-8 -*-
from pathlib import Path
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from unimol_tools import UniMolRepr


def safe_filename(name: str):
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def canonicalize_smiles(smiles: str):
    smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--id_col", type=str, default="drug_id")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--canonicalize", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df[[args.id_col, args.smiles_col]].drop_duplicates().reset_index(drop=True)

    model = UniMolRepr(data_type="molecule", remove_hs=False)

    failures = []
    success_count = 0
    embedding_dim = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Uni-Mol drug embeddings"):
        drug_id = str(row[args.id_col])
        smiles = str(row[args.smiles_col])

        smiles_used = smiles
        if args.canonicalize:
            cano = canonicalize_smiles(smiles)
            if cano is None:
                failures.append({
                    "drug_id": drug_id,
                    "smiles": smiles,
                    "error": "RDKit canonicalization failed"
                })
                continue
            smiles_used = cano

        try:
            out = model.get_repr([smiles_used])
            vec = np.array(out["cls_repr"][0], dtype=np.float32)
            vec = torch.tensor(vec)

            obj = {
                "drug_id": drug_id,
                "smiles": smiles_used,
                "mean": vec,   # 为了和你现有代码兼容，统一还叫 mean
            }

            save_path = output_dir / f"{safe_filename(drug_id)}.pt"
            torch.save(obj, save_path)

            success_count += 1
            if embedding_dim is None:
                embedding_dim = int(vec.shape[0])

        except Exception as e:
            failures.append({
                "drug_id": drug_id,
                "smiles": smiles_used,
                "error": str(e),
            })

    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)

    meta = {
        "num_input_drugs": int(len(df)),
        "num_success": int(success_count),
        "num_failed": int(len(failures)),
        "embedding_dim": embedding_dim,
        "canonicalize": bool(args.canonicalize),
        "feature_type": "unimol_cls_repr"
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved embeddings to: {output_dir}")
    print(f"Failure log: {output_dir / 'failures.csv'}")


if __name__ == "__main__":
    main()