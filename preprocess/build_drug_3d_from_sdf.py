# -*- coding: utf-8 -*-
"""
将 PubChem SDF 转成药物 3D 图特征（供 EGNN 使用）。

输入目录默认：data/raw/davis/pubchem_sdf
输出目录默认：data/processed/davis/drug_3d

每个输出 .pt 文件包含：
- x:          FloatTensor [N, F]
- pos:        FloatTensor [N, 3]
- edge_index: LongTensor  [2, E]

其中默认 F=10（可通过 --node_feat_dim 选择 10 或 43）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdchem


ATOM_SYMBOLS = [
    "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "other"
]

DEGREES = [0, 1, 2, 3, 4, 5, "other"]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    "other",
]


def one_hot_with_other(value, choices):
    if value not in choices:
        value = choices[-1]
    return [1.0 if value == c else 0.0 for c in choices]


def atom_feature_10(atom):
    """10维轻量节点特征：原子序数/电荷/芳香性/是否成环/度数/杂化。"""
    z = float(atom.GetAtomicNum()) / 100.0
    charge = float(atom.GetFormalCharge())
    aromatic = float(atom.GetIsAromatic())
    ring = float(atom.IsInRing())

    degree = atom.GetDegree()
    if degree not in DEGREES[:-1]:
        degree = "other"
    degree_oh = one_hot_with_other(degree, DEGREES)  # 7

    hyb = atom.GetHybridization()
    if hyb not in HYBRIDIZATIONS[:-1]:
        hyb = "other"
    hyb_oh = one_hot_with_other(hyb, HYBRIDIZATIONS)  # 4

    # 1 + 1 + 1 + 1 + 7 + 4 = 15
    raw = [z, charge, aromatic, ring] + degree_oh + hyb_oh
    # 压到 10维：取前10个，保持和当前 model 默认一致
    return raw[:10]


def atom_feature_43(atom):
    """与 2D 脚本近似对齐的 43 维特征，便于后续统一。"""
    symbols_16 = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I", "At", "other"]
    num_hs_6 = [0, 1, 2, 3, 4, "other"]
    chiral_4 = [
        rdchem.ChiralType.CHI_UNSPECIFIED,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        "other",
    ]
    hyb_6 = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
        "other",
    ]

    symbol = atom.GetSymbol()
    if symbol not in symbols_16:
        symbol = "other"

    degree = atom.GetDegree()
    if degree not in DEGREES[:-1]:
        degree = "other"

    num_h = atom.GetTotalNumHs()
    if num_h not in num_hs_6[:-1]:
        num_h = "other"

    hyb = atom.GetHybridization()
    if hyb not in hyb_6[:-1]:
        hyb = "other"

    chiral = atom.GetChiralTag()
    if chiral not in chiral_4[:-1]:
        chiral = "other"

    feats = []
    feats += one_hot_with_other(symbol, symbols_16)    # 16
    feats += one_hot_with_other(degree, DEGREES)       # 7
    feats += [float(atom.GetFormalCharge())]           # 1
    feats += one_hot_with_other(num_h, num_hs_6)       # 6
    feats += one_hot_with_other(hyb, hyb_6)            # 6
    feats += [float(atom.GetIsAromatic())]             # 1
    feats += [float(atom.IsInRing())]                  # 1
    feats += one_hot_with_other(chiral, chiral_4)      # 4
    feats += [atom.GetMass() * 0.01]                   # 1
    return feats


def mol_to_graph_3d(mol, node_feat_dim=10):
    if mol is None:
        return None, "mol is None"

    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None, "No atoms"

    if node_feat_dim == 10:
        x = torch.tensor([atom_feature_10(a) for a in mol.GetAtoms()], dtype=torch.float32)
    elif node_feat_dim == 43:
        x = torch.tensor([atom_feature_43(a) for a in mol.GetAtoms()], dtype=torch.float32)
    else:
        return None, f"Unsupported node_feat_dim={node_feat_dim}, expected 10 or 43"

    pos = []
    for i in range(num_atoms):
        p = conf.GetAtomPosition(i)
        pos.append([float(p.x), float(p.y), float(p.z)])
    pos = torch.tensor(pos, dtype=torch.float32)

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    out = {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "num_nodes": int(num_atoms),
    }
    return out, None


def read_first_mol_from_sdf(sdf_path: Path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    if supplier is None or len(supplier) == 0:
        return None
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/raw/davis/drugs.csv")
    parser.add_argument("--sdf_dir", type=str, default="data/raw/davis/pubchem_sdf")
    parser.add_argument("--output_dir", type=str, default="data/processed/davis/drug_3d")
    parser.add_argument("--id_col", type=str, default="drug_id")
    parser.add_argument("--node_feat_dim", type=int, default=10, choices=[10, 43])
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    sdf_dir = Path(args.sdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df[[args.id_col]].drop_duplicates().reset_index(drop=True)

    failures = []
    success = 0

    for idx, row in df.iterrows():
        drug_id = str(row[args.id_col]).strip()
        sdf_path = sdf_dir / f"{drug_id}.sdf"

        if not sdf_path.exists():
            failures.append({"drug_id": drug_id, "error": "sdf_not_found"})
            continue

        mol = read_first_mol_from_sdf(sdf_path)
        if mol is None:
            failures.append({"drug_id": drug_id, "error": "rdkit_read_failed"})
            continue

        if mol.GetNumConformers() == 0:
            failures.append({"drug_id": drug_id, "error": "no_conformer"})
            continue

        graph, err = mol_to_graph_3d(mol, node_feat_dim=args.node_feat_dim)
        if graph is None:
            failures.append({"drug_id": drug_id, "error": err})
            continue

        torch.save(graph, output_dir / f"{drug_id}.pt")
        success += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            print(f"[{idx + 1}/{len(df)}] processed, success={success}, failed={len(failures)}")

    fail_path = output_dir / "failures.csv"
    pd.DataFrame(failures).to_csv(fail_path, index=False)

    meta = {
        "num_input_drugs": int(len(df)),
        "num_success": int(success),
        "num_failed": int(len(failures)),
        "node_feat_dim": int(args.node_feat_dim),
        "input_csv": str(input_csv),
        "sdf_dir": str(sdf_dir),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Done")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Failures written to: {fail_path}")


if __name__ == "__main__":
    main()
