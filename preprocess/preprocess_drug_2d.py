from pathlib import Path
import json
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem


ATOM_SYMBOLS = [
    "B", "C", "N", "O", "F", "Si", "P", "S",
    "Cl", "As", "Se", "Br", "Te", "I", "At", "other"
]

DEGREES = [0, 1, 2, 3, 4, 5, "other"]
NUM_HS = [0, 1, 2, 3, 4, "other"]

HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    "other"
]

CHIRAL_TAGS = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    "other"
]

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
    "other"
]

STEREOS = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]


def one_hot_with_other(value, choices):
    if value not in choices:
        value = choices[-1]
    return [1.0 if value == c else 0.0 for c in choices]


def atom_to_feature(atom):
    symbol = atom.GetSymbol()
    if symbol not in ATOM_SYMBOLS:
        symbol = "other"

    degree = atom.GetDegree()
    if degree not in DEGREES[:-1]:
        degree = "other"

    num_h = atom.GetTotalNumHs()
    if num_h not in NUM_HS[:-1]:
        num_h = "other"

    hyb = atom.GetHybridization()
    if hyb not in HYBRIDIZATIONS[:-1]:
        hyb = "other"

    chiral = atom.GetChiralTag()
    if chiral not in CHIRAL_TAGS[:-1]:
        chiral = "other"

    feats = []
    feats += one_hot_with_other(symbol, ATOM_SYMBOLS)          # 16
    feats += one_hot_with_other(degree, DEGREES)               # 7
    feats += [float(atom.GetFormalCharge())]                   # 1
    feats += one_hot_with_other(num_h, NUM_HS)                 # 6
    feats += one_hot_with_other(hyb, HYBRIDIZATIONS)           # 6
    feats += [float(atom.GetIsAromatic())]                     # 1
    feats += [float(atom.IsInRing())]                          # 1
    feats += one_hot_with_other(chiral, CHIRAL_TAGS)           # 4
    feats += [atom.GetMass() * 0.01]                           # 1, 缩放一下

    return feats  # total = 43


def bond_to_feature(bond):
    bond_type = bond.GetBondType()
    if bond_type not in BOND_TYPES[:-1]:
        bond_type = "other"

    stereo = bond.GetStereo()
    if stereo not in STEREOS:
        stereo = rdchem.BondStereo.STEREONONE

    feats = []
    feats += one_hot_with_other(bond_type, BOND_TYPES)         # 5
    feats += [float(bond.GetIsConjugated())]                   # 1
    feats += [float(bond.GetIsAromatic())]                     # 1
    feats += [float(bond.IsInRing())]                          # 1
    feats += [1.0 if stereo == s else 0.0 for s in STEREOS]    # 6

    return feats  # total = 14


def smiles_to_pyg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "MolFromSmiles failed"

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None, f"SanitizeMol failed: {e}"

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None, "No atoms"

    x = torch.tensor(
        [atom_to_feature(atom) for atom in mol.GetAtoms()],
        dtype=torch.float
    )

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_to_feature(bond)

        # 双向边
        edge_indices.append([i, j])
        edge_attrs.append(feat)

        edge_indices.append([j, i])
        edge_attrs.append(feat)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 14), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_atoms
    )
    return data, None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="path to drugs.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="where to save .pt graphs")
    parser.add_argument("--id_col", type=str, default="drug_id")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df[[args.id_col, args.smiles_col]].drop_duplicates().reset_index(drop=True)

    failures = []
    success_count = 0
    first_x_dim = None
    first_edge_dim = None

    for idx, row in df.iterrows():
        drug_id = str(row[args.id_col])
        smiles = str(row[args.smiles_col])

        data, err = smiles_to_pyg(smiles)
        if data is None:
            failures.append({
                "drug_id": drug_id,
                "smiles": smiles,
                "error": err
            })
            continue

        save_path = output_dir / f"{drug_id}.pt"
        torch.save(data, save_path)

        success_count += 1

        if first_x_dim is None:
            first_x_dim = int(data.x.size(1))
            first_edge_dim = int(data.edge_attr.size(1)) if data.edge_attr.numel() > 0 else 14

        if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
            print(f"[{idx + 1}/{len(df)}] processed")

    fail_path = output_dir / "failures.csv"
    pd.DataFrame(failures).to_csv(fail_path, index=False)

    meta = {
        "num_input_drugs": int(len(df)),
        "num_success": int(success_count),
        "num_failed": int(len(failures)),
        "node_feature_dim": first_x_dim,
        "edge_feature_dim": first_edge_dim,
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved graphs to: {output_dir}")
    print(f"Failure log: {fail_path}")


if __name__ == "__main__":
    main()