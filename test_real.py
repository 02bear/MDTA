# -*- coding: utf-8 -*-
import pandas as pd
import torch
from torch_geometric.data import Batch

from models.model import MyModelMDTA


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device) if vv is not None else None) for kk, vv in v.items()}
        else:
            out[k] = v.to(device)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv("data/raw/davis/pairs.csv")
    row = df.iloc[0]

    drug_id = str(row["drug_id"])
    protein_id = str(row["protein_id"])
    label = float(row["label"])

    print("drug_id:", drug_id)
    print("protein_id:", protein_id)
    print("label:", label)

    drug1d_obj = torch.load(
        f"data/processed/davis/drug_1d_chemberta2/{drug_id}.pt",
        weights_only=False
    )
    drug_1d = drug1d_obj["mean"].unsqueeze(0)

    prot1d_obj = torch.load(
        f"data/processed/davis/protein_1d_esm2/{protein_id}.pt",
        weights_only=False
    )
    protein_1d = prot1d_obj["mean"].unsqueeze(0)

    drug_2d_graph = torch.load(
        f"data/processed/davis/drug_2d/{drug_id}.pt",
        weights_only=False
    )
    drug_2d = Batch.from_data_list([drug_2d_graph])

    protein_2d_graph = torch.load(
        f"data/processed/davis/protein_2d_graph/{protein_id}.pt",
        weights_only=False
    )
    protein_2d = Batch.from_data_list([protein_2d_graph])

    batch = {
        "drug_1d": drug_1d,
        "drug_2d": drug_2d,
        "protein_1d": protein_1d,
        "protein_2d": protein_2d,
        "protein_3d": None,
    }

    batch = move_batch_to_device(batch, device)

    model = MyModelMDTA(
        drug_1d_in_dim=768,
        drug_2d_node_dim=43,
        protein_1d_in_dim=1280,
        protein_2d_node_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        dropout=0.1,
        task="classification",
    ).to(device)

    model.eval()

    with torch.no_grad():
        out = model(batch)

    print("Forward success.")
    print("Output shape:", out.shape)
    print("Output:", out)


if __name__ == "__main__":
    main()
