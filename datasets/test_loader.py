# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

from datasets.davis_dataset import DavisDataset
from datasets.collate import mdta_collate_fn, move_batch_to_device
from models.model import MyModelMDTA


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = DavisDataset(
        pairs_csv="data/raw/davis/pairs.csv",
        drug_1d_dir="data/processed/davis/drug_1d_chemberta2",
        drug_2d_dir="data/processed/davis/drug_2d",
        protein_1d_dir="data/processed/davis/protein_1d_esm2",
        protein_2d_dir="data/processed/davis/protein_2d_graph",
        use_protein_3d=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=mdta_collate_fn,
    )

    batch = next(iter(loader))

    print("drug ids:", batch["drug_id"])
    print("protein ids:", batch["protein_id"])
    print("drug_1d shape:", batch["drug_1d"].shape)
    print("protein_1d shape:", batch["protein_1d"].shape)
    print("label shape:", batch["label"].shape)

    if batch["drug_2d"] is not None:
        print("drug_2d.x shape:", batch["drug_2d"].x.shape)
        print("drug_2d.edge_index shape:", batch["drug_2d"].edge_index.shape)

    if batch["protein_2d"] is not None:
        print("protein_2d.x shape:", batch["protein_2d"].x.shape)
        print("protein_2d.edge_index shape:", batch["protein_2d"].edge_index.shape)
        print("protein_2d.edge_weight shape:", batch["protein_2d"].edge_weight.shape)

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