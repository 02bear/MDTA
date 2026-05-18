# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d


def main():
    dataset = DavisDatasetP13D(
        pairs_csv="data/raw/davis/pairs.csv",
        drug_1d_dir="data/processed/davis/drug_unimol",
        protein_1d_dir="data/processed/davis/protein_1d_esm2",
        protein_3d_dir="data/processed/davis/protein_3d_gvp",
        drug_2d_dir="data/processed/davis/drug_2d",
        use_drug_2d=False,
    )

    print("dataset len:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=mdta_collate_fn_p13d,
    )

    batch = next(iter(loader))

    print("drug ids:", batch["drug_id"])
    print("protein ids:", batch["protein_id"])

    print("drug_1d shape:", batch["drug_1d"].shape)
    print("protein_1d shape:", batch["protein_1d"].shape)

    print("protein_3d.node_s shape:", batch["protein_3d"]["node_s"].shape)
    print("protein_3d.node_v shape:", batch["protein_3d"]["node_v"].shape)
    print("protein_3d.edge_index shape:", batch["protein_3d"]["edge_index"].shape)
    print("protein_3d.edge_s shape:", batch["protein_3d"]["edge_s"].shape)
    print("protein_3d.edge_v shape:", batch["protein_3d"]["edge_v"].shape)
    print("protein_3d.batch shape:", batch["protein_3d"]["batch"].shape)

    print("label shape:", batch["label"].shape)


if __name__ == "__main__":
    main()