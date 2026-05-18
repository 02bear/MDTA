# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, random_split

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

    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len

    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=mdta_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=mdta_collate_fn,
    )

    model = MyModelMDTA(
        drug_1d_in_dim=768,
        drug_2d_node_dim=43,
        protein_1d_in_dim=1280,
        protein_2d_node_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        dropout=0.1,
        task="regression",
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Train size:", len(train_set))
    print("Val size:", len(val_set))

    model.train()
    for step, batch in enumerate(train_loader):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        pred = model(batch)               # [B, 1]
        target = batch["label"]           # [B, 1]

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        print(f"step {step} | loss = {loss.item():.6f}")

        if step >= 2:
            break

    model.eval()
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        val_batch = move_batch_to_device(val_batch, device)
        val_pred = model(val_batch)
        val_loss = criterion(val_pred, val_batch["label"])

    print("Validation check loss:", float(val_loss))


if __name__ == "__main__":
    main()