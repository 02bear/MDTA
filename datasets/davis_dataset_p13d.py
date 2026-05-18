# -*- coding: utf-8 -*-
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


def _find_first_existing_column(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {name} column. Existing columns: {df.columns.tolist()}")


class DavisDatasetP13D(Dataset):
    """
    最小版本：
    - drug_1d
    - (optional) drug_2d
    - protein_1d
    - protein_3d
    - label
    并且自动过滤掉没有 protein_3d 文件的样本
    """
    def __init__(
        self,
        pairs_csv,
        drug_1d_dir,
        protein_1d_dir,
        protein_3d_dir,
        drug_2d_dir=None,
        use_drug_2d=False,
    ):
        super().__init__()

        self.pairs_csv = Path(pairs_csv)
        self.drug_1d_dir = Path(drug_1d_dir)
        self.protein_1d_dir = Path(protein_1d_dir)
        self.protein_3d_dir = Path(protein_3d_dir)

        self.drug_2d_dir = Path(drug_2d_dir) if drug_2d_dir is not None else None
        self.use_drug_2d = use_drug_2d

        df = pd.read_csv(self.pairs_csv)

        drug_col = _find_first_existing_column(
            df,
            ["drug_id", "compound_id", "Compound_ID", "drug", "ligand_id"],
            "drug_id"
        )
        protein_col = _find_first_existing_column(
            df,
            ["protein_id", "target_key", "protein", "target_id"],
            "protein_id"
        )
        label_col = _find_first_existing_column(
            df,
            ["label", "affinity", "y", "target", "pKd"],
            "label"
        )

        df = df[[drug_col, protein_col, label_col]].copy()
        df.columns = ["drug_id", "protein_id", "label"]

        # 转成字符串，避免 11314340 这种被 pandas 读成 int
        df["drug_id"] = df["drug_id"].astype(str)
        df["protein_id"] = df["protein_id"].astype(str)

        kept_rows = []
        missing_counts = {
            "drug_1d": 0,
            "drug_2d": 0,
            "protein_1d": 0,
            "protein_3d": 0,
        }

        for row in df.itertuples(index=False):
            drug_id = row.drug_id
            protein_id = row.protein_id

            drug_1d_path = self.drug_1d_dir / f"{drug_id}.pt"
            protein_1d_path = self.protein_1d_dir / f"{protein_id}.pt"
            protein_3d_path = self.protein_3d_dir / f"{protein_id}.pt"

            if not drug_1d_path.exists():
                missing_counts["drug_1d"] += 1
                continue
            if not protein_1d_path.exists():
                missing_counts["protein_1d"] += 1
                continue
            if not protein_3d_path.exists():
                missing_counts["protein_3d"] += 1
                continue

            if self.use_drug_2d:
                drug_2d_path = self.drug_2d_dir / f"{drug_id}.pt"
                if not drug_2d_path.exists():
                    missing_counts["drug_2d"] += 1
                    continue

            kept_rows.append({
                "drug_id": drug_id,
                "protein_id": protein_id,
                "label": float(row.label),
            })

        self.df = pd.DataFrame(kept_rows)

        print(f"[DavisDatasetP13D] input pairs: {len(df)}")
        print(f"[DavisDatasetP13D] kept pairs : {len(self.df)}")
        print(f"[DavisDatasetP13D] missing counts: {missing_counts}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        drug_id = row["drug_id"]
        protein_id = row["protein_id"]
        label = float(row["label"])

        drug_1d_obj = torch.load(self.drug_1d_dir / f"{drug_id}.pt", weights_only=False)
        protein_1d_obj = torch.load(self.protein_1d_dir / f"{protein_id}.pt", weights_only=False)
        protein_3d_obj = torch.load(self.protein_3d_dir / f"{protein_id}.pt", weights_only=False)

        drug_1d = drug_1d_obj["mean"].float()
        protein_1d = protein_1d_obj["mean"].float()

        if self.use_drug_2d:
            drug_2d = torch.load(self.drug_2d_dir / f"{drug_id}.pt", weights_only=False)
        else:
            drug_2d = None

        required_keys = ["node_s", "node_v", "edge_index", "edge_s", "edge_v"]
        missing_keys = [k for k in required_keys if k not in protein_3d_obj]
        if missing_keys:
            raise KeyError(
                f"protein_3d file missing required keys: {missing_keys}; file={self.protein_3d_dir / f'{protein_id}.pt'}"
            )

        sample = {
            "drug_id": drug_id,
            "protein_id": protein_id,
            "drug_1d": drug_1d,       # [512] 或 [768]
            "drug_2d": drug_2d,       # PyG Data or None
            "protein_1d": protein_1d, # [1280]
            "protein_3d": {
                "node_s": protein_3d_obj["node_s"].float(),
                "node_v": protein_3d_obj["node_v"].float(),
                "edge_index": protein_3d_obj["edge_index"].long(),
                "edge_s": protein_3d_obj["edge_s"].float(),
                "edge_v": protein_3d_obj["edge_v"].float(),
            },
            "label": torch.tensor([label], dtype=torch.float32),
        }
        return sample