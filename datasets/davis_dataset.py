# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset


def safe_filename(name: str):
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        name = name.replace(ch, "_")
    return name


class DavisDataset(Dataset):
    """
    从 pairs.csv 读取样本，并按 drug_id / protein_id 去磁盘索引真实特征文件。

    当前返回字段：
        {
            "drug_id": str,
            "protein_id": str,
            "drug_1d": Tensor [768],
            "drug_2d": PyG Data,
            "protein_1d": Tensor [1280],
            "protein_2d": PyG Data,
            "protein_3d": None,
            "label": Tensor [1]
        }
    """

    def __init__(
        self,
        pairs_csv: str,
        drug_1d_dir: str,
        drug_2d_dir: str,
        protein_1d_dir: str,
        protein_2d_dir: str,
        use_protein_3d: bool = False,
        protein_3d_dir: str | None = None,
    ):
        self.pairs_csv = Path(pairs_csv)
        self.drug_1d_dir = Path(drug_1d_dir)
        self.drug_2d_dir = Path(drug_2d_dir)
        self.protein_1d_dir = Path(protein_1d_dir)
        self.protein_2d_dir = Path(protein_2d_dir)

        self.use_protein_3d = use_protein_3d
        self.protein_3d_dir = Path(protein_3d_dir) if protein_3d_dir is not None else None

        self.df = pd.read_csv(self.pairs_csv).reset_index(drop=True)

        required_cols = {"drug_id", "protein_id", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"pairs.csv 缺少必要列: {missing}")

    def __len__(self):
        return len(self.df)

    def _load_drug_1d(self, drug_id: str):
        path = self.drug_1d_dir / f"{safe_filename(drug_id)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"drug_1d 文件不存在: {path}")

        obj = torch.load(path, weights_only=False)
        # 统一用 mean 作为 1D 分支输入
        return obj["mean"].float()   # [768]

    def _load_drug_2d(self, drug_id: str):
        path = self.drug_2d_dir / f"{safe_filename(drug_id)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"drug_2d 文件不存在: {path}")

        graph = torch.load(path, weights_only=False)
        return graph

    def _load_protein_1d(self, protein_id: str):
        path = self.protein_1d_dir / f"{safe_filename(protein_id)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"protein_1d 文件不存在: {path}")

        obj = torch.load(path, weights_only=False)
        return obj["mean"].float()   # [1280]

    def _load_protein_2d(self, protein_id: str):
        path = self.protein_2d_dir / f"{safe_filename(protein_id)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"protein_2d 文件不存在: {path}")

        graph = torch.load(path, weights_only=False)
        return graph

    def _load_protein_3d(self, protein_id: str):
        if not self.use_protein_3d:
            return None

        if self.protein_3d_dir is None:
            raise ValueError("use_protein_3d=True 但没有提供 protein_3d_dir")

        path = self.protein_3d_dir / f"{safe_filename(protein_id)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"protein_3d 文件不存在: {path}")

        obj = torch.load(path, weights_only=False)
        return obj

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        drug_id = str(row["drug_id"])
        protein_id = str(row["protein_id"])
        label = torch.tensor([float(row["label"])], dtype=torch.float32)

        sample = {
            "drug_id": drug_id,
            "protein_id": protein_id,
            "drug_1d": self._load_drug_1d(drug_id),
            "drug_2d": self._load_drug_2d(drug_id),
            "protein_1d": self._load_protein_1d(protein_id),
            "protein_2d": self._load_protein_2d(protein_id),
            "protein_3d": self._load_protein_3d(protein_id),
            "label": label,
        }
        return sample