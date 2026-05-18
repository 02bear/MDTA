# -*- coding: utf-8 -*-
import torch
from torch_geometric.data import Batch


def mdta_collate_fn(batch_data):
    """
    把单样本 dict 列表组装成模型可吃的 batch dict
    """
    drug_ids = [x["drug_id"] for x in batch_data]
    protein_ids = [x["protein_id"] for x in batch_data]

    drug_1d = torch.stack([x["drug_1d"] for x in batch_data], dim=0)         # [B, 768]
    protein_1d = torch.stack([x["protein_1d"] for x in batch_data], dim=0)   # [B, 1280]
    label = torch.stack([x["label"] for x in batch_data], dim=0)             # [B, 1]

    drug_2d_list = [x["drug_2d"] for x in batch_data]
    protein_2d_list = [x["protein_2d"] for x in batch_data]

    drug_2d = Batch.from_data_list(drug_2d_list) if all(g is not None for g in drug_2d_list) else None
    protein_2d = Batch.from_data_list(protein_2d_list) if all(g is not None for g in protein_2d_list) else None

    protein_3d_list = [x["protein_3d"] for x in batch_data]
    if all(v is None for v in protein_3d_list):
        protein_3d = None
    else:
        raise NotImplementedError("当前 collate_fn 还没接 protein_3d，请先保持 protein_3d=None")

    batch = {
        "drug_id": drug_ids,
        "protein_id": protein_ids,
        "drug_1d": drug_1d,
        "drug_2d": drug_2d,
        "protein_1d": protein_1d,
        "protein_2d": protein_2d,
        "protein_3d": protein_3d,
        "label": label,
    }
    return batch


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            out[k] = {
                kk: (vv.to(device) if vv is not None else None)
                for kk, vv in v.items()
            }
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = v.to(device)
    return out