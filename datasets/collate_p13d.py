# -*- coding: utf-8 -*-
import torch
from torch_geometric.data import Batch


def mdta_collate_fn_p13d(batch_list):
    out = {}

    out["drug_id"] = [b["drug_id"] for b in batch_list]
    out["protein_id"] = [b["protein_id"] for b in batch_list]

    out["drug_1d"] = torch.stack([b["drug_1d"] for b in batch_list], dim=0)

    drug_2d_list = [b["drug_2d"] for b in batch_list]
    if all(x is None for x in drug_2d_list):
        out["drug_2d"] = None
    else:
        out["drug_2d"] = Batch.from_data_list(drug_2d_list)

    out["protein_1d"] = torch.stack([b["protein_1d"] for b in batch_list], dim=0)

    drug_3d_list = [b["drug_3d"] for b in batch_list]
    if all(x is None for x in drug_3d_list):
        out["drug_3d"] = None
    else:
        d_x_list = []
        d_pos_list = []
        d_edge_index_list = []
        d_batch_list = []
        d_node_offset = 0

        for i, g in enumerate(drug_3d_list):
            n = g["x"].size(0)
            d_x_list.append(g["x"].float())
            d_pos_list.append(g["pos"].float())
            d_edge_index_list.append(g["edge_index"].long() + d_node_offset)
            d_batch_list.append(torch.full((n,), i, dtype=torch.long))
            d_node_offset += n

        out["drug_3d"] = {
            "x": torch.cat(d_x_list, dim=0),
            "pos": torch.cat(d_pos_list, dim=0),
            "edge_index": torch.cat(d_edge_index_list, dim=1),
            "batch": torch.cat(d_batch_list, dim=0),
        }

    node_s_list = []
    node_v_list = []
    edge_index_list = []
    edge_s_list = []
    edge_v_list = []
    batch_idx_list = []

    node_offset = 0
    for i, b in enumerate(batch_list):
        g = b["protein_3d"]
        n = g["node_s"].size(0)

        node_s_list.append(g["node_s"])
        node_v_list.append(g["node_v"])
        edge_index_list.append(g["edge_index"] + node_offset)
        edge_s_list.append(g["edge_s"])
        edge_v_list.append(g["edge_v"])
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))

        node_offset += n

    out["protein_3d"] = {
        "node_s": torch.cat(node_s_list, dim=0),
        "node_v": torch.cat(node_v_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_s": torch.cat(edge_s_list, dim=0),
        "edge_v": torch.cat(edge_v_list, dim=0),
        "batch": torch.cat(batch_idx_list, dim=0),
    }

    out["label"] = torch.stack([b["label"] for b in batch_list], dim=0)

    return out


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            out[k] = {}
            for kk, vv in v.items():
                if vv is None:
                    out[k][kk] = None
                else:
                    out[k][kk] = vv.to(device)
        elif hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
