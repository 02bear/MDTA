import os
import random
import torch
from torch_geometric.data import Data, Batch

from models.model import MyModelMDTA


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_edge_index(num_nodes, extra_edges=0):
    """
    先构造一个双向链式图，保证连通；
    再随机补一些边。
    """
    edges = []

    # chain edges
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    # random extra edges
    for _ in range(extra_edges):
        a = random.randint(0, num_nodes - 1)
        b = random.randint(0, num_nodes - 1)
        if a != b:
            edges.append((a, b))
            edges.append((b, a))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def make_random_graph(num_nodes, node_dim, extra_edges=4):
    """
    生成一个随机 PyG 图：
    - x: [N, node_dim]
    - edge_index: [2, E]
    """
    x = torch.randn(num_nodes, node_dim)
    edge_index = make_edge_index(num_nodes, extra_edges=extra_edges)
    return Data(x=x, edge_index=edge_index)


def make_random_drug_2d_batch(batch_size, node_dim):
    data_list = []
    for _ in range(batch_size):
        n_atoms = random.randint(8, 20)
        g = make_random_graph(n_atoms, node_dim=node_dim, extra_edges=6)
        data_list.append(g)
    return Batch.from_data_list(data_list)


def make_random_protein_2d_batch(batch_size, node_dim):
    data_list = []
    for _ in range(batch_size):
        n_res = random.randint(50, 120)
        g = make_random_graph(n_res, node_dim=node_dim, extra_edges=20)
        data_list.append(g)
    return Batch.from_data_list(data_list)


def make_random_protein_3d_batch(batch_size, node_s_dim=6, node_v_dim=3):
    """
    按你当前 Protein3DEncoder 的接口造假数据：
    {
        "node_s": [N, 6],
        "node_v": [N, 3, 3],
        "batch":  [N]
    }
    注意：
    - node_v_dim=3 表示有 3 个向量通道，每个向量是 3 维坐标
    """
    node_s_list = []
    node_v_list = []
    batch_idx_list = []

    total_nodes = 0
    for i in range(batch_size):
        n_res = random.randint(50, 120)

        node_s = torch.randn(n_res, node_s_dim)
        node_v = torch.randn(n_res, node_v_dim, 3)
        batch_idx = torch.full((n_res,), i, dtype=torch.long)

        node_s_list.append(node_s)
        node_v_list.append(node_v)
        batch_idx_list.append(batch_idx)

        total_nodes += n_res

    protein_3d = {
        "node_s": torch.cat(node_s_list, dim=0),   # [N, 6]
        "node_v": torch.cat(node_v_list, dim=0),   # [N, 3, 3]
        "batch": torch.cat(batch_idx_list, dim=0)  # [N]
    }
    return protein_3d


def make_fake_batch(batch_size=4, protein_seq_len=100):
    """
    按你当前设定造一整个 batch：
    - drug_1d:    [B, 768]
    - drug_2d:    PyG Batch
    - protein_1d: [B, L, 1280]
    - protein_2d: PyG Batch
    - protein_3d: dict(node_s, node_v, batch)
    """
    batch = {
        "drug_1d": torch.randn(batch_size, 768),
        "drug_2d": None,
        #"drug_2d": make_random_drug_2d_batch(batch_size=batch_size, node_dim=64),
        "protein_1d": torch.randn(batch_size, 1280),
        "protein_2d": None,
        #"protein_2d": make_random_protein_2d_batch(batch_size=batch_size, node_dim=128),
        "protein_3d": None,
        #"protein_3d": make_random_protein_3d_batch(batch_size=batch_size, node_s_dim=6, node_v_dim=3),
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
        else:
            out[k] = v.to(device)
    return out


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MyModelMDTA(
        drug_1d_in_dim=768,
        drug_2d_node_dim=64,
        protein_1d_in_dim=1280,
        protein_2d_node_dim=128,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        dropout=0.1,
        task="classification",
    ).to(device)

    model.eval()

    batch = make_fake_batch(batch_size=4, protein_seq_len=100)
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        out = model(batch)

    print("Forward success.")
    print("Output shape:", out.shape)
    print("Output sample:")
    print(out[:4])


if __name__ == "__main__":
    main()