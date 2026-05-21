import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class EGNNLayer(nn.Module):
    """
    简化版 EGNN layer（E(n)-equivariant）
    输入:
      h: [N, H]
      x: [N, 3]
      edge_index: [2, E]
    输出:
      h_new: [N, H]
      x_new: [N, 3]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor):
        src, dst = edge_index[0], edge_index[1]

        x_diff = x[src] - x[dst]                  # [E, 3]
        dist2 = torch.sum(x_diff * x_diff, dim=-1, keepdim=True)  # [E, 1]

        m_ij = self.edge_mlp(torch.cat([h[src], h[dst], dist2], dim=-1))  # [E, H]

        trans = x_diff * self.coord_mlp(m_ij)  # [E, 3]

        num_nodes = h.size(0)
        agg_m = torch.zeros_like(h)
        agg_m.index_add_(0, dst, m_ij)

        agg_x = torch.zeros_like(x)
        agg_x.index_add_(0, dst, trans)

        h_new = self.norm(h + self.node_mlp(torch.cat([h, agg_m], dim=-1)))
        x_new = x + agg_x
        return h_new, x_new


class Drug3DEGNNEncoder(nn.Module):
    """
    三层 EGNN 的药物 3D 编码器。

    期望输入 data(dict) 至少包含：
    {
        "x": Tensor [N, node_in_dim],
        "pos": Tensor [N, 3],
        "edge_index": LongTensor [2, E],
        "batch": LongTensor [N],
    }
    输出:
        graph_emb: [B, out_dim]
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 128,
        out_dim: int | None = None,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_layers != 3:
            raise ValueError(f"Drug3DEGNNEncoder 固定为 3 层，收到 n_layers={n_layers}")

        if out_dim is None:
            out_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([EGNNLayer(hidden_dim, dropout=dropout) for _ in range(3)])

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, data: dict) -> torch.Tensor:
        required = ["x", "pos", "edge_index", "batch"]
        missing = [k for k in required if k not in data or data[k] is None]
        if missing:
            raise ValueError(f"Drug3DEGNNEncoder 缺少必需输入: {missing}")

        h = self.input_proj(data["x"])        # [N, H]
        x = data["pos"]                        # [N, 3]
        edge_index = data["edge_index"]       # [2, E]
        batch = data["batch"]                 # [N]

        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        graph_emb = global_mean_pool(h, batch)
        return self.out_proj(graph_emb)
