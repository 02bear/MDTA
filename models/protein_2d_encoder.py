import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool


class Protein2DEncoder(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, out_dim=128, heads=4, dropout=0.1):
        super().__init__()

        # 第一层：利用加权图
        self.gcn = GCNConv(node_in_dim, hidden_dim)

        # 5层 GATv2
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
            for _ in range(5)
        ])

        # LayerNorm 让深一点时更稳
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(5)
        ])

        # 最后映射到 out_dim
        self.out_proj = nn.Linear(hidden_dim, out_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        # 先吃加权图
        x = self.gcn(x, edge_index, edge_weight=edge_weight)
        x = self.act(x)
        x = self.dropout(x)

        # 5层 GAT + residual
        for gat, norm in zip(self.gat_layers, self.norm_layers):
            h = gat(x, edge_index)
            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)
            x = x + h   # residual connection

        # 输出投影
        x = self.out_proj(x)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = global_mean_pool(x, batch)
        return x