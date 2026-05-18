import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class Drug2DEncoder(nn.Module):
    """
    输入：
        data.x          [N, node_in_dim]
        data.edge_index [2, E]
        data.batch      [N]
    输出：
        [B, out_dim]
    """
    def __init__(self, node_in_dim, hidden_dim=128, out_dim=256, heads=4, dropout=0.1):
        super().__init__()

        self.conv1 = GATv2Conv(node_in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)

        self.project = nn.Sequential(
            nn.Linear(hidden_dim * heads, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))

        graph_emb = global_mean_pool(x, batch)
        out = self.project(graph_emb)
        return out