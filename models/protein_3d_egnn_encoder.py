import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add


class EGNNLayerEfficient(nn.Module):
    """
    高效版 EGNN layer（完整几何特征版）。

    使用特征：
    - 节点隐状态 h
    - 真实 3D 坐标 x 计算得到的坐标差与距离 dist2
    - 边标量特征 edge_s
    - 边向量特征 edge_v（展开后拼接）
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_s_dim: int,
        edge_v_dim: int,
        dropout: float = 0.1,
        coord_scale: float = 0.1,
    ):
        super().__init__()
        self.edge_v_flat_dim = edge_v_dim * 3

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_s_dim + self.edge_v_flat_dim, hidden_dim),
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
        self.coord_scale = coord_scale

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ):
        src, dst = edge_index[0], edge_index[1]

        x_diff = x[src] - x[dst]  # [E, 3]
        dist2 = (x_diff ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        edge_v_flat = edge_v.reshape(edge_v.size(0), -1)  # [E, edge_v_dim*3]
        edge_input = torch.cat([h[src], h[dst], dist2, edge_s, edge_v_flat], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # [E, H]

        delta_x = x_diff * self.coord_scale * self.coord_mlp(m_ij)  # [E, 3]

        agg_m = scatter_add(m_ij, dst, dim=0, dim_size=h.size(0))
        agg_x = scatter_add(delta_x, dst, dim=0, dim_size=x.size(0))

        h_new = self.norm(h + self.node_mlp(torch.cat([h, agg_m], dim=-1)))
        x_new = x + agg_x
        return h_new, x_new


class Protein3DEGNNEncoder(nn.Module):
    """
    蛋白 3D 的 3 层 EGNN 编码器（完整几何特征版）。

    期望输入 data(dict) 包含：
    {
        "node_s": Tensor [N, node_s_dim],
        "node_v": Tensor [N, node_v_dim, 3],  # 方向向量特征，不作为 EGNN 坐标
        "coords": Tensor [N, 3],              # 真实 C-alpha 坐标，作为 EGNN 坐标
        "edge_index": LongTensor [2, E],
        "edge_s": Tensor [E, edge_s_dim],
        "edge_v": Tensor [E, edge_v_dim, 3],
        "batch": LongTensor [N],
    }
    输出：
        默认 return_node=False:
            graph_emb: [B, out_dim]
        当 return_node=True:
            {
                "node_feat": Tensor [N, out_dim],   # 残基级局部特征
                "graph_feat": Tensor [B, out_dim],  # 与原 baseline 一致的全局特征
                "batch": LongTensor [N],            # 每个残基所属蛋白编号
            }
    """

    def __init__(
        self,
        node_s_dim: int,
        hidden_dim: int = 256,
        out_dim: int | None = None,
        dropout: float = 0.1,
        coord_scale: float = 0.1,
        n_layers: int = 3,
        edge_s_dim: int = 32,
        edge_v_dim: int = 1,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim
        if n_layers != 3:
            raise ValueError(f"Protein3DEGNNEncoder 固定为 3 层，收到 n_layers={n_layers}")

        self.input_proj = nn.Sequential(
            nn.Linear(node_s_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList(
            [
                EGNNLayerEfficient(
                    hidden_dim=hidden_dim,
                    edge_s_dim=edge_s_dim,
                    edge_v_dim=edge_v_dim,
                    dropout=dropout,
                    coord_scale=coord_scale,
                )
                for _ in range(3)
            ]
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _center_pos_by_graph(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if num_graphs == 0:
            return pos

        sum_pos = scatter_add(pos, batch, dim=0, dim_size=num_graphs)  # [B, 3]
        ones = torch.ones((pos.size(0), 1), device=pos.device, dtype=pos.dtype)
        cnt = scatter_add(ones, batch, dim=0, dim_size=num_graphs).clamp_min_(1.0)  # [B, 1]
        mean_pos = sum_pos / cnt
        return pos - mean_pos[batch]

    def forward(self, data: dict, return_node: bool = False):
        required = ["node_s", "node_v", "coords", "edge_index", "edge_s", "edge_v", "batch"]
        missing = [k for k in required if k not in data or data[k] is None]
        if missing:
            raise ValueError(f"Protein3DEGNNEncoder 缺少必需输入: {missing}")

        node_s = data["node_s"]
        node_v = data["node_v"]
        coords = data["coords"]
        edge_index = data["edge_index"]
        edge_s = data["edge_s"]
        edge_v = data["edge_v"]
        batch = data["batch"]

        if node_v.dim() != 3 or node_v.size(-1) != 3 or node_v.size(1) < 1:
            raise ValueError(f"node_v 形状应为 [N,nv,3] 且 nv>=1，实际 {tuple(node_v.shape)}")
        if coords.dim() != 2 or coords.size(-1) != 3:
            raise ValueError(f"coords 形状应为 [N,3]，实际 {tuple(coords.shape)}")
        if coords.size(0) != node_s.size(0):
            raise ValueError(
                f"coords 和 node_s 的节点数不一致: coords={coords.size(0)}, node_s={node_s.size(0)}"
            )
        if batch.size(0) != node_s.size(0):
            raise ValueError(
                f"batch 和 node_s 的节点数不一致: batch={batch.size(0)}, node_s={node_s.size(0)}"
            )

        h = self.input_proj(node_s)  # [N, H]

        # 关键修正：使用构图脚本保存的真实 C-alpha 坐标 coords 作为 EGNN 坐标。
        # node_v[:, 0, :] 是 forward unit vector，不是真实坐标，不能作为 x。
        x = coords.to(device=h.device, dtype=h.dtype)  # [N, 3]
        x = self._center_pos_by_graph(x, batch)
        # Scale real C-alpha coordinates from Angstrom-level values
        # to a numerically stable range for EGNN distance features.
        x = x / 10.0

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_s, edge_v)

        # 保持 baseline 的全局输出路径不变：先 pool，再 out_proj。
        graph_emb = self.out_proj(global_mean_pool(h, batch))

        if not return_node:
            return graph_emb

        # 局部交互需要 pooling 之前的残基级表示。
        # 这里复用 out_proj 统一到 out_dim；不改变 graph_emb 的计算方式。
        node_feat = self.out_proj(h)
        return {
            "node_feat": node_feat,
            "graph_feat": graph_emb,
            "batch": batch,
        }
