import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

try:
    from gvp import GVP, GVPConvLayer, LayerNorm
except Exception as e:
    GVP = None
    GVPConvLayer = None
    LayerNorm = None
    _GVP_IMPORT_ERROR = e
else:
    _GVP_IMPORT_ERROR = None


class Protein3DEncoder(nn.Module):
    """
    真正的 3-layer GVP-GNN 编码器。

    期望输入 data(dict) 至少包含：
    {
        "node_s": Tensor [N, ns_in],
        "node_v": Tensor [N, nv_in, 3],
        "edge_index": LongTensor [2, E],
        "edge_s": Tensor [E, es_in],
        "edge_v": Tensor [E, ev_in, 3],
        "batch": LongTensor [N],
    }

    注意：缺少 edge_* 时不能称为 GVP-GNN，本模块会直接报错。
    """

    def __init__(
        self,
        node_s_dim,
        node_v_dim,
        hidden_dim=256,
        out_dim=None,
        dropout=0.1,
        edge_s_dim=32,
        edge_v_dim=1,
        n_layers=3,
    ):
        super().__init__()

        if GVP is None or GVPConvLayer is None or LayerNorm is None:
            raise ImportError(
                "Failed to import gvp modules. Please install gvp-pytorch (or project gvp dependency) first. "
                f"Original error: {_GVP_IMPORT_ERROR}"
            )

        if n_layers != 3:
            raise ValueError(f"This encoder is configured for strict 3-layer GVP-GNN, got n_layers={n_layers}")

        if out_dim is None:
            out_dim = hidden_dim

        self.node_in_dim = (node_s_dim, node_v_dim)
        self.edge_in_dim = (edge_s_dim, edge_v_dim)
        self.node_h_dim = (hidden_dim, max(1, hidden_dim // 16))

        self.W_v = nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None), vector_gate=True),
        )
        self.W_e = nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.node_h_dim, activations=(None, None), vector_gate=True),
        )

        self.layers = nn.ModuleList(
            [
                GVPConvLayer(
                    self.node_h_dim,
                    self.node_h_dim,
                    drop_rate=dropout,
                    vector_gate=True,
                )
                for _ in range(3)
            ]
        )

        self.W_out = nn.Sequential(
            LayerNorm(self.node_h_dim),
            GVP(self.node_h_dim, (out_dim, 0), activations=(None, None), vector_gate=True),
        )

    def forward(self, data):
        required = ["node_s", "node_v", "edge_index", "edge_s", "edge_v", "batch"]
        missing = [k for k in required if k not in data or data[k] is None]
        if missing:
            raise ValueError(
                "Protein3DEncoder requires full graph inputs for true GVP-GNN. "
                f"Missing keys: {missing}."
            )

        node_s = data["node_s"]
        node_v = data["node_v"]
        edge_index = data["edge_index"]
        edge_s = data["edge_s"]
        edge_v = data["edge_v"]
        batch = data["batch"]

        h_V = (node_s, node_v)
        h_E = (edge_s, edge_v)

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        out_s = self.W_out(h_V)
        graph_emb = global_mean_pool(out_s, batch)
        return graph_emb
