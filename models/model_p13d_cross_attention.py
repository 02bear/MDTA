# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_3d_egnn_encoder import Drug3DEGNNEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class FourModalCrossAttention(nn.Module):
    """
    只在四个 encoder 输出之后、原始 ConcatFusion 之前加入一层双向交叉注意力。

    输入:
        drug_1d_feat:    [B, H]
        drug_3d_feat:    [B, H]
        protein_1d_feat: [B, H]
        protein_3d_feat: [B, H]

    交互:
        drug tokens    = [drug_1d, drug_3d]       作为 query，关注 protein tokens
        protein tokens = [protein_1d, protein_3d] 作为 query，关注 drug tokens

    输出:
        更新后的四个模态特征，形状均为 [B, H]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # hidden_dim=128 时，4 heads 每个 head 维度为 32。
        # 不新增命令行参数，保持训练脚本和原始模型参数不变。
        num_heads = 4
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )

        self.drug_to_protein_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.protein_to_drug_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        drug_1d_feat: torch.Tensor,
        drug_3d_feat: torch.Tensor,
        protein_1d_feat: torch.Tensor,
        protein_3d_feat: torch.Tensor,
    ):
        drug_tokens = torch.stack([drug_1d_feat, drug_3d_feat], dim=1)              # [B, 2, H]
        protein_tokens = torch.stack([protein_1d_feat, protein_3d_feat], dim=1)    # [B, 2, H]

        drug_context, _ = self.drug_to_protein_attn(
            query=drug_tokens,
            key=protein_tokens,
            value=protein_tokens,
            need_weights=False,
        )
        protein_context, _ = self.protein_to_drug_attn(
            query=protein_tokens,
            key=drug_tokens,
            value=drug_tokens,
            need_weights=False,
        )

        drug_tokens = self.drug_norm(drug_tokens + drug_context)
        protein_tokens = self.protein_norm(protein_tokens + protein_context)

        drug_1d_feat = drug_tokens[:, 0, :]
        drug_3d_feat = drug_tokens[:, 1, :]
        protein_1d_feat = protein_tokens[:, 0, :]
        protein_3d_feat = protein_tokens[:, 1, :]

        return drug_1d_feat, drug_3d_feat, protein_1d_feat, protein_3d_feat


class MyModelMDTAP13D(nn.Module):
    """
    使用:
    - drug_1d
    - drug_3d (EGNN, 3层)
    - protein_1d
    - protein_3d

    与原始 baseline 保持一致:
    - 四个 encoder 不变
    - drug 两模态仍然使用原始 ConcatFusion
    - protein 两模态仍然使用原始 ConcatFusion
    - 最后仍然拼接 drug_feat 和 protein_feat 后送入 Decoder

    唯一新增:
    - 在四个 encoder 输出之后、ConcatFusion 之前加入一层 FourModalCrossAttention
    """

    def __init__(
        self,
        drug_1d_in_dim=768,
        drug_3d_node_in_dim=10,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        dropout=0.1,
        task="regression",
    ):
        super().__init__()

        self.drug_1d_encoder = Drug1DEncoder(input_dim=drug_1d_in_dim, hidden_dim=hidden_dim)
        self.drug_3d_encoder = Drug3DEGNNEncoder(
            node_in_dim=drug_3d_node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=3,
            dropout=dropout,
        )
        self.drug_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout,
        )

        self.protein_1d_encoder = Protein1DEncoder(input_dim=protein_1d_in_dim, hidden_dim=hidden_dim)
        self.protein_3d_encoder = Protein3DEGNNEncoder(
            node_s_dim=protein_3d_node_s_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
            n_layers=3,
        )
        self.protein_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout,
        )

        self.cross_attention = FourModalCrossAttention(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
        )

    def forward(self, batch):
        drug_1d_feat = self.drug_1d_encoder(batch["drug_1d"])          # [B, H]
        drug_3d_feat = self.drug_3d_encoder(batch["drug_3d"])          # [B, H]

        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"]) # [B, H]
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"]) # [B, H]

        drug_1d_feat, drug_3d_feat, protein_1d_feat, protein_3d_feat = self.cross_attention(
            drug_1d_feat=drug_1d_feat,
            drug_3d_feat=drug_3d_feat,
            protein_1d_feat=protein_1d_feat,
            protein_3d_feat=protein_3d_feat,
        )

        drug_feat = self.drug_fusion([drug_1d_feat, drug_3d_feat])     # [B, H]
        protein_feat = self.protein_fusion([protein_1d_feat, protein_3d_feat])

        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        out = self.decoder(pair_feat)
        return out
