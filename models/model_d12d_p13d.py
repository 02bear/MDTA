# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_2d_encoder import Drug2DEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_encoder import Protein3DEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class MyModelMDTAD12DP13D(nn.Module):
    """
    Drug:    1D + 2D
    Protein: 1D + 3D
    无对比学习，无其他额外处理
    """

    def __init__(
        self,
        drug_1d_in_dim=768,
        drug_2d_node_dim=43,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        dropout=0.1,
        task="regression",
    ):
        super().__init__()

        # -------- Drug encoders --------
        self.drug_1d_encoder = Drug1DEncoder(
            input_dim=drug_1d_in_dim,
            hidden_dim=hidden_dim
        )

        self.drug_2d_encoder = Drug2DEncoder(
            node_in_dim=drug_2d_node_dim,
            hidden_dim=hidden_dim // 2,
            out_dim=hidden_dim,
            heads=4,
            dropout=dropout
        )

        self.drug_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )

        # -------- Protein encoders --------
        self.protein_1d_encoder = Protein1DEncoder(
            input_dim=protein_1d_in_dim,
            hidden_dim=hidden_dim
        )

        self.protein_3d_encoder = Protein3DEncoder(
            node_s_dim=protein_3d_node_s_dim,
            node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout
        )

        self.protein_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )

        # -------- Pair decoder --------
        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task
        )

    def forward(self, batch):
        # Drug branch
        drug_1d_feat = self.drug_1d_encoder(batch["drug_1d"])     # [B, H]
        drug_2d_feat = self.drug_2d_encoder(batch["drug_2d"])     # [B, H]
        drug_feat = self.drug_fusion([drug_1d_feat, drug_2d_feat])  # [B, H]

        # Protein branch
        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])  # [B, H]
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])  # [B, H]
        protein_feat = self.protein_fusion([protein_1d_feat, protein_3d_feat])  # [B, H]

        # Pair prediction
        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)  # [B, 2H]
        out = self.decoder(pair_feat)
        return out