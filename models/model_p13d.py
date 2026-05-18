# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models.drug_1d_encoder import Drug1DEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_encoder import Protein3DEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class MyModelMDTAP13D(nn.Module):
    """
    只用:
    - drug_1d
    - protein_1d
    - protein_3d
    """
    def __init__(
        self,
        drug_1d_in_dim=768,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        dropout=0.1,
        task="regression",
    ):
        super().__init__()

        self.drug_1d_encoder = Drug1DEncoder(
            input_dim=drug_1d_in_dim,
            hidden_dim=hidden_dim
        )

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

        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task
        )

    def forward(self, batch):
        drug_feat = self.drug_1d_encoder(batch["drug_1d"])            # [B, 128]
        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"]) # [B, 128]
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"]) # [B, 128]

        protein_feat = self.protein_fusion([protein_1d_feat, protein_3d_feat])

        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        out = self.decoder(pair_feat)
        return out