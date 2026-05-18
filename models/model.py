import torch
import torch.nn as nn

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_2d_encoder import Drug2DEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_2d_encoder import Protein2DEncoder
from models.protein_3d_encoder import Protein3DEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class MyModelMDTA(nn.Module):
    def __init__(
        self,
        drug_1d_in_dim,
        drug_2d_node_dim,
        protein_1d_in_dim,
        protein_2d_node_dim,
        protein_3d_node_s_dim,
        protein_3d_node_v_dim,
        hidden_dim=256,
        dropout=0.1,
        task="classification",
    ):
        super().__init__()

        # ========== Drug encoders ==========
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

        # ========== Protein encoders ==========
        self.protein_1d_encoder = Protein1DEncoder(
            input_dim=protein_1d_in_dim,
            hidden_dim=hidden_dim
        )
        self.protein_2d_encoder = Protein2DEncoder(
            node_in_dim=protein_2d_node_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            heads=4,
            dropout=dropout
        )

        self.protein_3d_encoder = Protein3DEncoder(
            node_s_dim=protein_3d_node_s_dim,
            node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout
        )
        self.protein_fusion_2 = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )

        self.protein_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )

        # ========== Pair decoder ==========
        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task
        )

    def forward(self, batch):
        # Drug branch
        drug_1d_emb = self.drug_1d_encoder(batch["drug_1d"])

        if batch["drug_2d"] is not None:
            drug_2d_emb = self.drug_2d_encoder(batch["drug_2d"])
            drug_feat = self.drug_fusion([drug_1d_emb, drug_2d_emb])
        else:
            drug_feat = drug_1d_emb

        # Protein branch
        protein_1d_emb = self.protein_1d_encoder(batch["protein_1d"])
        protein_feats = [protein_1d_emb]

        if batch["protein_2d"] is not None:
            protein_2d_emb = self.protein_2d_encoder(batch["protein_2d"])
            protein_feats.append(protein_2d_emb)

        if batch["protein_3d"] is not None:
            protein_3d_emb = self.protein_3d_encoder(batch["protein_3d"])
            protein_feats.append(protein_3d_emb)

        if len(protein_feats) == 1:
            protein_feat = protein_feats[0]
        elif len(protein_feats) == 2:
            protein_feat = self.protein_fusion_2(protein_feats)
        else:
            protein_feat = self.protein_fusion(protein_feats)

        # Drug-protein interaction
        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)

        # Prediction
        out = self.decoder(pair_feat)
        return out

    # def forward(self, batch):
    #     # Drug branch
    #     drug_1d_emb = self.drug_1d_encoder(batch["drug_1d"])
    #     drug_2d_emb = self.drug_2d_encoder(batch["drug_2d"])
    #     drug_feat = self.drug_fusion([drug_1d_emb, drug_2d_emb])
    #
    #     # Protein branch
    #     protein_1d_emb = self.protein_1d_encoder(batch["protein_1d"])
    #     protein_2d_emb = self.protein_2d_encoder(batch["protein_2d"])
    #     protein_3d_emb = self.protein_3d_encoder(batch["protein_3d"])
    #     protein_feat = self.protein_fusion([protein_1d_emb, protein_2d_emb, protein_3d_emb])
    #
    #     # Drug-protein interaction
    #     pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
    #
    #     # Prediction
    #     out = self.decoder(pair_feat)
    #     return out