import torch
import torch.nn as nn

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_2d_encoder import Drug2DEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_2d_encoder import Protein2DEncoder
from models.protein_3d_encoder import Protein3DEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class MyModelMDTAAblation(nn.Module):
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
        use_drug_1d=True,
        use_drug_2d=True,
        use_protein_1d=True,
        use_protein_2d=True,
        use_protein_3d=False,
    ):
        super().__init__()

        self.use_drug_1d = use_drug_1d
        self.use_drug_2d = use_drug_2d
        self.use_protein_1d = use_protein_1d
        self.use_protein_2d = use_protein_2d
        self.use_protein_3d = use_protein_3d

        if not (self.use_drug_1d or self.use_drug_2d):
            raise ValueError("至少要启用一个 DRUG 分支")
        if not (self.use_protein_1d or self.use_protein_2d or self.use_protein_3d):
            raise ValueError("至少要启用一个 PROTEIN 分支")

        # ========== Drug encoders ==========
        self.drug_1d_encoder = None
        self.drug_2d_encoder = None
        self.drug_fusion = None

        if self.use_drug_1d:
            self.drug_1d_encoder = Drug1DEncoder(
                input_dim=drug_1d_in_dim,
                hidden_dim=hidden_dim
            )

        if self.use_drug_2d:
            self.drug_2d_encoder = Drug2DEncoder(
                node_in_dim=drug_2d_node_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                heads=4,
                dropout=dropout
            )

        if self.use_drug_1d and self.use_drug_2d:
            self.drug_fusion = ConcatFusion(
                input_dims=[hidden_dim, hidden_dim],
                out_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                dropout=dropout
            )

        # ========== Protein encoders ==========
        self.protein_1d_encoder = None
        self.protein_2d_encoder = None
        self.protein_3d_encoder = None
        self.protein_fusion_2 = None
        self.protein_fusion_3 = None

        if self.use_protein_1d:
            self.protein_1d_encoder = Protein1DEncoder(
                input_dim=protein_1d_in_dim,
                hidden_dim=hidden_dim
            )

        if self.use_protein_2d:
            self.protein_2d_encoder = Protein2DEncoder(
                node_in_dim=protein_2d_node_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                heads=4,
                dropout=dropout
            )

        if self.use_protein_3d:
            self.protein_3d_encoder = Protein3DEncoder(
                node_s_dim=protein_3d_node_s_dim,
                node_v_dim=protein_3d_node_v_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout
            )

        num_protein_branches = int(self.use_protein_1d) + int(self.use_protein_2d) + int(self.use_protein_3d)

        if num_protein_branches >= 2:
            self.protein_fusion_2 = ConcatFusion(
                input_dims=[hidden_dim, hidden_dim],
                out_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                dropout=dropout
            )

        if num_protein_branches >= 3:
            self.protein_fusion_3 = ConcatFusion(
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
        drug_feats = []

        if self.use_drug_1d and batch["drug_1d"] is not None:
            drug_1d_emb = self.drug_1d_encoder(batch["drug_1d"])
            drug_feats.append(drug_1d_emb)

        if self.use_drug_2d and batch["drug_2d"] is not None:
            drug_2d_emb = self.drug_2d_encoder(batch["drug_2d"])
            drug_feats.append(drug_2d_emb)

        if len(drug_feats) == 0:
            raise ValueError("当前 BATCH 没有可用的 DRUG 特征")
        elif len(drug_feats) == 1:
            drug_feat = drug_feats[0]
        else:
            drug_feat = self.drug_fusion(drug_feats)

        # Protein branch
        protein_feats = []

        if self.use_protein_1d and batch["protein_1d"] is not None:
            protein_1d_emb = self.protein_1d_encoder(batch["protein_1d"])
            protein_feats.append(protein_1d_emb)

        if self.use_protein_2d and batch["protein_2d"] is not None:
            protein_2d_emb = self.protein_2d_encoder(batch["protein_2d"])
            protein_feats.append(protein_2d_emb)

        if self.use_protein_3d and batch["protein_3d"] is not None:
            protein_3d_emb = self.protein_3d_encoder(batch["protein_3d"])
            protein_feats.append(protein_3d_emb)

        if len(protein_feats) == 0:
            raise ValueError("当前 BATCH 没有可用的 PROTEIN 特征")
        elif len(protein_feats) == 1:
            protein_feat = protein_feats[0]
        elif len(protein_feats) == 2:
            protein_feat = self.protein_fusion_2(protein_feats)
        else:
            protein_feat = self.protein_fusion_3(protein_feats)

        # Pair interaction
        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)

        # Prediction
        out = self.decoder(pair_feat)
        return out