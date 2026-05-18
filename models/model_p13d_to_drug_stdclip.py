# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_1d_encoder import Drug1DEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_encoder import Protein3DEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class MyModelMDTAP13DToDrugSTDCLIP(nn.Module):
    """
    drug_1d + protein_1d + protein_3d
    Standard CLIP between drug and protein:
    - positive pairs: diagonal (i, i)
    - negative pairs: all off-diagonal pairs
    """

    def __init__(
        self,
        drug_1d_in_dim=768,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        contrastive_dim=128,
        dropout=0.1,
        task="regression",
        temperature_init=0.07,
    ):
        super().__init__()

        self.drug_1d_encoder = Drug1DEncoder(
            input_dim=drug_1d_in_dim,
            hidden_dim=hidden_dim,
        )

        self.protein_1d_encoder = Protein1DEncoder(
            input_dim=protein_1d_in_dim,
            hidden_dim=hidden_dim,
        )

        self.protein_3d_encoder = Protein3DEncoder(
            node_s_dim=protein_3d_node_s_dim,
            node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        self.protein_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout,
        )

        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
        )

        self.drug_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim),
        )

        self.protein_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim),
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32)
        )

    def compute_std_clip_loss(self, drug_feat, protein_feat):
        z_d = F.normalize(self.drug_proj(drug_feat), p=2, dim=-1)
        z_p = F.normalize(self.protein_proj(protein_feat), p=2, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * torch.matmul(z_d, z_p.t())

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_d2p = F.cross_entropy(logits, labels)
        loss_p2d = F.cross_entropy(logits.t(), labels)
        clip_loss = 0.5 * (loss_d2p + loss_p2d)

        return clip_loss, logits

    def forward(self, batch, return_aux=True):
        drug_feat = self.drug_1d_encoder(batch["drug_1d"])

        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])
        protein_feat = self.protein_fusion([protein_1d_feat, protein_3d_feat])

        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        pred = self.decoder(pair_feat)

        clip_loss, clip_logits = self.compute_std_clip_loss(drug_feat, protein_feat)

        if return_aux:
            aux = {
                "clip_loss": clip_loss,
                "clip_logits": clip_logits,
                "logit_scale": self.logit_scale.exp().detach(),
                "drug_feat": drug_feat,
                "protein_feat": protein_feat,
            }
            return pred, aux
        return pred
