import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_1d_encoder import Drug1DEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_encoder import Protein3DEncoder
from models.decoder import Decoder


class MyModelMDTAP13DPCL(nn.Module):
    def __init__(
        self,
        drug_1d_in_dim=768,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        contrastive_dim=128,
        dropout=0.1,
        temperature_init=0.07,
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

        # projection heads for protein 1D / 3D contrast
        self.proj_1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim),
        )

        self.proj_3d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim),
        )

        # gate fusion replaces simple concat
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32)
        )

    def compute_pcl_loss(self, h1, h3):
        z1 = F.normalize(self.proj_1d(h1), p=2, dim=-1)
        z3 = F.normalize(self.proj_3d(h3), p=2, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * torch.matmul(z1, z3.t())   # [B, B]

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_1to3 = F.cross_entropy(logits, labels)
        loss_3to1 = F.cross_entropy(logits.t(), labels)
        pcl_loss = 0.5 * (loss_1to3 + loss_3to1)

        return pcl_loss, logits

    def fuse_protein(self, h1, h3):
        gate = self.gate_mlp(torch.cat([h1, h3], dim=-1))   # [B, H]
        hp = gate * h1 + (1.0 - gate) * h3
        return hp, gate

    def forward(self, batch, return_aux=True):
        drug_feat = self.drug_1d_encoder(batch["drug_1d"])          # [B, H]
        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])   # [B, H]
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])   # [B, H]

        pcl_loss, pcl_logits = self.compute_pcl_loss(protein_1d_feat, protein_3d_feat)
        protein_feat, gate = self.fuse_protein(protein_1d_feat, protein_3d_feat)

        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        pred = self.decoder(pair_feat)

        if return_aux:
            aux = {
                "pcl_loss": pcl_loss,
                "pcl_logits": pcl_logits,
                "mean_gate": gate.mean().detach(),
                "logit_scale": self.logit_scale.exp().detach(),
            }
            return pred, aux
        else:
            return pred