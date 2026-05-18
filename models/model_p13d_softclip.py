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


class MyModelMDTAP13DSOFTCLIP(nn.Module):
    """
    drug_1d + protein_1d + protein_3d
    使用 affinity-aware soft CLIP：
    - 不再把对角线当唯一正样本
    - 用 batch 内真实 affinity 子矩阵构造 soft targets
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
        affinity_temperature=1.0,
    ):
        super().__init__()

        # -------- Encoders --------
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

        # -------- Decoder --------
        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task
        )

        # -------- Soft-CLIP projection heads --------
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

        # learnable similarity scale
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32)
        )

        # target soft distribution temperature
        self.affinity_temperature = affinity_temperature

    @staticmethod
    def masked_softmax(x, mask, dim=-1):
        """
        x:    [B, B]
        mask: [B, B] bool
        """
        x = x.masked_fill(~mask, -1e9)
        prob = F.softmax(x, dim=dim)
        prob = prob * mask.float()
        prob = prob / prob.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        return prob

    def compute_softclip_loss(self, drug_feat, protein_feat, affinity_matrix, affinity_mask):
        """
        drug_feat:       [B, H]
        protein_feat:    [B, H]
        affinity_matrix: [B, B]  当前 batch 中 (drug_i, protein_j) 的真实 affinity
        affinity_mask:   [B, B]  该位置是否在当前 split lookup 中可用
        """
        z_d = F.normalize(self.drug_proj(drug_feat), p=2, dim=-1)       # [B, C]
        z_p = F.normalize(self.protein_proj(protein_feat), p=2, dim=-1) # [B, C]

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * torch.matmul(z_d, z_p.t())  # [B, B]

        # 只在已知 affinity 的位置上比较
        masked_logits = logits.masked_fill(~affinity_mask, -1e9)

        # soft targets: drug -> protein
        target_d2p = self.masked_softmax(
            affinity_matrix / self.affinity_temperature,
            affinity_mask,
            dim=1
        )
        pred_logprob_d2p = F.log_softmax(masked_logits, dim=1)
        loss_d2p = -(target_d2p * pred_logprob_d2p).sum(dim=1).mean()

        # soft targets: protein -> drug
        target_p2d = self.masked_softmax(
            affinity_matrix.t() / self.affinity_temperature,
            affinity_mask.t(),
            dim=1
        )
        pred_logprob_p2d = F.log_softmax(masked_logits.t(), dim=1)
        loss_p2d = -(target_p2d * pred_logprob_p2d).sum(dim=1).mean()

        softclip_loss = 0.5 * (loss_d2p + loss_p2d)

        mean_valid_per_row = affinity_mask.sum(dim=1).float().mean()
        mean_valid_offdiag = (affinity_mask.sum(dim=1) - 1).float().mean()

        return softclip_loss, masked_logits, mean_valid_per_row, mean_valid_offdiag

    def forward(self, batch, affinity_matrix=None, affinity_mask=None, return_aux=True):
        # -------- main features --------
        drug_feat = self.drug_1d_encoder(batch["drug_1d"])                 # [B, H]

        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])     # [B, H]
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])     # [B, H]
        protein_feat = self.protein_fusion([protein_1d_feat, protein_3d_feat])  # [B, H]

        # -------- regression prediction --------
        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        pred = self.decoder(pair_feat)

        aux = {}

        if affinity_matrix is not None and affinity_mask is not None:
            softclip_loss, softclip_logits, mean_valid_per_row, mean_valid_offdiag = self.compute_softclip_loss(
                drug_feat=drug_feat,
                protein_feat=protein_feat,
                affinity_matrix=affinity_matrix,
                affinity_mask=affinity_mask,
            )
            aux["clip_loss"] = softclip_loss
            aux["clip_logits"] = softclip_logits
            aux["mean_valid_per_row"] = mean_valid_per_row.detach()
            aux["mean_valid_offdiag"] = mean_valid_offdiag.detach()
        else:
            aux["clip_loss"] = torch.tensor(0.0, device=pred.device)
            aux["clip_logits"] = None
            aux["mean_valid_per_row"] = torch.tensor(0.0, device=pred.device)
            aux["mean_valid_offdiag"] = torch.tensor(0.0, device=pred.device)

        aux["logit_scale"] = self.logit_scale.exp().detach()
        aux["drug_feat"] = drug_feat
        aux["protein_feat"] = protein_feat

        if return_aux:
            return pred, aux
        else:
            return pred