# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_1d_encoder import Drug1DEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_encoder import Protein3DEncoder
from models.decoder import Decoder


class MyModelMDTAP13DSOFTCLIPPCL(nn.Module):
    """
    drug_1d + protein_1d + protein_3d
    dual contrastive design:
    1) affinity-aware soft CLIP between drug and fused protein
    2) PCL between protein 1D and protein 3D
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
        pcl_temperature_init=None,
        diag_prior_weight=0.5,
    ):
        super().__init__()

        self.drug_1d_encoder = Drug1DEncoder(input_dim=drug_1d_in_dim, hidden_dim=hidden_dim)
        self.protein_1d_encoder = Protein1DEncoder(input_dim=protein_1d_in_dim, hidden_dim=hidden_dim)
        self.protein_3d_encoder = Protein3DEncoder(
            node_s_dim=protein_3d_node_s_dim,
            node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        self.proj_1d = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, contrastive_dim))
        self.proj_3d = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, contrastive_dim))
        self.gate_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.drug_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, contrastive_dim))
        self.protein_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, contrastive_dim))

        self.decoder = Decoder(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, dropout=dropout, task=task)
        self.logit_scale_softclip = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32))
        if pcl_temperature_init is None:
            pcl_temperature_init = temperature_init
        self.logit_scale_pcl = nn.Parameter(torch.tensor(math.log(1.0 / pcl_temperature_init), dtype=torch.float32))
        self.affinity_temperature = affinity_temperature
        self.diag_prior_weight = diag_prior_weight

    def fuse_protein(self, h1, h3):
        gate = self.gate_mlp(torch.cat([h1, h3], dim=-1))
        hp = gate * h1 + (1.0 - gate) * h3
        return hp, gate

    def compute_pcl_loss(self, h1, h3):
        z1 = F.normalize(self.proj_1d(h1), p=2, dim=-1)
        z3 = F.normalize(self.proj_3d(h3), p=2, dim=-1)
        logit_scale = self.logit_scale_pcl.exp().clamp(max=100.0)
        logits = logit_scale * torch.matmul(z1, z3.t())
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_1to3 = F.cross_entropy(logits, labels)
        loss_3to1 = F.cross_entropy(logits.t(), labels)
        pcl_loss = 0.5 * (loss_1to3 + loss_3to1)
        return pcl_loss, logits

    @staticmethod
    def masked_softmax(x, mask, dim=-1):
        x = x.masked_fill(~mask, -1e9)
        prob = F.softmax(x, dim=dim)
        prob = prob * mask.float()
        prob = prob / prob.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        return prob

    def compute_softclip_loss(self, drug_feat, protein_feat, affinity_matrix, affinity_mask):
        z_d = F.normalize(self.drug_proj(drug_feat), p=2, dim=-1)
        z_p = F.normalize(self.protein_proj(protein_feat), p=2, dim=-1)
        logit_scale = self.logit_scale_softclip.exp().clamp(max=100.0)
        logits = logit_scale * torch.matmul(z_d, z_p.t())
        masked_logits = logits.masked_fill(~affinity_mask, -1e9)

        # affinity soft target
        target_d2p_aff = self.masked_softmax(affinity_matrix / self.affinity_temperature, affinity_mask, dim=1)
        target_p2d_aff = self.masked_softmax(affinity_matrix.t() / self.affinity_temperature, affinity_mask.t(), dim=1)

        # diagonal one-hot prior
        B = affinity_matrix.size(0)
        eye = torch.eye(B, device=affinity_matrix.device, dtype=affinity_matrix.dtype)
        diag_mask = eye.bool() & affinity_mask
        diag_prior_d2p = diag_mask.float()
        diag_prior_d2p = diag_prior_d2p / diag_prior_d2p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        diag_prior_p2d = diag_prior_d2p.t()

        w = float(self.diag_prior_weight)
        w = max(0.0, min(1.0, w))

        # blend affinity target with diagonal prior, then renormalize
        target_d2p = (1.0 - w) * target_d2p_aff + w * diag_prior_d2p
        target_d2p = target_d2p * affinity_mask.float()
        target_d2p = target_d2p / target_d2p.sum(dim=1, keepdim=True).clamp_min(1e-12)

        target_p2d = (1.0 - w) * target_p2d_aff + w * diag_prior_p2d
        target_p2d = target_p2d * affinity_mask.t().float()
        target_p2d = target_p2d / target_p2d.sum(dim=1, keepdim=True).clamp_min(1e-12)

        pred_logprob_d2p = F.log_softmax(masked_logits, dim=1)
        loss_d2p = -(target_d2p * pred_logprob_d2p).sum(dim=1).mean()

        pred_logprob_p2d = F.log_softmax(masked_logits.t(), dim=1)
        loss_p2d = -(target_p2d * pred_logprob_p2d).sum(dim=1).mean()

        softclip_loss = 0.5 * (loss_d2p + loss_p2d)
        mean_valid_per_row = affinity_mask.sum(dim=1).float().mean()
        mean_valid_offdiag = (affinity_mask.sum(dim=1) - 1).float().mean()
        mean_diag_prior_mass = torch.diag(target_d2p).mean()
        return softclip_loss, masked_logits, mean_valid_per_row, mean_valid_offdiag, mean_diag_prior_mass

    def forward(self, batch, affinity_matrix=None, affinity_mask=None, return_aux=True):
        drug_feat = self.drug_1d_encoder(batch["drug_1d"])
        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])
        pcl_loss, pcl_logits = self.compute_pcl_loss(protein_1d_feat, protein_3d_feat)
        protein_feat, gate = self.fuse_protein(protein_1d_feat, protein_3d_feat)

        if affinity_matrix is not None and affinity_mask is not None:
            clip_loss, clip_logits, mean_valid_per_row, mean_valid_offdiag, mean_diag_prior_mass = self.compute_softclip_loss(drug_feat, protein_feat, affinity_matrix, affinity_mask)
        else:
            clip_loss = torch.tensor(0.0, device=protein_feat.device)
            clip_logits = None
            mean_valid_per_row = torch.tensor(0.0, device=protein_feat.device)
            mean_valid_offdiag = torch.tensor(0.0, device=protein_feat.device)
            mean_diag_prior_mass = torch.tensor(0.0, device=protein_feat.device)

        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        pred = self.decoder(pair_feat)
        if return_aux:
            aux = {
                "clip_loss": clip_loss,
                "clip_logits": clip_logits,
                "pcl_loss": pcl_loss,
                "pcl_logits": pcl_logits,
                "mean_gate": gate.mean().detach(),
                "mean_valid_per_row": mean_valid_per_row.detach(),
                "mean_valid_offdiag": mean_valid_offdiag.detach(),
                "mean_diag_prior_mass": mean_diag_prior_mass.detach(),
                "diag_prior_weight": torch.tensor(self.diag_prior_weight, device=protein_feat.device),
                "logit_scale_softclip": self.logit_scale_softclip.exp().detach(),
                "logit_scale_pcl": self.logit_scale_pcl.exp().detach(),
            }
            return pred, aux
        return pred
