# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_3d_egnn_encoder import Drug3DEGNNEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder
from models.decoder import Decoder

#更改点：将模态间的PCL更新为attention机制
class ModalityInteractionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.modality_embed = nn.Parameter(torch.empty(1, 2, hidden_dim))
        nn.init.normal_(self.modality_embed, mean=0.0, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, feat_1d: torch.Tensor, feat_3d: torch.Tensor, return_attn: bool = True):
        #把 1D 和 3D 当成“两个token”
        tokens = torch.stack([feat_1d, feat_3d], dim=1)
        tokens = tokens + self.modality_embed
        #让这两个token相互注意
        attn_out, attn_weights = self.attn(
            tokens,
            tokens,
            tokens,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        tokens = self.norm1(tokens + self.dropout(attn_out))
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + self.dropout(ffn_out))
        h1_inter = tokens[:, 0, :]
        h3_inter = tokens[:, 1, :]
        gate_input = torch.cat(
            [h1_inter, h3_inter, torch.abs(h1_inter - h3_inter), h1_inter * h3_inter],
            dim=-1,
        )
        gate = self.gate_mlp(gate_input)
        fused = gate * h1_inter + (1.0 - gate) * h3_inter
        fused = self.out_norm(fused)
        return fused, gate, attn_weights if return_attn else None, h1_inter, h3_inter


class MyModelMDTAP13DSOFTCLIPATTENTION(nn.Module):
    """
    保留 softclip 策略，只将 PCL 替换为 modality interaction self-attention。
    模态：drug(1D+3D) + protein(1D+3D)
    """

    def __init__(
        self,
        drug_1d_in_dim=768,
        drug_3d_node_in_dim=10,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        contrastive_dim=128,
        dropout=0.1,
        task="regression",
        temperature_init=0.07,
        affinity_temperature=1.0,
        labelsim_tau=1.0,
        labelsim_mix=0.3,
        attn_heads=4,
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
        self.protein_1d_encoder = Protein1DEncoder(input_dim=protein_1d_in_dim, hidden_dim=hidden_dim)
        self.protein_3d_encoder = Protein3DEGNNEncoder(
            node_s_dim=protein_3d_node_s_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
            n_layers=3,
        )

        # softclip 保留
        self.drug_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, contrastive_dim))
        self.protein_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, contrastive_dim))
        self.logit_scale_softclip = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32))
        self.affinity_temperature = affinity_temperature
        self.labelsim_tau = labelsim_tau
        self.labelsim_mix = labelsim_mix

        # PCL 替换为 attention 融合
        self.drug_fusion = ModalityInteractionFusion(hidden_dim=hidden_dim, num_heads=attn_heads, dropout=dropout)
        self.protein_fusion = ModalityInteractionFusion(hidden_dim=hidden_dim, num_heads=attn_heads, dropout=dropout)

        self.decoder = Decoder(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, dropout=dropout, task=task)

    @staticmethod
    def masked_softmax(x, mask, dim=-1):
        x = x.masked_fill(~mask, -1e9)
        prob = F.softmax(x, dim=dim)
        prob = prob * mask.float()
        prob = prob / prob.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        return prob

    def compute_label_similarity(self, affinity_matrix, affinity_mask):
        diag_vals = torch.diag(affinity_matrix)
        diff = torch.abs(diag_vals.unsqueeze(1) - diag_vals.unsqueeze(0))
        tau = max(1e-6, float(self.labelsim_tau))
        sim = torch.exp(-diff / tau)
        valid = torch.diag(affinity_mask).bool()
        pair_mask = valid.unsqueeze(1) & valid.unsqueeze(0)
        sim = sim * pair_mask.float()
        sim = sim / sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return sim

    def compute_softclip_loss(self, drug_feat, protein_feat, affinity_matrix, affinity_mask):
        z_d = F.normalize(self.drug_proj(drug_feat), p=2, dim=-1)
        z_p = F.normalize(self.protein_proj(protein_feat), p=2, dim=-1)
        logit_scale = self.logit_scale_softclip.exp().clamp(max=100.0)
        logits = logit_scale * torch.matmul(z_d, z_p.t())
        masked_logits = logits.masked_fill(~affinity_mask, -1e9)

        target_d2p_base = self.masked_softmax(affinity_matrix / self.affinity_temperature, affinity_mask, dim=1)
        target_p2d_base = self.masked_softmax(affinity_matrix.t() / self.affinity_temperature, affinity_mask.t(), dim=1)

        label_sim = self.compute_label_similarity(affinity_matrix, affinity_mask)
        target_d2p_refine = torch.matmul(target_d2p_base, label_sim.t())
        target_p2d_refine = torch.matmul(target_p2d_base, label_sim.t())

        mix = min(1.0, max(0.0, float(self.labelsim_mix)))
        target_d2p = (1.0 - mix) * target_d2p_base + mix * target_d2p_refine
        target_d2p = target_d2p * affinity_mask.float()
        target_d2p = target_d2p / target_d2p.sum(dim=1, keepdim=True).clamp_min(1e-12)

        target_p2d = (1.0 - mix) * target_p2d_base + mix * target_p2d_refine
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
        drug_1d_feat = self.drug_1d_encoder(batch["drug_1d"])
        drug_3d_feat = self.drug_3d_encoder(batch["drug_3d"])
        drug_feat, drug_gate, drug_attn_weights, drug_1d_inter, drug_3d_inter = self.drug_fusion(
            drug_1d_feat,
            drug_3d_feat,
            return_attn=return_aux,
        )

        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])
        protein_feat, protein_gate, protein_attn_weights, protein_1d_inter, protein_3d_inter = self.protein_fusion(
            protein_1d_feat,
            protein_3d_feat,
            return_attn=return_aux,
        )

        if affinity_matrix is not None and affinity_mask is not None:
            clip_loss, clip_logits, mean_valid_per_row, mean_valid_offdiag, mean_diag_prior_mass = self.compute_softclip_loss(drug_feat, protein_feat, affinity_matrix, affinity_mask)
        else:
            clip_loss = torch.tensor(0.0, device=drug_feat.device)
            clip_logits = None
            mean_valid_per_row = torch.tensor(0.0, device=drug_feat.device)
            mean_valid_offdiag = torch.tensor(0.0, device=drug_feat.device)
            mean_diag_prior_mass = torch.tensor(0.0, device=drug_feat.device)

        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        pred = self.decoder(pair_feat)
        if return_aux:
            if drug_attn_weights is not None:
                drug_attn_mean = drug_attn_weights.detach().mean(dim=(0, 1))
            else:
                drug_attn_mean = torch.zeros(2, 2, device=drug_feat.device)
            if protein_attn_weights is not None:
                protein_attn_mean = protein_attn_weights.detach().mean(dim=(0, 1))
            else:
                protein_attn_mean = torch.zeros(2, 2, device=drug_feat.device)
            aux = {
                "clip_loss": clip_loss,
                "clip_logits": clip_logits,
                "mean_gate": ((drug_gate.mean() + protein_gate.mean()) / 2).detach(),
                "mean_drug_gate": drug_gate.mean().detach(),
                "mean_protein_gate": protein_gate.mean().detach(),
                "mean_valid_per_row": mean_valid_per_row.detach(),
                "mean_valid_offdiag": mean_valid_offdiag.detach(),
                "mean_diag_prior_mass": mean_diag_prior_mass.detach(),
                "labelsim_tau": torch.tensor(self.labelsim_tau, device=drug_feat.device),
                "labelsim_mix": torch.tensor(self.labelsim_mix, device=drug_feat.device),
                "logit_scale_softclip": self.logit_scale_softclip.exp().detach(),
                "drug_attn_1d_to_1d": drug_attn_mean[0, 0].detach(),
                "drug_attn_1d_to_3d": drug_attn_mean[0, 1].detach(),
                "drug_attn_3d_to_1d": drug_attn_mean[1, 0].detach(),
                "drug_attn_3d_to_3d": drug_attn_mean[1, 1].detach(),
                "protein_attn_1d_to_1d": protein_attn_mean[0, 0].detach(),
                "protein_attn_1d_to_3d": protein_attn_mean[0, 1].detach(),
                "protein_attn_3d_to_1d": protein_attn_mean[1, 0].detach(),
                "protein_attn_3d_to_3d": protein_attn_mean[1, 1].detach(),
                "drug_feat_norm": drug_feat.norm(dim=-1).mean().detach(),
                "protein_feat_norm": protein_feat.norm(dim=-1).mean().detach(),
                "drug_1d_inter_norm": drug_1d_inter.norm(dim=-1).mean().detach(),
                "drug_3d_inter_norm": drug_3d_inter.norm(dim=-1).mean().detach(),
                "protein_1d_inter_norm": protein_1d_inter.norm(dim=-1).mean().detach(),
                "protein_3d_inter_norm": protein_3d_inter.norm(dim=-1).mean().detach(),
            }
            return pred, aux
        return pred
