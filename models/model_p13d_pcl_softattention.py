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


class MyModelMDTAP13DPCLSOFTATTENTION(nn.Module):
    """
    PCL + batch-level bidirectional drug-protein soft attention
    + label-similarity refined soft target.

    Modalities:
      drug:    1D + 3D
      protein: 1D + 3D
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
        pcl_temperature_init=None,
        drug_pcl_temperature_init=None,
        labelsim_tau=1.0,
        labelsim_mix=0.3,
    ):
        super().__init__()

        self.drug_1d_encoder = Drug1DEncoder(
            input_dim=drug_1d_in_dim,
            hidden_dim=hidden_dim,
        )
        self.drug_3d_encoder = Drug3DEGNNEncoder(
            node_in_dim=drug_3d_node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=3,
            dropout=dropout,
        )

        self.protein_1d_encoder = Protein1DEncoder(
            input_dim=protein_1d_in_dim,
            hidden_dim=hidden_dim,
        )
        self.protein_3d_encoder = Protein3DEGNNEncoder(
            node_s_dim=protein_3d_node_s_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
            n_layers=3,
        )

        # Protein PCL projection heads.
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

        # Drug PCL projection heads.
        self.drug_proj_1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim),
        )
        self.drug_proj_3d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim),
        )

        # 1D/3D modality fusion gates.
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.drug_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Query/key projections for batch-level drug-protein attention.
        # Scores are computed in contrastive_dim space.
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

        self.attn_dim = contrastive_dim
        self.attn_scale = math.sqrt(float(self.attn_dim))

        # Values are aggregated in hidden_dim space, so FFN/decoder stay compatible.
        self.drug_attn_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.protein_attn_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.drug_attn_norm = nn.LayerNorm(hidden_dim)
        self.protein_attn_norm = nn.LayerNorm(hidden_dim)

        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
        )

        if pcl_temperature_init is None:
            pcl_temperature_init = temperature_init
        if drug_pcl_temperature_init is None:
            drug_pcl_temperature_init = temperature_init

        self.logit_scale_pcl = nn.Parameter(
            torch.tensor(math.log(1.0 / pcl_temperature_init), dtype=torch.float32)
        )
        self.logit_scale_drug_pcl = nn.Parameter(
            torch.tensor(math.log(1.0 / drug_pcl_temperature_init), dtype=torch.float32)
        )

        self.affinity_temperature = affinity_temperature
        self.labelsim_tau = labelsim_tau
        self.labelsim_mix = labelsim_mix

    def fuse_protein(self, h1, h3):
        gate = self.gate_mlp(torch.cat([h1, h3], dim=-1))
        hp = gate * h1 + (1.0 - gate) * h3
        return hp, gate

    def fuse_drug(self, h1, h3):
        gate = self.drug_gate_mlp(torch.cat([h1, h3], dim=-1))
        hd = gate * h1 + (1.0 - gate) * h3
        return hd, gate

    def compute_pcl_loss(self, h1, h3, logit_scale_param, proj_1d, proj_3d):
        z1 = F.normalize(proj_1d(h1), p=2, dim=-1)
        z3 = F.normalize(proj_3d(h3), p=2, dim=-1)

        logit_scale = logit_scale_param.exp().clamp(max=100.0)
        logits = logit_scale * torch.matmul(z1, z3.t())

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_1to3 = F.cross_entropy(logits, labels)
        loss_3to1 = F.cross_entropy(logits.t(), labels)

        pcl_loss = 0.5 * (loss_1to3 + loss_3to1)
        return pcl_loss, logits

    @staticmethod
    def masked_softmax(x, mask, dim=-1, eps=1e-12):
        """
        Softmax over valid mask positions only.
        Invalid positions are exactly 0.
        """
        x = x.masked_fill(~mask, -1e9)
        prob = F.softmax(x, dim=dim)
        prob = prob.masked_fill(~mask, 0.0)
        return prob / prob.sum(dim=dim, keepdim=True).clamp_min(eps)

    @staticmethod
    def normalize_prob(prob, mask=None, dim=-1, eps=1e-12):
        """
        Normalize a probability-like tensor while keeping invalid mask positions at 0.
        Only valid positions are clamped to eps.
        """
        if mask is not None:
            prob = prob.masked_fill(~mask, 0.0)
            prob = torch.where(mask, prob.clamp_min(eps), torch.zeros_like(prob))
        else:
            prob = prob.clamp_min(eps)
        return prob / prob.sum(dim=dim, keepdim=True).clamp_min(eps)

    @staticmethod
    def masked_kl_div(input_prob, target_prob, mask, eps=1e-12):
        """
        KL(target || input) on valid mask positions only.

        Do not pass log(0) at invalid positions into F.kl_div.
        Otherwise 0 * (-inf) can produce NaN.
        """
        input_prob = input_prob.masked_fill(~mask, 0.0)
        target_prob = target_prob.masked_fill(~mask, 0.0)

        # Valid positions get eps floor; invalid positions get safe dummy values.
        safe_input = torch.where(mask, input_prob.clamp_min(eps), torch.ones_like(input_prob))
        safe_target = torch.where(mask, target_prob.clamp_min(eps), torch.zeros_like(target_prob))

        kl = F.kl_div(safe_input.log(), safe_target, reduction="none")
        kl = kl.masked_fill(~mask, 0.0)

        # Same spirit as reduction="batchmean": divide by batch size.
        return kl.sum() / input_prob.size(0)

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

    def compute_attention_loss(self, drug_feat, protein_feat, affinity_matrix, affinity_mask):
        # Query/key in projected contrastive space.构造QKV
        drug_q = self.drug_proj(drug_feat)
        drug_k = self.drug_proj(drug_feat)
        protein_q = self.protein_proj(protein_feat)
        protein_k = self.protein_proj(protein_feat)

        #计算cross-attention logits，得到QK^T/sqrt(d)
        scores_d2p = torch.matmul(drug_q, protein_k.t()) / self.attn_scale
        scores_p2d = torch.matmul(protein_q, drug_k.t()) / self.attn_scale

        #logits变成预测概率分布，得到attn_d2p和attn_p2d
        attn_d2p = self.masked_softmax(scores_d2p, affinity_mask, dim=1)
        attn_p2d = self.masked_softmax(scores_p2d, affinity_mask.t(), dim=1)

        #构造监督信号，把真实label matrix变成概率分布
        target_d2p_base = self.masked_softmax(
            affinity_matrix / self.affinity_temperature,
            affinity_mask,
            dim=1,
        )
        target_p2d_base = self.masked_softmax(
            affinity_matrix.t() / self.affinity_temperature,
            affinity_mask.t(),
            dim=1,
        )
        #计算drug之间的相似性
        label_sim = self.compute_label_similarity(affinity_matrix, affinity_mask)

        # Diffuse probability mass to label-similar samples.
        # label_sim is row-normalized; using label_sim means distribution mass is mixed
        # according to source-row label similarity.
        # 把label信息在相似的drug上传播
        target_d2p_refine = torch.matmul(target_d2p_base, label_sim)
        target_p2d_refine = torch.matmul(target_p2d_base, label_sim)

        mix = min(1.0, max(0.0, float(self.labelsim_mix)))

        #融合base和refined target，得到最终的target_d2p和target_p2d
        target_d2p = (1.0 - mix) * target_d2p_base + mix * target_d2p_refine
        target_p2d = (1.0 - mix) * target_p2d_base + mix * target_p2d_refine

        eps = 1e-12
        attn_d2p = self.normalize_prob(attn_d2p, affinity_mask, dim=1, eps=eps)
        attn_p2d = self.normalize_prob(attn_p2d, affinity_mask.t(), dim=1, eps=eps)
        target_d2p = self.normalize_prob(target_d2p, affinity_mask, dim=1, eps=eps)
        target_p2d = self.normalize_prob(target_p2d, affinity_mask.t(), dim=1, eps=eps)

        # 核心loss计算，KL散度
        l_attn_d2p = self.masked_kl_div(attn_d2p, target_d2p, affinity_mask, eps=eps)
        l_attn_p2d = self.masked_kl_div(attn_p2d, target_p2d, affinity_mask.t(), eps=eps)
        l_attn = l_attn_d2p + l_attn_p2d

        # Values are aggregated in hidden feature space, not projected space.
        drug_context = torch.matmul(attn_d2p, protein_feat)
        protein_context = torch.matmul(attn_p2d, drug_feat)

        drug_enhanced = drug_feat + drug_context
        protein_enhanced = protein_feat + protein_context

        drug_ffn = self.drug_attn_ffn(drug_enhanced)
        protein_ffn = self.protein_attn_ffn(protein_enhanced)

        d_final = self.drug_attn_norm(drug_enhanced + drug_ffn)
        p_final = self.protein_attn_norm(protein_enhanced + protein_ffn)

        mean_valid_per_row = affinity_mask.sum(dim=1).float().mean()
        mean_valid_offdiag = (affinity_mask.sum(dim=1) - 1).float().mean()
        mean_diag_prior_mass = torch.diag(target_d2p).mean()

        return (
            l_attn,
            d_final,
            p_final,
            attn_d2p,
            attn_p2d,
            mean_valid_per_row,
            mean_valid_offdiag,
            mean_diag_prior_mass,
        )

    def forward(self, batch, affinity_matrix=None, affinity_mask=None, return_aux=True):
        drug_1d_feat = self.drug_1d_encoder(batch["drug_1d"])
        drug_3d_feat = self.drug_3d_encoder(batch["drug_3d"])
        drug_feat, drug_gate = self.fuse_drug(drug_1d_feat, drug_3d_feat)

        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])
        protein_feat, protein_gate = self.fuse_protein(protein_1d_feat, protein_3d_feat)

        protein_pcl_loss, protein_pcl_logits = self.compute_pcl_loss(
            protein_1d_feat,
            protein_3d_feat,
            self.logit_scale_pcl,
            self.proj_1d,
            self.proj_3d,
        )
        drug_pcl_loss, drug_pcl_logits = self.compute_pcl_loss(
            drug_1d_feat,
            drug_3d_feat,
            self.logit_scale_drug_pcl,
            self.drug_proj_1d,
            self.drug_proj_3d,
        )
        pcl_loss = protein_pcl_loss + drug_pcl_loss

        if affinity_matrix is not None and affinity_mask is not None:
            (
                attn_loss,
                drug_final,
                protein_final,
                attn_d2p,
                attn_p2d,
                mean_valid_per_row,
                mean_valid_offdiag,
                mean_diag_prior_mass,
            ) = self.compute_attention_loss(
                drug_feat,
                protein_feat,
                affinity_matrix,
                affinity_mask,
            )
        else:
            device = drug_feat.device
            attn_loss = torch.tensor(0.0, device=device)
            drug_final = drug_feat
            protein_final = protein_feat
            attn_d2p = None
            attn_p2d = None
            mean_valid_per_row = torch.tensor(0.0, device=device)
            mean_valid_offdiag = torch.tensor(0.0, device=device)
            mean_diag_prior_mass = torch.tensor(0.0, device=device)

        pair_feat = torch.cat([drug_final, protein_final], dim=-1)
        pred = self.decoder(pair_feat)

        if return_aux:
            aux = {
                "attn_loss": attn_loss,
                "pcl_loss": pcl_loss,
                "protein_pcl_loss": protein_pcl_loss.detach(),
                "drug_pcl_loss": drug_pcl_loss.detach(),
                "protein_pcl_logits": protein_pcl_logits,
                "drug_pcl_logits": drug_pcl_logits,
                "attn_d2p": attn_d2p,
                "attn_p2d": attn_p2d,
                "mean_gate": ((drug_gate.mean() + protein_gate.mean()) / 2).detach(),
                "mean_drug_gate": drug_gate.mean().detach(),
                "mean_protein_gate": protein_gate.mean().detach(),
                "mean_valid_per_row": mean_valid_per_row.detach(),
                "mean_valid_offdiag": mean_valid_offdiag.detach(),
                "mean_diag_prior_mass": mean_diag_prior_mass.detach(),
                "labelsim_tau": torch.tensor(self.labelsim_tau, device=pred.device),
                "labelsim_mix": torch.tensor(self.labelsim_mix, device=pred.device),
            }
            return pred, aux

        return pred