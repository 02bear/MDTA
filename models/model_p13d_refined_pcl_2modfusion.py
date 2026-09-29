# -*- coding: utf-8 -*-
"""
MDTA-REFINED model with 2-stage fusion.

策略：
1. Drug 1D/3D encoder -> same-entity PCL -> drug-side I2MoE feature fusion -> drug_feat
2. Protein 1D/3D encoder -> same-entity PCL -> protein-side I2MoE feature fusion -> protein_feat
3. Final DTA regression uses an explicit drug-protein interaction head:
   concat([drug_feat, protein_feat, drug_feat * protein_feat, |drug_feat - protein_feat|])
   plus a light bilinear interaction term.

说明：
- 不包含 SoftCLIP
- 不包含 label-sim diffusion
- 不需要 affinity_matrix / affinity_mask
- 只保留 MSE 主任务 + 同一实体 1D/3D PCL + I2MoE interaction auxiliary loss
"""

import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_3d_egnn_encoder import Drug3DEGNNEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder


class ProjectionHead(nn.Module):
    """
    Projection head for same-entity 1D-3D PCL.
    """

    def __init__(self, hidden_dim: int, contrastive_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def symmetric_infonce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
    clamp_logits: float = 20.0,
):
    """
    Standard symmetric InfoNCE for same-entity 1D-3D alignment.
    """
    if z_a.size(0) != z_b.size(0):
        raise ValueError(f"InfoNCE batch mismatch: {z_a.size(0)} vs {z_b.size(0)}")

    # batch_size=1 时没有 batch 内负样本，直接返回 0，防止 CE 退化。
    if z_a.size(0) <= 1:
        return z_a.new_zeros(())

    temperature = max(float(temperature), 1e-6)
    logits = torch.matmul(z_a, z_b.t()) / temperature

    if clamp_logits is not None and clamp_logits > 0:
        logits = logits.clamp(min=-float(clamp_logits), max=float(clamp_logits))

    labels = torch.arange(logits.size(0), device=logits.device)
    loss_a_to_b = F.cross_entropy(logits, labels)
    loss_b_to_a = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a_to_b + loss_b_to_a)


class I2MoEFeatureFusion(nn.Module):
    """
    I2MoE / IMoE-style feature fusion for any number of modalities.

    与你之前的 I2MoEFusionRegressor 不同：
    - 这里每个 expert 输出 hidden_dim 维特征，而不是直接输出 1 个回归值。
    - 这个模块用于同一实体内部的 1D/3D 融合：
        drug_1d + drug_3d -> drug_feat
        protein_1d + protein_3d -> protein_feat
    - expert 数量仍然是 num_modalities + 2：
        unimodal experts + synergy expert + redundancy expert
    """

    def __init__(
        self,
        hidden_dim: int,
        num_modalities: int,
        modality_names: Optional[List[str]] = None,
        dropout: float = 0.1,
        gate_temperature: float = 1.0,
        interaction_margin: float = 0.05,
        expert_hidden_mult: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        if num_modalities < 2:
            raise ValueError(f"num_modalities must be >= 2, got {num_modalities}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be positive, got {gate_temperature}")
        if expert_hidden_mult <= 0:
            raise ValueError(f"expert_hidden_mult must be positive, got {expert_hidden_mult}")

        if modality_names is None:
            modality_names = [f"mod{i}" for i in range(num_modalities)]
        if len(modality_names) != num_modalities:
            raise ValueError(
                "len(modality_names) must equal num_modalities: "
                f"{len(modality_names)} vs {num_modalities}"
            )

        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.modality_names = list(modality_names)
        self.num_experts = num_modalities + 2
        self.syn_idx = num_modalities
        self.red_idx = num_modalities + 1
        self.gate_temperature = float(gate_temperature)
        self.interaction_margin = float(interaction_margin)

        input_dim = hidden_dim * num_modalities
        expert_hidden_dim = hidden_dim * expert_hidden_mult

        self.input_norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        self.output_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.num_experts)
            ]
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, expert_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, self.num_experts),
        )

        self._uni_aux_names = self._build_uni_aux_names(self.modality_names)

    @staticmethod
    def _clean_name(name: str, fallback: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
        return cleaned or fallback

    @classmethod
    def _build_uni_aux_names(cls, modality_names: List[str]) -> List[str]:
        counts = {}
        aux_names = []
        for idx, name in enumerate(modality_names):
            base = cls._clean_name(name, f"mod{idx}")
            count = counts.get(base, 0)
            counts[base] = count + 1
            if count > 0:
                base = f"{base}_{count}"
            aux_names.append(f"i2moe_w_uni_{base}")
        return aux_names

    def _check_inputs(self, zs: List[torch.Tensor]) -> None:
        if len(zs) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(zs)}")
        if not zs:
            raise ValueError("zs must not be empty")

        batch_size = zs[0].size(0)
        for idx, z in enumerate(zs):
            if z.dim() != 2:
                raise ValueError(f"zs[{idx}] must be 2D [B, hidden_dim], got shape {tuple(z.shape)}")
            if z.size(0) != batch_size:
                raise ValueError(f"Batch size mismatch at zs[{idx}]: {z.size(0)} vs {batch_size}")
            if z.size(-1) != self.hidden_dim:
                raise ValueError(f"Hidden dim mismatch at zs[{idx}]: {z.size(-1)} vs {self.hidden_dim}")

    def _concat_inputs(self, zs: List[torch.Tensor]) -> torch.Tensor:
        return self.input_norm(torch.cat(zs, dim=-1))

    def _expert_features(self, zs: List[torch.Tensor]) -> torch.Tensor:
        x = self._concat_inputs(zs)
        feats = [expert(x) for expert in self.experts]
        return torch.stack(feats, dim=1)  # [B, num_experts, hidden_dim]

    def _interaction_loss(self, zs: List[torch.Tensor], full_feats: torch.Tensor) -> torch.Tensor:
        """
        Interaction-aware regularization:
        - unimodal expert 对自己的模态敏感，对其他模态相对稳定；
        - synergy expert 对任意模态缺失都敏感；
        - redundancy expert 对任意模态缺失都相对稳定。
        """
        losses = []
        masked_feats = []

        for mask_idx in range(self.num_modalities):
            z_masked = list(zs)
            z_masked[mask_idx] = torch.randn_like(zs[mask_idx])
            masked_feats.append(self._expert_features(z_masked))

        masked_feats = torch.stack(masked_feats, dim=0)  # [M, B, E, H]

        for expert_idx in range(self.num_modalities):
            anchor = full_feats[:, expert_idx, :]
            for mask_idx in range(self.num_modalities):
                other = masked_feats[mask_idx, :, expert_idx, :]
                if mask_idx == expert_idx:
                    dist = (anchor - other).pow(2).mean(dim=-1)
                    losses.append(F.relu(self.interaction_margin - dist).mean())
                else:
                    losses.append(F.mse_loss(anchor, other))

        syn_anchor = full_feats[:, self.syn_idx, :]
        red_anchor = full_feats[:, self.red_idx, :]
        for mask_idx in range(self.num_modalities):
            syn_other = masked_feats[mask_idx, :, self.syn_idx, :]
            syn_dist = (syn_anchor - syn_other).pow(2).mean(dim=-1)
            losses.append(F.relu(self.interaction_margin - syn_dist).mean())

            red_other = masked_feats[mask_idx, :, self.red_idx, :]
            losses.append(F.mse_loss(red_anchor, red_other))

        if not losses:
            return torch.zeros((), device=zs[0].device)
        return torch.stack(losses).mean()

    def forward(self, zs: List[torch.Tensor], compute_interaction_loss: bool = True):
        self._check_inputs(zs)

        gate_input = self._concat_inputs(zs)
        gate_logits = self.gate(gate_input)
        weights = F.softmax(gate_logits / self.gate_temperature, dim=-1)  # [B, E]

        expert_feats = self._expert_features(zs)  # [B, E, H]
        fused = torch.sum(weights.unsqueeze(-1) * expert_feats, dim=1)
        fused = self.output_norm(fused)

        if self.training and compute_interaction_loss:
            interaction_loss = self._interaction_loss(zs, expert_feats)
        else:
            interaction_loss = torch.zeros((), device=zs[0].device)

        aux = {
            "i2moe_loss": interaction_loss,
            "i2moe_w_synergy": weights[:, self.syn_idx].mean().detach(),
            "i2moe_w_redundancy": weights[:, self.red_idx].mean().detach(),
        }
        for idx, key in enumerate(self._uni_aux_names):
            aux[key] = weights[:, idx].mean().detach()

        return fused, aux


class PairInteractionRegressor(nn.Module):
    """
    Final drug-protein interaction head.

    为什么不用第三次 I2MoE：
    drug_feat 和 protein_feat 已经不是“同一实体的多模态信息”，而是两个不同实体。
    DTA 最后需要建模二者的匹配关系，所以这里用显式交互项更稳：
    [drug, protein, drug * protein, |drug - protein|] + bilinear interaction。
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)

    def forward(self, drug_feat: torch.Tensor, protein_feat: torch.Tensor):
        d = self.drug_norm(drug_feat)
        p = self.protein_norm(protein_feat)
        pair_feat = torch.cat([d, p, d * p, torch.abs(d - p)], dim=-1)
        return self.mlp(pair_feat) + self.bilinear(d, p)


class MyModelMDTAP13DRefined(nn.Module):
    """
    Stable refined MDTA model with PCL -> entity-level I2MoE -> pair interaction regression.
    """

    def __init__(
        self,
        drug_1d_in_dim: int = 768,
        drug_3d_node_in_dim: int = 10,
        protein_1d_in_dim: int = 1280,
        protein_3d_node_s_dim: int = 6,
        protein_3d_node_v_dim: int = 3,  # 保留用于兼容 train 脚本
        hidden_dim: int = 128,
        contrastive_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.1,
        use_drug_pcl: bool = True,
        i2moe_gate_temperature: float = 1.0,
        i2moe_interaction_margin: float = 0.05,
        i2moe_expert_hidden_mult: int = 2,
        task: str = "regression",
    ):
        super().__init__()

        self.temperature = temperature
        self.use_drug_pcl = use_drug_pcl
        self.task = task

        # Drug encoders
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

        # Protein encoders
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

        # PCL projection heads
        self.protein_proj_1d = ProjectionHead(hidden_dim, contrastive_dim, dropout)
        self.protein_proj_3d = ProjectionHead(hidden_dim, contrastive_dim, dropout)
        self.drug_proj_1d = ProjectionHead(hidden_dim, contrastive_dim, dropout)
        self.drug_proj_3d = ProjectionHead(hidden_dim, contrastive_dim, dropout)

        # Entity-level I2MoE feature fusion
        self.drug_i2moe_fusion = I2MoEFeatureFusion(
            hidden_dim=hidden_dim,
            num_modalities=2,
            modality_names=["drug1d", "drug3d"],
            dropout=dropout,
            gate_temperature=i2moe_gate_temperature,
            interaction_margin=i2moe_interaction_margin,
            expert_hidden_mult=i2moe_expert_hidden_mult,
        )
        self.protein_i2moe_fusion = I2MoEFeatureFusion(
            hidden_dim=hidden_dim,
            num_modalities=2,
            modality_names=["prot1d", "prot3d"],
            dropout=dropout,
            gate_temperature=i2moe_gate_temperature,
            interaction_margin=i2moe_interaction_margin,
            expert_hidden_mult=i2moe_expert_hidden_mult,
        )

        # Final drug-protein interaction regressor
        self.regressor = PairInteractionRegressor(hidden_dim=hidden_dim, dropout=dropout)

    @staticmethod
    def _prefix_aux(aux, prefix: str):
        return {f"{prefix}_{key}": value for key, value in aux.items()}

    def compute_pcl_loss(
        self,
        drug_1d_feat: torch.Tensor,
        drug_3d_feat: torch.Tensor,
        protein_1d_feat: torch.Tensor,
        protein_3d_feat: torch.Tensor,
    ):
        """
        Same-entity 1D-3D PCL.

        默认启用 protein PCL + drug PCL：
            total_pcl = 0.5 * (protein_pcl + drug_pcl)

        如果 use_drug_pcl=False：
            total_pcl = protein_pcl
        """
        protein_pcl = symmetric_infonce(
            self.protein_proj_1d(protein_1d_feat),
            self.protein_proj_3d(protein_3d_feat),
            temperature=self.temperature,
        )

        if not self.use_drug_pcl:
            drug_pcl = protein_pcl.new_zeros(())
            total_pcl = protein_pcl
            return protein_pcl, drug_pcl, total_pcl

        drug_pcl = symmetric_infonce(
            self.drug_proj_1d(drug_1d_feat),
            self.drug_proj_3d(drug_3d_feat),
            temperature=self.temperature,
        )
        total_pcl = 0.5 * (protein_pcl + drug_pcl)
        return protein_pcl, drug_pcl, total_pcl

    def forward(self, batch, return_aux: bool = True):
        # 1) Encode each modality.
        drug_1d_feat = self.drug_1d_encoder(batch["drug_1d"])
        drug_3d_feat = self.drug_3d_encoder(batch["drug_3d"])
        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])

        # 2) Same-entity I2MoE feature fusion.
        drug_feat, drug_i2moe_aux = self.drug_i2moe_fusion(
            [drug_1d_feat, drug_3d_feat],
            compute_interaction_loss=self.training,
        )
        protein_feat, protein_i2moe_aux = self.protein_i2moe_fusion(
            [protein_1d_feat, protein_3d_feat],
            compute_interaction_loss=self.training,
        )

        # 3) Final DTA regression by explicit drug-protein interaction.
        pred = self.regressor(drug_feat, protein_feat)

        if not return_aux:
            return pred

        # 4) PCL is computed on modality-specific features before fusion.
        protein_pcl_loss, drug_pcl_loss, pcl_loss = self.compute_pcl_loss(
            drug_1d_feat=drug_1d_feat,
            drug_3d_feat=drug_3d_feat,
            protein_1d_feat=protein_1d_feat,
            protein_3d_feat=protein_3d_feat,
        )

        i2moe_loss = 0.5 * (
            drug_i2moe_aux["i2moe_loss"] + protein_i2moe_aux["i2moe_loss"]
        )

        aux = {
            # 训练脚本会用这两个 loss 反传，不能 detach。
            "pcl_loss": pcl_loss,
            "i2moe_loss": i2moe_loss,

            # 下面只用于日志，可以 detach。
            "protein_pcl_loss": protein_pcl_loss.detach(),
            "drug_pcl_loss": drug_pcl_loss.detach(),
            "drug_i2moe_loss": drug_i2moe_aux["i2moe_loss"].detach(),
            "protein_i2moe_loss": protein_i2moe_aux["i2moe_loss"].detach(),
            "mean_drug_gate": drug_i2moe_aux["i2moe_w_uni_drug1d"].detach(),
            "mean_protein_gate": protein_i2moe_aux["i2moe_w_uni_prot1d"].detach(),
            "pair_cosine": F.cosine_similarity(drug_feat, protein_feat, dim=-1).mean().detach(),
        }

        # Detailed I2MoE weights for logging/debugging.
        for key, value in self._prefix_aux(drug_i2moe_aux, "drug").items():
            if key != "drug_i2moe_loss":
                aux[key] = value.detach()
        for key, value in self._prefix_aux(protein_i2moe_aux, "protein").items():
            if key != "protein_i2moe_loss":
                aux[key] = value.detach()

        return pred, aux
