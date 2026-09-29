# -*- coding: utf-8 -*-
"""
MDTA-REFINED model.

这个文件只放模型结构，不放训练流程、数据划分、metrics、main 函数。
对应 train_refined.py 中的：

    from models.model_p13d_refined import MyModelMDTAP13DRefined

模型策略：
1. Drug:    1D ChemBERTa embedding + 3D EGNN
2. Protein: 1D ESM2 embedding      + 3D EGNN/GVP-style protein graph encoder
3. Fusion:  simple gated fusion
4. Head:    MLP regression decoder
5. Aux:     optional same-entity 1D-3D PCL, no SoftCLIP, no label-sim, no affinity matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_3d_egnn_encoder import Drug3DEGNNEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder
from models.decoder import Decoder


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_1d: torch.Tensor, h_3d: torch.Tensor):
        h_1d = torch.nan_to_num(h_1d, nan=0.0, posinf=1e4, neginf=-1e4)
        h_3d = torch.nan_to_num(h_3d, nan=0.0, posinf=1e4, neginf=-1e4)

        gate = self.gate_mlp(torch.cat([h_1d, h_3d], dim=-1))
        fused = gate * h_1d + (1.0 - gate) * h_3d
        fused = self.out_norm(fused)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)

        return fused, gate


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
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        z = self.net(x)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        z = F.normalize(z, p=2, dim=-1, eps=1e-8)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        return z


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


class MyModelMDTAP13DRefined(nn.Module):
    """
    Stable refined MDTA model.

    重要：
    - 不包含 SoftCLIP
    - 不包含 label-sim diffusion
    - 不需要 affinity_matrix / affinity_mask
    - 不做 drug-protein batch-level soft target
    - 只保留 MSE 主任务 + 可选同一实体 1D/3D PCL
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
        task: str = "regression",
    ):
        super().__init__()

        self.temperature = temperature
        self.use_drug_pcl = use_drug_pcl

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

        self.drug_fusion = GatedFusion(
            hidden_dim=hidden_dim,
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

        self.protein_fusion = GatedFusion(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # PCL projection heads
        self.protein_proj_1d = ProjectionHead(
            hidden_dim=hidden_dim,
            contrastive_dim=contrastive_dim,
            dropout=dropout,
        )
        self.protein_proj_3d = ProjectionHead(
            hidden_dim=hidden_dim,
            contrastive_dim=contrastive_dim,
            dropout=dropout,
        )

        self.drug_proj_1d = ProjectionHead(
            hidden_dim=hidden_dim,
            contrastive_dim=contrastive_dim,
            dropout=dropout,
        )
        self.drug_proj_3d = ProjectionHead(
            hidden_dim=hidden_dim,
            contrastive_dim=contrastive_dim,
            dropout=dropout,
        )

        # Regression decoder
        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
        )

    def compute_pcl_loss(
        self,
        drug_1d_feat: torch.Tensor,
        drug_3d_feat: torch.Tensor,
        protein_1d_feat: torch.Tensor,
        protein_3d_feat: torch.Tensor,
    ):
        """
        Same-entity 1D-3D PCL.

        默认只启用 protein PCL：
            total_pcl = protein_pcl

        如果 use_drug_pcl=True：
            total_pcl = 0.5 * (protein_pcl + drug_pcl)
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
        # Drug
        drug_1d_feat = self.drug_1d_encoder(batch["drug_1d"])
        drug_3d_feat = self.drug_3d_encoder(batch["drug_3d"])
        drug_feat, drug_gate = self.drug_fusion(drug_1d_feat, drug_3d_feat)

        # Protein
        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])
        protein_feat, protein_gate = self.protein_fusion(protein_1d_feat, protein_3d_feat)

        # Regression
        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        pred = self.decoder(pair_feat)

        if not return_aux:
            return pred

        protein_pcl_loss, drug_pcl_loss, pcl_loss = self.compute_pcl_loss(
            drug_1d_feat=drug_1d_feat,
            drug_3d_feat=drug_3d_feat,
            protein_1d_feat=protein_1d_feat,
            protein_3d_feat=protein_3d_feat,
        )

        aux = {
            # 这个不能 detach，因为 train_refined.py 里要用它反传
            "pcl_loss": pcl_loss,

            # 下面这些只用于日志，可以 detach
            "protein_pcl_loss": protein_pcl_loss.detach(),
            "drug_pcl_loss": drug_pcl_loss.detach(),
            "mean_drug_gate": drug_gate.mean().detach(),
            "mean_protein_gate": protein_gate.mean().detach(),
        }

        return pred, aux
