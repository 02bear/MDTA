# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_1d_encoder import Drug1DEncoder
from models.drug_3d_egnn_encoder import Drug3DEGNNEncoder
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder
from models.fusion import ConcatFusion
from models.decoder import Decoder


class MyModelMDTAP13D(nn.Module):
    """
    Baseline P13D + Affinity-aware Gramian Volume Contrastive Learning.

    主回归分支保持 baseline 不变:
      drug_1d_encoder + drug_3d_encoder -> drug_fusion(ConcatFusion)
      protein_1d_encoder + protein_3d_encoder -> protein_fusion(ConcatFusion)
      concat(drug_feat, protein_feat) -> decoder(MLP)

    新增辅助分支:
      对 drug_1d / drug_3d / protein_1d / protein_3d 四个模态特征做 projector，
      使用 affinity-aware Gramian volume contrastive loss 作为训练时的辅助损失。
    """

    def __init__(
        self,
        drug_1d_in_dim=768,
        drug_3d_node_in_dim=10,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        dropout=0.1,
        task="regression",
        contrastive_dim=128,
        vol_temperature=0.1,
        vol_label_tau=1.0,
        vol_det_eps=1e-6,
    ):
        super().__init__()

        # =========================
        # Baseline encoders / fusions / decoder: unchanged
        # =========================
        self.drug_1d_encoder = Drug1DEncoder(input_dim=drug_1d_in_dim, hidden_dim=hidden_dim)
        self.drug_3d_encoder = Drug3DEGNNEncoder(
            node_in_dim=drug_3d_node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=3,
            dropout=dropout,
        )
        self.drug_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
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

        # =========================
        # New auxiliary contrastive projectors
        # =========================
        self.contrastive_dim = contrastive_dim
        self.vol_temperature = float(vol_temperature)
        self.vol_label_tau = float(vol_label_tau)
        self.vol_det_eps = float(vol_det_eps)

        self.drug_1d_projector = self._build_projector(hidden_dim, contrastive_dim, dropout)
        self.drug_3d_projector = self._build_projector(hidden_dim, contrastive_dim, dropout)
        self.protein_1d_projector = self._build_projector(hidden_dim, contrastive_dim, dropout)
        self.protein_3d_projector = self._build_projector(hidden_dim, contrastive_dim, dropout)

    @staticmethod
    def _build_projector(input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        """
        Lightweight projector, following the idea of mapping different modalities
        into a shared contrastive space. It is only used by the auxiliary loss.
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _project_modalities(
        self,
        drug_1d_feat: torch.Tensor,
        drug_3d_feat: torch.Tensor,
        protein_1d_feat: torch.Tensor,
        protein_3d_feat: torch.Tensor,
    ) -> List[torch.Tensor]:
        z_drug_1d = F.normalize(self.drug_1d_projector(drug_1d_feat), p=2, dim=-1, eps=1e-8)
        z_drug_3d = F.normalize(self.drug_3d_projector(drug_3d_feat), p=2, dim=-1, eps=1e-8)
        z_protein_1d = F.normalize(self.protein_1d_projector(protein_1d_feat), p=2, dim=-1, eps=1e-8)
        z_protein_3d = F.normalize(self.protein_3d_projector(protein_3d_feat), p=2, dim=-1, eps=1e-8)
        return [z_drug_1d, z_drug_3d, z_protein_1d, z_protein_3d]

    def _gram_volume(self, modality_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute sqrt(det(Gram)) for a list of modality tensors.

        Each tensor may have shape:
          [B, D]      -> returns [B]
          [B, B, D]   -> returns [B, B]

        We add eps * I to the Gram matrix and use slogdet for numerical stability.
        """
        if len(modality_tensors) < 2:
            raise ValueError("Gram volume requires at least two modality tensors.")

        z = torch.stack(modality_tensors, dim=-2)  # [..., M, D]
        z_float = z.float()
        gram = torch.matmul(z_float, z_float.transpose(-1, -2))  # [..., M, M]

        m = gram.size(-1)
        eye = torch.eye(m, device=gram.device, dtype=gram.dtype)
        view_shape = [1] * (gram.dim() - 2) + [m, m]
        gram = gram + self.vol_det_eps * eye.view(*view_shape)

        sign, logabsdet = torch.linalg.slogdet(gram)
        fallback = torch.full_like(logabsdet, math.log(self.vol_det_eps))
        safe_logdet = torch.where(sign > 0, logabsdet, fallback)
        safe_logdet = torch.clamp(safe_logdet, min=-60.0, max=20.0)

        volume = torch.exp(0.5 * safe_logdet)
        return volume.to(dtype=modality_tensors[0].dtype)

    def _label_negative_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Build label-aware negative weights.

        If two samples have similar affinity labels, they should not be strongly
        pushed apart. The weight approaches 0 for similar labels and approaches 1
        for very different labels:
            w_ij = 1 - exp(-|y_i - y_j| / tau_y)
        """
        labels = labels.detach().view(-1, 1).float()
        diff = torch.abs(labels - labels.t())
        sim = torch.exp(-diff / max(self.vol_label_tau, 1e-8))
        weights = (1.0 - sim).clamp(min=0.0, max=1.0)

        bsz = labels.size(0)
        eye = torch.eye(bsz, device=labels.device, dtype=torch.bool)
        weights = weights.masked_fill(eye, 0.0)
        return weights

    @staticmethod
    def _weighted_infonce_from_logits(
        logits: torch.Tensor,
        negative_weights: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Weighted InfoNCE with diagonal positives.

        denominator = positive + sum_j!=i w_ij * negative_ij
        where w_ij is label-aware. This avoids forcing pairs with similar labels
        to be treated as equally hard negatives.
        """
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits)

        pos = torch.diagonal(exp_logits, dim1=0, dim2=1).clamp_min(eps)
        denom = pos + (exp_logits * negative_weights).sum(dim=1)
        denom = denom.clamp_min(eps)

        loss = -torch.log(pos / denom)
        return loss.mean()

    def compute_affinity_aware_volume_loss(
        self,
        projected_modalities: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Affinity-aware four-modal Gramian volume contrastive loss.

        For each anchor modality:
          forward negative: replace only the anchor modality across the batch;
          reverse negative: keep the anchor fixed and replace the other modalities.

        This follows the GRAM-style "single-modality altered negative" idea, but
        uses affinity-aware negative weighting for Davis-style regression labels.
        """
        bsz = projected_modalities[0].size(0)
        if bsz <= 1:
            return projected_modalities[0].new_zeros(())

        n_modalities = len(projected_modalities)
        if n_modalities != 4:
            raise ValueError(f"Expected 4 modalities, got {n_modalities}.")

        negative_weights = self._label_negative_weights(labels).to(projected_modalities[0].device)
        temperature = max(self.vol_temperature, 1e-8)

        losses = []
        for anchor_idx in range(n_modalities):
            anchor = projected_modalities[anchor_idx]           # [B, D]
            others = [
                projected_modalities[k]
                for k in range(n_modalities)
                if k != anchor_idx
            ]

            # Forward direction:
            # row i keeps other modalities from sample i,
            # column j uses anchor modality from sample j.
            anchor_candidates = anchor.unsqueeze(0).expand(bsz, bsz, -1)  # [i, j] = anchor_j
            fixed_others = [x.unsqueeze(1).expand(bsz, bsz, -1) for x in others]  # [i, j] = other_i
            volume_fw = self._gram_volume([anchor_candidates] + fixed_others)
            logits_fw = -volume_fw / temperature
            losses.append(self._weighted_infonce_from_logits(logits_fw, negative_weights))

            # Reverse direction:
            # row i keeps anchor modality from sample i,
            # column j uses all other modalities from sample j.
            fixed_anchor = anchor.unsqueeze(1).expand(bsz, bsz, -1)  # [i, j] = anchor_i
            other_candidates = [x.unsqueeze(0).expand(bsz, bsz, -1) for x in others]  # [i, j] = other_j
            volume_bw = self._gram_volume([fixed_anchor] + other_candidates)
            logits_bw = -volume_bw / temperature
            losses.append(self._weighted_infonce_from_logits(logits_bw, negative_weights))

        return torch.stack(losses).mean()

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        # =========================
        # Baseline forward path: unchanged in structure
        # =========================
        drug_1d_feat = self.drug_1d_encoder(batch["drug_1d"])           # [B, H]
        drug_3d_feat = self.drug_3d_encoder(batch["drug_3d"])           # [B, H]
        drug_feat = self.drug_fusion([drug_1d_feat, drug_3d_feat])      # [B, H]

        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])  # [B, H]
        protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])  # [B, H]
        protein_feat = self.protein_fusion([protein_1d_feat, protein_3d_feat])

        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        out = self.decoder(pair_feat)

        if not return_aux:
            return out

        projected_modalities = self._project_modalities(
            drug_1d_feat=drug_1d_feat,
            drug_3d_feat=drug_3d_feat,
            protein_1d_feat=protein_1d_feat,
            protein_3d_feat=protein_3d_feat,
        )
        volume_loss = self.compute_affinity_aware_volume_loss(
            projected_modalities=projected_modalities,
            labels=batch["label"],
        )

        return {
            "pred": out,
            "volume_loss": volume_loss,
            "drug_feat": drug_feat,
            "protein_feat": protein_feat,
            "pair_feat": pair_feat,
        }
