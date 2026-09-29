# -*- coding: utf-8 -*-
"""
AR2-B1: Frozen E2 + Double-Centered Interaction Bottleneck.

Scientific purpose
------------------
AR2-A1 showed a clear failure mode:
    pair scores -> uniform
while the branch could still improve MSE by using mean local features.

B1 removes that bypass by enforcing the following structural invariant:

    no atom-residue interaction preference
        => centered attention == 0
        => z_AR2 == 0
        => delta_AR2 == 0

Design
------
1) Keep the validated fold-specific E2 checkpoint fully frozen.
2) Reuse ONLY the trained E2 selector indices as a proposal.
3) Do NOT learn a new 64->24 hard reranker in B1.
   The first sorted Top-24 E2 proposal residues are used directly.
4) Drug-side AR nodes:
      frozen Drug3D node tokens, heavy atoms only.
5) Protein-side AR nodes:
      frozen Protein3D residue node tokens.
6) Pair score is strict multiplicative cross interaction:
      q_a^T k_r / sqrt(d) + MLP(q_a * k_r)
   No [q_a, k_r], no |q_a-k_r| in the strict B1 path.
7) Remove atom and residue main effects with double centering:
      S_int = S - mean_r(S) - mean_a(S) + mean_ar(S)
8) Global pair attention is centered against the uniform null:
      alpha = softmax(S_int)
      w = alpha - 1/N_valid
9) Pair embedding is also double centered, removing additive atom/residue
   main-effect structure.
10) z_AR2 = sum_{a,r} w_ar * E_int_ar
11) Bias-free residual head:
      delta_AR2 = H(z_AR2)
   Therefore H(0) == 0 exactly.
12) Final:
      y = y_E2 + delta_AR2

Important
---------
- No intermolecular Euclidean distance is used: Davis ligand SDF and protein
  structures are independent coordinate frames.
- The old E2 AR scores/features are NOT used as B1 interaction evidence.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.model_p13d_finegrained_residual import (
    MyModelMDTAP13DFineGrained,
)


def masked_softmax(
    score: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Numerically safe softmax; fully masked rows become exactly zero."""
    mask = mask.bool()
    masked = score.masked_fill(~mask, -1e9)
    weight = torch.softmax(masked, dim=dim)
    weight = weight * mask.to(weight.dtype)
    return weight / weight.sum(dim=dim, keepdim=True).clamp_min(eps)


def _masked_double_center_scalar(
    x: torch.Tensor,                 # [B,A,K]
    pair_mask: torch.Tensor,         # [B,A,K]
) -> torch.Tensor:
    """
    Two-way (atom x residue) masked double centering.

    For the valid rectangular atom-residue block:
        x_int = x - atom_row_mean - residue_col_mean + grand_mean

    If x = f(atom) + g(residue) + c, x_int is exactly zero up to numerical error.
    """
    m = pair_mask.to(x.dtype)
    x0 = x * m

    row_count = m.sum(dim=2, keepdim=True).clamp_min(1.0)
    col_count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    grand_count = m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)

    row_mean = x0.sum(dim=2, keepdim=True) / row_count
    col_mean = x0.sum(dim=1, keepdim=True) / col_count
    grand_mean = x0.sum(dim=(1, 2), keepdim=True) / grand_count

    centered = (x - row_mean - col_mean + grand_mean) * m
    return centered


def _masked_double_center_vector(
    x: torch.Tensor,                 # [B,A,K,H]
    pair_mask: torch.Tensor,         # [B,A,K]
) -> torch.Tensor:
    """Vector-valued version of masked two-way double centering."""
    m = pair_mask.unsqueeze(-1).to(x.dtype)
    x0 = x * m

    row_count = m.sum(dim=2, keepdim=True).clamp_min(1.0)
    col_count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    grand_count = m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)

    row_mean = x0.sum(dim=2, keepdim=True) / row_count
    col_mean = x0.sum(dim=1, keepdim=True) / col_count
    grand_mean = x0.sum(dim=(1, 2), keepdim=True) / grand_count

    centered = (x - row_mean - col_mean + grand_mean) * m
    return centered


class StrictCenteredARv2B1(nn.Module):
    """
    Double-centered interaction bottleneck.

    Inputs
    ------
    atom_tokens:
        [N_atom_total,H], frozen Drug3D node tokens.
    atom_batch:
        [N_atom_total].
    heavy_atom_indices:
        [B,Amax], GLOBAL indices into atom_tokens.
    heavy_atom_mask:
        [B,Amax].
    residue_tokens:
        [N_res_total,H], frozen Protein3D residue tokens.
    residue_batch:
        [N_res_total].
    candidate_residue_indices:
        [B,Kcandidate], GLOBAL E2 selector indices, already sorted.
    candidate_residue_mask:
        [B,Kcandidate].

    B1 directly uses the first ar_top_k sorted candidates.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        interaction_dim: int = 128,
        ar_top_k: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim <= 0 or interaction_dim <= 0:
            raise ValueError("hidden_dim and interaction_dim must be positive.")
        if ar_top_k <= 0:
            raise ValueError(f"ar_top_k must be positive, got {ar_top_k}.")

        self.hidden_dim = int(hidden_dim)
        self.interaction_dim = int(interaction_dim)
        self.ar_top_k = int(ar_top_k)

        # Separate projections. bias=False keeps zero-preserving behavior cleaner.
        self.atom_proj = nn.Sequential(
            nn.Linear(hidden_dim, interaction_dim, bias=False),
            nn.LayerNorm(interaction_dim, elementwise_affine=False),
        )
        self.residue_proj = nn.Sequential(
            nn.Linear(hidden_dim, interaction_dim, bias=False),
            nn.LayerNorm(interaction_dim, elementwise_affine=False),
        )

        # Strict multiplicative pair score. No unary concatenation.
        self.pair_score_mlp = nn.Sequential(
            nn.Linear(interaction_dim, interaction_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, 1, bias=False),
        )

        # Pair embedding, also strict multiplicative.
        self.pair_edge_mlp = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

    @staticmethod
    def _validate_global_indices(
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
        owner_batch: torch.Tensor,
        name: str,
    ) -> None:
        if not valid_mask.any():
            return

        safe = indices.clamp_min(0)
        if int(safe[valid_mask].max().item()) >= owner_batch.numel():
            raise IndexError(
                f"{name}: mapping index exceeds node count "
                f"(max={int(safe[valid_mask].max().item())}, "
                f"num_nodes={owner_batch.numel()})."
            )

        batch_size = indices.size(0)
        expected = torch.arange(
            batch_size,
            device=indices.device,
        ).view(batch_size, *([1] * (indices.dim() - 1)))

        actual = owner_batch[safe]
        bad = valid_mask & (actual != expected)
        if bad.any():
            first = torch.nonzero(bad, as_tuple=False)[0].tolist()
            raise ValueError(
                f"{name}: index points to another batch sample at {first}."
            )

    @staticmethod
    def _center_residual_max(
        centered_score: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Max absolute row/column mean after centering.
        Used only as a numerical correctness diagnostic.
        """
        pair_mask = atom_mask.unsqueeze(-1) & residue_mask.unsqueeze(1)
        m = pair_mask.to(centered_score.dtype)

        row_count = m.sum(2).clamp_min(1.0)
        col_count = m.sum(1).clamp_min(1.0)

        row_mean = (
            (centered_score * m).sum(2) / row_count
        )
        col_mean = (
            (centered_score * m).sum(1) / col_count
        )

        row_mean = row_mean * atom_mask.to(row_mean.dtype)
        col_mean = col_mean * residue_mask.to(col_mean.dtype)

        return torch.maximum(
            row_mean.abs().amax(),
            col_mean.abs().amax(),
        )

    def forward(
        self,
        atom_tokens: torch.Tensor,
        atom_batch: torch.Tensor,
        heavy_atom_indices: torch.Tensor,
        heavy_atom_mask: torch.Tensor,
        residue_tokens: torch.Tensor,
        residue_batch: torch.Tensor,
        candidate_residue_indices: torch.Tensor,
        candidate_residue_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        heavy_atom_indices = heavy_atom_indices.long()
        heavy_atom_mask = (
            heavy_atom_mask.bool() & (heavy_atom_indices >= 0)
        )
        candidate_residue_indices = candidate_residue_indices.long()
        candidate_residue_mask = (
            candidate_residue_mask.bool()
            & (candidate_residue_indices >= 0)
        )

        batch_size = heavy_atom_indices.size(0)
        if candidate_residue_indices.size(0) != batch_size:
            raise ValueError(
                "Heavy-atom and candidate-residue batch sizes differ."
            )

        if candidate_residue_indices.size(1) < self.ar_top_k:
            raise ValueError(
                f"E2 candidate width={candidate_residue_indices.size(1)} "
                f"is smaller than ar_top_k={self.ar_top_k}."
            )

        # B1: FIXED proposal -> first sorted Top-24 only.
        selected_global_idx = candidate_residue_indices[
            :, : self.ar_top_k
        ]
        selected_mask = candidate_residue_mask[
            :, : self.ar_top_k
        ]

        self._validate_global_indices(
            heavy_atom_indices,
            heavy_atom_mask,
            atom_batch,
            "heavy_atom_indices",
        )
        self._validate_global_indices(
            selected_global_idx,
            selected_mask,
            residue_batch,
            "selected_residue_indices",
        )

        if (~heavy_atom_mask.any(dim=1)).any():
            bad = torch.nonzero(
                ~heavy_atom_mask.any(dim=1),
                as_tuple=False,
            ).view(-1)
            raise ValueError(
                f"Samples with no valid heavy atoms: "
                f"{bad.detach().cpu().tolist()}"
            )
        if (~selected_mask.any(dim=1)).any():
            bad = torch.nonzero(
                ~selected_mask.any(dim=1),
                as_tuple=False,
            ).view(-1)
            raise ValueError(
                f"Samples with no valid selected residues: "
                f"{bad.detach().cpu().tolist()}"
            )

        safe_atom = heavy_atom_indices.clamp_min(0)
        safe_res = selected_global_idx.clamp_min(0)

        atom_raw = atom_tokens[safe_atom]             # [B,A,H]
        residue_raw = residue_tokens[safe_res]        # [B,K,H]

        atom_raw = (
            atom_raw
            * heavy_atom_mask.unsqueeze(-1).to(atom_raw.dtype)
        )
        residue_raw = (
            residue_raw
            * selected_mask.unsqueeze(-1).to(residue_raw.dtype)
        )

        q = self.atom_proj(atom_raw)                   # [B,A,D]
        k = self.residue_proj(residue_raw)             # [B,K,D]

        pair_mask = (
            heavy_atom_mask.unsqueeze(-1)
            & selected_mask.unsqueeze(1)
        )

        # Strict multiplicative pair feature.
        product = q.unsqueeze(2) * k.unsqueeze(1)     # [B,A,K,D]

        raw_score = (
            product.sum(dim=-1) / math.sqrt(self.interaction_dim)
            + self.pair_score_mlp(product).squeeze(-1)
        )
        raw_score = raw_score * pair_mask.to(raw_score.dtype)

        # Remove atom-only / residue-only additive score effects.
        interaction_score = _masked_double_center_scalar(
            raw_score,
            pair_mask,
        )

        # Global pair attention from interaction-only score.
        alpha = masked_softmax(
            interaction_score.flatten(1),
            pair_mask.flatten(1),
            dim=-1,
        ).view_as(interaction_score)

        valid_pair_count = pair_mask.sum(
            dim=(1, 2),
            keepdim=True,
        ).clamp_min(1).to(alpha.dtype)

        uniform = (
            pair_mask.to(alpha.dtype)
            / valid_pair_count
        )

        # Center against the exact uniform null.
        centered_weight = alpha - uniform
        centered_weight = (
            centered_weight
            * pair_mask.to(centered_weight.dtype)
        )

        # Strict multiplicative pair embedding + double centering.
        raw_edge = self.pair_edge_mlp(product)         # [B,A,K,H]
        interaction_edge = _masked_double_center_vector(
            raw_edge,
            pair_mask,
        )

        # The only AR2 bottleneck seen by the residual head.
        z_ar2 = (
            centered_weight.unsqueeze(-1)
            * interaction_edge
        ).sum(dim=(1, 2))                              # [B,H]

        # Diagnostics.
        entropy_num = -(
            alpha.clamp_min(1e-12).log()
            * alpha
            * pair_mask.to(alpha.dtype)
        ).sum(dim=(1, 2))
        valid_n = pair_mask.sum(
            dim=(1, 2),
        ).clamp_min(2).to(alpha.dtype)
        normalized_entropy = entropy_num / torch.log(valid_n)

        tv_from_uniform = 0.5 * centered_weight.abs().sum(
            dim=(1, 2)
        )
        centered_weight_l1 = centered_weight.abs().sum(
            dim=(1, 2)
        )
        z_norm = z_ar2.norm(dim=-1)

        center_residual_max = self._center_residual_max(
            interaction_score,
            heavy_atom_mask,
            selected_mask,
        )

        return {
            "z_ar2": z_ar2,
            "raw_pair_scores": raw_score,
            "interaction_pair_scores": interaction_score,
            "pair_mask": pair_mask,
            "pair_attention": alpha,
            "centered_pair_weight": centered_weight,
            "interaction_pair_edge": interaction_edge,
            "selected_residue_indices": selected_global_idx,
            "selected_residue_mask": selected_mask,
            "normalized_pair_entropy": normalized_entropy,
            "tv_from_uniform": tv_from_uniform,
            "centered_weight_l1": centered_weight_l1,
            "z_norm": z_norm,
            "center_residual_max": center_residual_max,
            "heavy_atom_mask": heavy_atom_mask,
        }


class E2StrictARv2B1StageA(nn.Module):
    """
    Frozen E2 predictor + trainable B1 interaction residual.

        final_pred = e2_pred + delta_AR2
        delta_AR2 = H(z_AR2)

    H is bias-free, so z_AR2 == 0 => delta_AR2 == 0 exactly.
    """

    def __init__(
        self,
        drug_1d_in_dim: int = 768,
        drug_3d_node_in_dim: int = 10,
        protein_1d_in_dim: int = 1280,
        protein_3d_node_s_dim: int = 6,
        protein_3d_node_v_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        task: str = "regression",
        candidate_top_k: int = 64,
        old_interaction_heads: int = 4,
        interaction_dim: int = 128,
        ar_top_k: int = 24,
        freeze_e2: bool = True,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.candidate_top_k = int(candidate_top_k)
        self.ar_top_k = int(ar_top_k)

        self.e2 = MyModelMDTAP13DFineGrained(
            drug_1d_in_dim=drug_1d_in_dim,
            drug_3d_node_in_dim=drug_3d_node_in_dim,
            protein_1d_in_dim=protein_1d_in_dim,
            protein_3d_node_s_dim=protein_3d_node_s_dim,
            protein_3d_node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
            pocket_top_k=candidate_top_k,
            interaction_heads=old_interaction_heads,
        )

        self.ar2 = StrictCenteredARv2B1(
            hidden_dim=hidden_dim,
            interaction_dim=interaction_dim,
            ar_top_k=ar_top_k,
            dropout=dropout,
        )

        # STRICT ZERO-PRESERVING residual head.
        self.ar2_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

        # Keep initial FINAL == E2.
        nn.init.zeros_(self.ar2_delta_head[-1].weight)

        self.freeze_e2 = bool(freeze_e2)
        self.set_e2_frozen(self.freeze_e2)

    def set_e2_frozen(self, frozen: bool = True) -> None:
        self.freeze_e2 = bool(frozen)
        for p in self.e2.parameters():
            p.requires_grad_(not self.freeze_e2)
        if self.freeze_e2:
            self.e2.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_e2:
            self.e2.eval()
        return self

    def load_e2_checkpoint_state(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        self.e2.load_state_dict(state_dict, strict=True)

    def zero_input_delta(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        z = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
        )
        return self.ar2_delta_head(z)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_details: bool = False,
    ):
        if self.freeze_e2:
            with torch.no_grad():
                e2 = self.e2(batch, return_debug=True)
        else:
            e2 = self.e2(batch, return_debug=True)

        ar2 = self.ar2(
            atom_tokens=e2["drug_atom_tokens"],
            atom_batch=e2["drug_atom_batch"],
            heavy_atom_indices=batch["heavy_atom_indices"],
            heavy_atom_mask=batch["heavy_atom_mask"],
            residue_tokens=e2["protein_residue_tokens"],
            residue_batch=e2["protein_residue_batch"],
            candidate_residue_indices=e2["pocket_indices"],
            candidate_residue_mask=e2["pocket_mask"],
        )

        delta = self.ar2_delta_head(ar2["z_ar2"])
        delta = delta.view_as(e2["pred"])
        pred = e2["pred"] + delta

        if not return_details:
            return pred

        return {
            "pred": pred,
            "e2_pred": e2["pred"],
            "base_pred": e2["base_pred"],
            "old_ar_delta": e2["local_delta"],
            "ar2_delta": delta,
            "ar2_feat": ar2["z_ar2"],
            "ar2_raw_pair_scores": ar2["raw_pair_scores"],
            "ar2_interaction_pair_scores": ar2[
                "interaction_pair_scores"
            ],
            "ar2_pair_mask": ar2["pair_mask"],
            "ar2_pair_attention": ar2["pair_attention"],
            "ar2_centered_pair_weight": ar2[
                "centered_pair_weight"
            ],
            "ar2_selected_residue_indices": ar2[
                "selected_residue_indices"
            ],
            "ar2_selected_residue_mask": ar2[
                "selected_residue_mask"
            ],
            "ar2_normalized_pair_entropy": ar2[
                "normalized_pair_entropy"
            ],
            "ar2_tv_from_uniform": ar2["tv_from_uniform"],
            "ar2_centered_weight_l1": ar2[
                "centered_weight_l1"
            ],
            "ar2_z_norm": ar2["z_norm"],
            "ar2_center_residual_max": ar2[
                "center_residual_max"
            ],
        }
