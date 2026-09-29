# -*- coding: utf-8 -*-
"""
AR2-A1: Frozen E2 backbone + Strict Cross-Only Atom-Residue residual.

Design
------
1) Keep the validated fold-specific E2 checkpoint intact and frozen.
2) Reuse ONLY E2's trained drug-conditioned Top-64 selector as a high-recall
   candidate proposal. Only its selected residue INDICES are reused.
   The old weighted pocket tokens / old AR scores are NOT used by AR2.
3) Drug side: use frozen Drug3D-EGNN node tokens, but only heavy atoms.
   Heavy-atom indices are supplied by the already validated
   drug_functional_groups_v2 cache, which is aligned to E2 drug_3d node order.
4) Protein side: use frozen Protein3D-EGNN raw residue node tokens.
5) New strict cross-only reranking:
      heavy atoms x 64 candidate residues -> pair compatibility
      -> residue log-mean-exp score -> hard Top-24 AR residues.
6) New bidirectional conditional interaction:
      atom -> residue softmax AND residue -> atom softmax.
7) AR2 residual head sees ONLY z_AR2:
      final_pred = e2_pred + H_AR2(z_AR2)
   No direct D / P / old-z_AR shortcut.
8) Last residual layer is zero-initialized, so before training:
      final_pred == e2_pred exactly.

Important
---------
- AR2 does NOT use intermolecular Euclidean distances because Davis drug SDF
  and protein structures are in independent coordinate frames.
- The old E2 local-interaction branch remains inside the frozen backbone only
  because it is part of the checkpoint predictor. Its scores/features are never
  used as AR2 evidence.
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
    """Softmax on valid entries; fully masked rows become exactly zero."""
    mask = mask.bool()
    masked = score.masked_fill(~mask, -1e9)
    weight = torch.softmax(masked, dim=dim)
    weight = weight * mask.to(weight.dtype)
    return weight / weight.sum(dim=dim, keepdim=True).clamp_min(eps)


def masked_logmeanexp(
    score: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    log(mean(exp(score))) over valid entries.
    Unlike logsumexp, this does not reward a drug merely for having more atoms.
    Fully masked positions are returned as -inf.
    """
    mask = mask.bool()
    masked = score.masked_fill(~mask, -torch.inf)
    lse = torch.logsumexp(masked, dim=dim)
    count = mask.sum(dim=dim).clamp_min(1).to(score.dtype)
    out = lse - torch.log(count)
    any_valid = mask.any(dim=dim)
    return torch.where(
        any_valid,
        out,
        torch.full_like(out, -torch.inf),
    )


class StrictCrossARv2(nn.Module):
    """
    Strict pair-conditioned atom-residue interaction.

    Inputs
    ------
    atom_tokens:
        [N_atom_total, H] frozen E2 Drug3D node tokens.
    atom_batch:
        [N_atom_total].
    heavy_atom_indices:
        [B, A_hmax], GLOBAL indices into atom_tokens.
    heavy_atom_mask:
        [B, A_hmax].
    residue_tokens:
        [N_res_total, H] frozen E2 Protein3D node tokens.
    residue_batch:
        [N_res_total].
    candidate_residue_indices:
        [B, K_candidate], GLOBAL indices into residue_tokens.
    candidate_residue_mask:
        [B, K_candidate].

    Returns
    -------
    Dict with:
        z_ar2                    [B,H]
        pair_scores              [B,A_hmax,K_ar]
        pair_mask                [B,A_hmax,K_ar]
        joint_pair_attention     [B,A_hmax,K_ar]
        atom_to_residue_attention[B,A_hmax,K_ar]
        residue_to_atom_attention[B,A_hmax,K_ar]
        candidate_scores         [B,K_candidate]
        selected_residue_indices [B,K_ar] global residue indices
        selected_residue_mask    [B,K_ar]
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

        # Separate node projections: same dimensionality, different semantics.
        # bias=False avoids creating a compatibility score from a learned unary bias.
        self.atom_proj = nn.Sequential(
            nn.Linear(hidden_dim, interaction_dim, bias=False),
            nn.LayerNorm(interaction_dim),
        )
        self.residue_proj = nn.Sequential(
            nn.Linear(hidden_dim, interaction_dim, bias=False),
            nn.LayerNorm(interaction_dim),
        )

        self.atom_value = nn.Linear(hidden_dim, interaction_dim, bias=False)
        self.residue_value = nn.Linear(hidden_dim, interaction_dim, bias=False)

        # STRICT CROSS-ONLY pair scorer:
        # input = [q_a * k_r, |q_a-k_r|], never [q_a, k_r] separately.
        self.pair_score_mlp = nn.Sequential(
            nn.Linear(interaction_dim * 2, interaction_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, 1),
        )

        # Turn pair-conditioned contexts into atom-side / residue-side interaction tokens.
        self.atom_relation_mlp = nn.Sequential(
            nn.Linear(interaction_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.residue_relation_mlp = nn.Sequential(
            nn.Linear(interaction_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Pooling scores are computed from already pair-conditioned relation tokens.
        self.atom_pool_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.residue_pool_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Final AR representation is again cross-only between the two directions.
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
                f"{name}: index points to a node belonging to another batch "
                f"sample at {first}."
            )

    def _pair_score(
        self,
        atom_q: torch.Tensor,       # [B,A,D]
        residue_k: torch.Tensor,    # [B,K,D]
        pair_mask: torch.Tensor,    # [B,A,K]
    ) -> torch.Tensor:
        a = atom_q.unsqueeze(2)         # [B,A,1,D]
        r = residue_k.unsqueeze(1)      # [B,1,K,D]

        product = a * r
        absdiff = torch.abs(a - r)
        cross = torch.cat([product, absdiff], dim=-1)

        nonlinear = self.pair_score_mlp(cross).squeeze(-1)
        bilinear = (a * r).sum(dim=-1) / math.sqrt(self.interaction_dim)
        score = bilinear + nonlinear
        return score.masked_fill(~pair_mask, -1e9)

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
        heavy_atom_mask = heavy_atom_mask.bool() & (heavy_atom_indices >= 0)
        candidate_residue_indices = candidate_residue_indices.long()
        candidate_residue_mask = (
            candidate_residue_mask.bool() & (candidate_residue_indices >= 0)
        )

        batch_size = heavy_atom_indices.size(0)
        if candidate_residue_indices.size(0) != batch_size:
            raise ValueError("Heavy-atom and candidate-residue batch sizes differ.")

        self._validate_global_indices(
            heavy_atom_indices,
            heavy_atom_mask,
            atom_batch,
            "heavy_atom_indices",
        )
        self._validate_global_indices(
            candidate_residue_indices,
            candidate_residue_mask,
            residue_batch,
            "candidate_residue_indices",
        )

        if (~heavy_atom_mask.any(dim=1)).any():
            bad = torch.nonzero(~heavy_atom_mask.any(dim=1), as_tuple=False).view(-1)
            raise ValueError(
                f"Samples with no valid heavy atoms: {bad.detach().cpu().tolist()}"
            )
        if (~candidate_residue_mask.any(dim=1)).any():
            bad = torch.nonzero(
                ~candidate_residue_mask.any(dim=1), as_tuple=False
            ).view(-1)
            raise ValueError(
                f"Samples with no valid candidate residues: "
                f"{bad.detach().cpu().tolist()}"
            )

        # Gather frozen contextualized heavy-atom / residue tokens.
        safe_atom = heavy_atom_indices.clamp_min(0)
        safe_res = candidate_residue_indices.clamp_min(0)

        heavy_raw = atom_tokens[safe_atom]                 # [B,A,H]
        candidate_raw = residue_tokens[safe_res]           # [B,Kc,H]

        heavy_raw = (
            heavy_raw
            * heavy_atom_mask.unsqueeze(-1).to(heavy_raw.dtype)
        )
        candidate_raw = (
            candidate_raw
            * candidate_residue_mask.unsqueeze(-1).to(candidate_raw.dtype)
        )

        atom_q = self.atom_proj(heavy_raw)                 # [B,A,D]
        candidate_k = self.residue_proj(candidate_raw)     # [B,Kc,D]

        candidate_pair_mask = (
            heavy_atom_mask.unsqueeze(-1)
            & candidate_residue_mask.unsqueeze(1)
        )
        candidate_pair_scores = self._pair_score(
            atom_q,
            candidate_k,
            candidate_pair_mask,
        )                                                   # [B,A,Kc]

        # Cross-only reranking of the 64 high-recall proposal residues.
        # For every candidate residue, aggregate compatibility across heavy atoms.
        candidate_scores = masked_logmeanexp(
            candidate_pair_scores,
            candidate_pair_mask,
            dim=1,
        )                                                   # [B,Kc]

        k_ar = min(self.ar_top_k, candidate_scores.size(1))
        top_scores, top_pos = torch.topk(
            candidate_scores,
            k=k_ar,
            dim=1,
            largest=True,
            sorted=True,
        )

        selected_mask = torch.gather(
            candidate_residue_mask,
            1,
            top_pos,
        )
        selected_global_idx = torch.gather(
            candidate_residue_indices,
            1,
            top_pos,
        )

        gather_h = top_pos.unsqueeze(-1).expand(
            -1, -1, self.hidden_dim
        )
        selected_raw = torch.gather(
            candidate_raw,
            1,
            gather_h,
        )                                                   # [B,Kar,H]

        selected_k = self.residue_proj(selected_raw)        # [B,Kar,D]
        atom_v = self.atom_value(heavy_raw)                 # [B,A,D]
        residue_v = self.residue_value(selected_raw)        # [B,Kar,D]

        pair_mask = (
            heavy_atom_mask.unsqueeze(-1)
            & selected_mask.unsqueeze(1)
        )
        pair_scores = self._pair_score(
            atom_q,
            selected_k,
            pair_mask,
        )                                                   # [B,A,Kar]

        # Direction 1: each atom asks which residues are compatible with it.
        alpha_r_given_a = masked_softmax(
            pair_scores,
            pair_mask,
            dim=-1,
        )
        atom_context = torch.einsum(
            "bak,bkd->bad",
            alpha_r_given_a,
            residue_v,
        )
        atom_relation_input = torch.cat(
            [
                atom_q * atom_context,
                torch.abs(atom_q - atom_context),
            ],
            dim=-1,
        )
        atom_relation = self.atom_relation_mlp(atom_relation_input)
        atom_relation = (
            atom_relation
            * heavy_atom_mask.unsqueeze(-1).to(atom_relation.dtype)
        )

        # Direction 2: each residue asks which atoms are compatible with it.
        beta_a_given_r = masked_softmax(
            pair_scores,
            pair_mask,
            dim=1,
        )
        residue_context = torch.einsum(
            "bak,bad->bkd",
            beta_a_given_r,
            atom_v,
        )
        residue_relation_input = torch.cat(
            [
                selected_k * residue_context,
                torch.abs(selected_k - residue_context),
            ],
            dim=-1,
        )
        residue_relation = self.residue_relation_mlp(
            residue_relation_input
        )
        residue_relation = (
            residue_relation
            * selected_mask.unsqueeze(-1).to(residue_relation.dtype)
        )

        # Pool pair-conditioned atom-side representations.
        atom_gate = self.atom_pool_score(atom_relation).squeeze(-1)
        atom_weight = masked_softmax(
            atom_gate,
            heavy_atom_mask,
            dim=-1,
        )
        z_atom = torch.sum(
            atom_weight.unsqueeze(-1) * atom_relation,
            dim=1,
        )

        # Pool pair-conditioned residue-side representations.
        residue_gate = self.residue_pool_score(
            residue_relation
        ).squeeze(-1)
        residue_weight = masked_softmax(
            residue_gate,
            selected_mask,
            dim=-1,
        )
        z_residue = torch.sum(
            residue_weight.unsqueeze(-1) * residue_relation,
            dim=1,
        )

        # STRICT CROSS-ONLY final AR representation.
        z_ar2 = self.out_mlp(
            torch.cat(
                [
                    z_atom * z_residue,
                    torch.abs(z_atom - z_residue),
                ],
                dim=-1,
            )
        )

        # Separate joint normalization only as a 2D interaction-evidence matrix.
        # It is NOT the main pooling mechanism used to construct z_AR2.
        joint_pair_attention = masked_softmax(
            pair_scores.flatten(1),
            pair_mask.flatten(1),
            dim=-1,
        ).view_as(pair_scores)

        # Useful diagnostics.
        valid_joint = joint_pair_attention[pair_mask]
        if valid_joint.numel() > 0:
            # Mean normalized entropy per sample is computed below more carefully.
            pass

        entropy_num = -(
            joint_pair_attention.clamp_min(1e-12).log()
            * joint_pair_attention
            * pair_mask.to(joint_pair_attention.dtype)
        ).sum(dim=(1, 2))
        pair_count = pair_mask.sum(dim=(1, 2)).clamp_min(2).to(
            joint_pair_attention.dtype
        )
        normalized_joint_entropy = entropy_num / torch.log(pair_count)

        return {
            "z_ar2": z_ar2,
            "pair_scores": pair_scores,
            "pair_mask": pair_mask,
            "joint_pair_attention": joint_pair_attention,
            "atom_to_residue_attention": alpha_r_given_a,
            "residue_to_atom_attention": beta_a_given_r,
            "atom_pool_weight": atom_weight,
            "residue_pool_weight": residue_weight,
            "candidate_scores": candidate_scores,
            "candidate_pair_scores": candidate_pair_scores,
            "selected_candidate_position": top_pos,
            "selected_residue_indices": selected_global_idx,
            "selected_residue_mask": selected_mask,
            "selected_residue_scores": top_scores,
            "heavy_atom_mask": heavy_atom_mask,
            "normalized_joint_entropy": normalized_joint_entropy,
        }


class E2StrictARv2StageA(nn.Module):
    """
    Stage AR2-A1:
        frozen original E2 predictor + trainable strict AR2 residual.

    The original E2 remains untouched for exact checkpoint compatibility.
    Only the old selector INDICES are reused as the Top-64 proposal pool.

    final = e2_pred + ar2_delta
    ar2_delta = H(z_AR2)        # STRICT: z_AR2 only
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

        # Original fold-specific E2 class, unchanged.
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

        self.ar2 = StrictCrossARv2(
            hidden_dim=hidden_dim,
            interaction_dim=interaction_dim,
            ar_top_k=ar_top_k,
            dropout=dropout,
        )

        # STRICT NO-SHORTCUT residual head: only z_AR2 is visible.
        self.ar2_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.ar2_delta_head[-1].weight)
        nn.init.zeros_(self.ar2_delta_head[-1].bias)

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

        # Use ONLY:
        # - frozen contextualized atom/residue node tokens,
        # - old selector Top-64 INDICES/MASK as candidate proposal.
        # Never use old pocket_tokens, interaction_scores, or old local feature
        # as evidence for AR2.
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

        ar2_delta = self.ar2_delta_head(ar2["z_ar2"])
        ar2_delta = ar2_delta.view_as(e2["pred"])
        pred = e2["pred"] + ar2_delta

        if not return_details:
            return pred

        return {
            "pred": pred,
            "e2_pred": e2["pred"],
            "base_pred": e2["base_pred"],
            "old_ar_delta": e2["local_delta"],
            "ar2_delta": ar2_delta,
            "ar2_feat": ar2["z_ar2"],
            "ar2_pair_scores": ar2["pair_scores"],
            "ar2_pair_mask": ar2["pair_mask"],
            "ar2_joint_pair_attention": ar2["joint_pair_attention"],
            "ar2_atom_to_residue_attention": ar2[
                "atom_to_residue_attention"
            ],
            "ar2_residue_to_atom_attention": ar2[
                "residue_to_atom_attention"
            ],
            "ar2_selected_residue_indices": ar2[
                "selected_residue_indices"
            ],
            "ar2_selected_residue_mask": ar2[
                "selected_residue_mask"
            ],
            "ar2_candidate_scores": ar2["candidate_scores"],
            "ar2_normalized_joint_entropy": ar2[
                "normalized_joint_entropy"
            ],
            "drug_atom_tokens": e2["drug_atom_tokens"],
            "drug_atom_batch": e2["drug_atom_batch"],
            "protein_residue_tokens": e2["protein_residue_tokens"],
            "protein_residue_batch": e2["protein_residue_batch"],
            "old_candidate_indices": e2["pocket_indices"],
            "old_candidate_mask": e2["pocket_mask"],
        }
