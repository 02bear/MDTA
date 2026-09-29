# -*- coding: utf-8 -*-
"""
E2-guided hierarchical three-granularity DTA, V1.

Core rule
---------
Keep the already validated E2 model intact:

    Global:
        drug_1d + drug_3d -> drug_feat
        protein_1d + protein_3d -> protein_feat
        base_pred = Decoder([drug_feat, protein_feat])

    Fine / AR (E2):
        drug-global-conditioned Top-K residue selection
        all drug atoms x Top-K residues -> local_interaction_feat
        e2_pred = base_pred + local_delta

Add only one new Meso / FP residual branch:

    1) Reuse E2 atom tokens and E2 atom-residue interaction scores.
    2) Aggregate E2 AR evidence by BRICS fragment.
    3) Build CAVIAR pocket representations from E2 protein residue tokens.
    4) Learn fragment-pocket compatibility.
    5) Pool FP evidence and predict fp_delta.
    6) final_pred = e2_pred + fp_delta.

Important:
- Stage A freezes the entire pretrained E2 branch. E2 is used only as a fixed
  feature/prediction provider while the new FP branch learns a residual correction.
- No all-residue raw-score pocket prior in V1.
- No Top-K/CAVIAR overlap feature in V1.
- No repeated atom x CAVIAR-residue interaction.
- No Transformer / MoE / PCL / auxiliary affinity heads / extra loss.
- fp_delta_head is zero initialized, so before training:
      final_pred == e2_pred
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.model_p13d_finegrained_residual import (
    MyModelMDTAP13DFineGrained,
)


def masked_softmax(
    score: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Numerically safe softmax that returns zero on fully masked rows."""
    mask = mask.bool()
    masked = score.masked_fill(~mask, -1e9)
    weight = torch.softmax(masked, dim=dim)
    weight = weight * mask.to(weight.dtype)
    return weight / weight.sum(dim=dim, keepdim=True).clamp_min(eps)


class MaskedAttentionPool(nn.Module):
    """Attention pooling over a padded token dimension."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_dim = dim if dim >= 0 else x.dim() + dim
        expected_dim = x.dim() - 2
        if normalized_dim != expected_dim:
            raise ValueError(
                "MaskedAttentionPool V1 only supports pooling over the "
                f"penultimate token dimension, got dim={dim} "
                f"(normalized={normalized_dim}) for x.shape={tuple(x.shape)}."
            )
        score = self.score(x).squeeze(-1)
        weight = masked_softmax(score, mask, dim=-1)
        pooled = (weight.unsqueeze(-1) * x).sum(dim=-2)
        return pooled, weight


class E2GuidedFragmentPocketRefiner(nn.Module):
    """
    Build meso-scale fragment-pocket evidence from E2 outputs.

    Required batch fields:
        fragment_atom_indices:       [B,F,A_f] global indices into E2 atom tokens
        fragment_atom_mask:          [B,F,A_f]
        fragment_mask:               [B,F]
        pocket_residue_indices:      [B,P,R_p] global indices into E2 residue tokens
        pocket_residue_mask:         [B,P,R_p]
        pocket_mask:                 [B,P]

    Required E2 debug fields:
        drug_atom_tokens             [N_atom,H]
        drug_atom_batch              [N_atom]
        protein_residue_tokens       [N_res,H]
        protein_residue_batch        [N_res]
        pocket_tokens                [B,K,H]    (E2 selected/weighted residues)
        pocket_mask                  [B,K]      (E2 Top-K mask)
        interaction_scores           [B,Amax,K]
        atom_mask                    [B,Amax]
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.fragment_pool = MaskedAttentionPool(hidden_dim, dropout)
        self.pocket_pool = MaskedAttentionPool(hidden_dim, dropout)

        # E2 AR-guided fragment feature:
        # [fragment_feat, fragment_residue_context,
        #  product, abs_diff, fragment_ar_mass]
        self.fragment_ar_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # True meso FP pair. No repeated atom x CAVIAR-residue AR here.
        # [fragment, pocket, product, abs_diff, fragment_AR]
        self.fp_edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.fp_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Explicit NULL pocket lets a BRICS fragment choose "no structural pocket".
        self.null_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.fragment_joint = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.fragment_evidence_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _graph_offsets(batch_index: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        counts = torch.bincount(batch_index, minlength=batch_size)
        offsets = torch.cumsum(counts, dim=0) - counts
        return counts, offsets

    @staticmethod
    def _validate_global_indices(
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
        owner_batch: torch.Tensor,
        name: str,
    ) -> None:
        """Ensure a mapping never points into another sample in the batch."""
        if not valid_mask.any():
            return
        safe = indices.clamp_min(0)
        if int(safe[valid_mask].max().item()) >= owner_batch.numel():
            raise IndexError(
                f"{name}: mapping index exceeds available node count: "
                f"max={int(safe[valid_mask].max().item())}, "
                f"num_nodes={owner_batch.numel()}"
            )
        batch_size = indices.size(0)
        expected = torch.arange(
            batch_size, device=indices.device
        ).view(batch_size, *([1] * (indices.dim() - 1)))
        actual = owner_batch[safe]
        bad = valid_mask & (actual != expected)
        if bad.any():
            first = torch.nonzero(bad, as_tuple=False)[0].tolist()
            raise ValueError(
                f"{name}: mapping points to a node belonging to another batch "
                f"sample at index {first}. This usually means BRICS/CAVIAR "
                f"indices are not aligned with the original E2 atom/residue order."
            )

    def _build_fragment_features(
        self,
        atom_tokens: torch.Tensor,
        atom_batch: torch.Tensor,
        e2_selected_residue_tokens: torch.Tensor,
        e2_selected_residue_mask: torch.Tensor,
        interaction_scores: torch.Tensor,
        e2_atom_mask: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fragment_indices = batch["fragment_atom_indices"].long()
        fragment_atom_mask = batch["fragment_atom_mask"].bool()
        fragment_mask = batch["fragment_mask"].bool()

        batch_size, num_fragments, max_fragment_atoms = fragment_indices.shape
        if e2_selected_residue_tokens.size(0) != batch_size:
            raise ValueError("E2 selected-residue batch size does not match fragment batch size.")

        valid_fragment_atom = (
            fragment_atom_mask
            & fragment_mask.unsqueeze(-1)
            & (fragment_indices >= 0)
        )
        self._validate_global_indices(
            fragment_indices,
            valid_fragment_atom,
            atom_batch,
            "fragment_atom_indices",
        )

        safe_global = fragment_indices.clamp_min(0)
        fragment_atoms = atom_tokens[safe_global]  # [B,F,A,H]
        fragment_atoms = fragment_atoms * valid_fragment_atom.unsqueeze(-1).to(fragment_atoms.dtype)
        fragment_feat, fragment_atom_weight = self.fragment_pool(
            fragment_atoms,
            valid_fragment_atom,
        )
        fragment_feat = fragment_feat * fragment_mask.unsqueeze(-1).to(fragment_feat.dtype)

        # Reconstruct the same global E2 pair attention mass from interaction_scores.
        # This is not a new AR network; it only reuses E2's already-computed scores.
        pair_mask = e2_atom_mask.unsqueeze(-1) & e2_selected_residue_mask.unsqueeze(1)
        pair_alpha = masked_softmax(
            interaction_scores.flatten(1),
            pair_mask.flatten(1),
            dim=-1,
        ).view_as(interaction_scores)
        atom_ar_mass = pair_alpha.sum(-1)  # [B,Amax], sum over all atoms ~= 1.

        # Also derive an E2-score-guided residue context for every dense atom.
        per_atom_residue_weight = masked_softmax(
            interaction_scores,
            pair_mask,
            dim=-1,
        )
        atom_residue_context = torch.einsum(
            "bak,bkh->bah",
            per_atom_residue_weight,
            e2_selected_residue_tokens,
        )  # [B,Amax,H]

        counts, offsets = self._graph_offsets(atom_batch, batch_size)
        local_indices = fragment_indices - offsets.view(batch_size, 1, 1)
        safe_local = local_indices.clamp_min(0)

        local_limit = counts.view(batch_size, 1, 1)
        local_bad = valid_fragment_atom & (
            (local_indices < 0) | (local_indices >= local_limit)
        )
        if local_bad.any():
            first = torch.nonzero(local_bad, as_tuple=False)[0].tolist()
            raise ValueError(
                "BRICS atom index cannot be mapped to E2 dense atom position "
                f"at {first}. Check atom ordering/alignment between "
                "drug_3d and drug_atom_v2 preprocessing."
            )

        batch_index = torch.arange(
            batch_size, device=atom_tokens.device
        ).view(batch_size, 1, 1).expand(
            batch_size, num_fragments, max_fragment_atoms
        )

        gathered_mass = atom_ar_mass[batch_index, safe_local]
        gathered_mass = gathered_mass * valid_fragment_atom.to(gathered_mass.dtype)

        gathered_residue_context = atom_residue_context[batch_index, safe_local]
        gathered_residue_context = (
            gathered_residue_context
            * valid_fragment_atom.unsqueeze(-1).to(gathered_residue_context.dtype)
        )

        fragment_ar_mass = gathered_mass.sum(-1)  # [B,F]

        # AR mass determines which atoms contributed strongly in E2.
        # Add a tiny uniform term so a numerically zero-mass fragment still has
        # a defined residue context.
        ar_atom_weight = gathered_mass + 1e-6 * valid_fragment_atom.to(gathered_mass.dtype)
        ar_atom_weight = ar_atom_weight / ar_atom_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        fragment_residue_context = (
            ar_atom_weight.unsqueeze(-1) * gathered_residue_context
        ).sum(-2)

        ar_input = torch.cat(
            [
                fragment_feat,
                fragment_residue_context,
                fragment_feat * fragment_residue_context,
                torch.abs(fragment_feat - fragment_residue_context),
                fragment_ar_mass.unsqueeze(-1),
            ],
            dim=-1,
        )
        fragment_ar_feat = self.fragment_ar_proj(ar_input)
        fragment_ar_feat = (
            fragment_ar_feat * fragment_mask.unsqueeze(-1).to(fragment_ar_feat.dtype)
        )

        return {
            "fragment_feat": fragment_feat,
            "fragment_ar_feat": fragment_ar_feat,
            "fragment_ar_mass": fragment_ar_mass,
            "fragment_atom_weight": fragment_atom_weight,
        }

    def _build_pocket_features(
        self,
        residue_tokens: torch.Tensor,
        residue_batch: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pocket_indices = batch["pocket_residue_indices"].long()
        pocket_residue_mask = batch["pocket_residue_mask"].bool()
        pocket_mask = batch["pocket_mask"].bool()

        valid_pocket_residue = (
            pocket_residue_mask
            & pocket_mask.unsqueeze(-1)
            & (pocket_indices >= 0)
        )
        self._validate_global_indices(
            pocket_indices,
            valid_pocket_residue,
            residue_batch,
            "pocket_residue_indices",
        )

        safe = pocket_indices.clamp_min(0)
        pocket_residues = residue_tokens[safe]  # [B,P,R,H]
        pocket_residues = (
            pocket_residues
            * valid_pocket_residue.unsqueeze(-1).to(pocket_residues.dtype)
        )
        pocket_feat, pocket_residue_weight = self.pocket_pool(
            pocket_residues,
            valid_pocket_residue,
        )
        pocket_feat = pocket_feat * pocket_mask.unsqueeze(-1).to(pocket_feat.dtype)

        return {
            "pocket_feat": pocket_feat,
            "pocket_residue_weight": pocket_residue_weight,
        }

    def forward(
        self,
        e2: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fragment = self._build_fragment_features(
            atom_tokens=e2["drug_atom_tokens"],
            atom_batch=e2["drug_atom_batch"],
            e2_selected_residue_tokens=e2["pocket_tokens"],
            e2_selected_residue_mask=e2["pocket_mask"],
            interaction_scores=e2["interaction_scores"],
            e2_atom_mask=e2["atom_mask"],
            batch=batch,
        )
        pocket = self._build_pocket_features(
            residue_tokens=e2["protein_residue_tokens"],
            residue_batch=e2["protein_residue_batch"],
            batch=batch,
        )

        fragment_feat = fragment["fragment_feat"]
        fragment_ar_feat = fragment["fragment_ar_feat"]
        pocket_feat = pocket["pocket_feat"]
        fragment_mask = batch["fragment_mask"].bool()
        pocket_mask = batch["pocket_mask"].bool()

        # [B,F,P,H]
        frag = fragment_feat.unsqueeze(2)
        ar = fragment_ar_feat.unsqueeze(2)
        pock = pocket_feat.unsqueeze(1)

        num_pockets = pocket_feat.size(1)
        frag_expand = frag.expand(-1, -1, num_pockets, -1)
        ar_expand = ar.expand(-1, -1, num_pockets, -1)
        pock_expand = pock.expand(-1, fragment_feat.size(1), -1, -1)

        edge_input = torch.cat(
            [
                frag_expand,
                pock_expand,
                frag_expand * pock_expand,
                torch.abs(frag_expand - pock_expand),
                ar_expand,
            ],
            dim=-1,
        )
        fp_edge = self.fp_edge_mlp(edge_input)
        fp_real_score = self.fp_score(fp_edge).squeeze(-1)

        real_mask = fragment_mask.unsqueeze(-1) & pocket_mask.unsqueeze(1)
        null_score = self.null_score(
            torch.cat([fragment_feat, fragment_ar_feat], dim=-1)
        ).squeeze(-1)

        all_score = torch.cat([fp_real_score, null_score.unsqueeze(-1)], dim=-1)
        all_mask = torch.cat(
            [real_mask, fragment_mask.unsqueeze(-1)],
            dim=-1,
        )
        fp_weight_all = masked_softmax(all_score, all_mask, dim=-1)
        fp_weight = fp_weight_all[..., :num_pockets]
        null_weight = fp_weight_all[..., -1]

        fp_context = (fp_weight.unsqueeze(-1) * fp_edge).sum(2)

        fragment_joint = self.fragment_joint(
            torch.cat(
                [fragment_feat, fragment_ar_feat, fp_context],
                dim=-1,
            )
        )
        fragment_joint = (
            fragment_joint * fragment_mask.unsqueeze(-1).to(fragment_joint.dtype)
        )

        fragment_evidence_score = self.fragment_evidence_score(
            fragment_joint
        ).squeeze(-1)
        fragment_evidence_weight = masked_softmax(
            fragment_evidence_score,
            fragment_mask,
            dim=-1,
        )
        fp_feat = (
            fragment_evidence_weight.unsqueeze(-1) * fragment_joint
        ).sum(1)

        # Diagnostics only.
        entropy = -(
            fp_weight_all.clamp_min(1e-8).log() * fp_weight_all
        ).sum(-1)
        valid_fragment_count = fragment_mask.sum().clamp_min(1)
        mean_fp_entropy = (
            entropy * fragment_mask.to(entropy.dtype)
        ).sum() / valid_fragment_count
        mean_null_weight = (
            null_weight * fragment_mask.to(null_weight.dtype)
        ).sum() / valid_fragment_count
        mean_fragment_ar_mass = (
            fragment["fragment_ar_mass"]
            * fragment_mask.to(fragment["fragment_ar_mass"].dtype)
        ).sum() / valid_fragment_count

        return {
            "fp_feat": fp_feat,
            "fragment_feat": fragment_feat,
            "fragment_ar_feat": fragment_ar_feat,
            "fragment_ar_mass": fragment["fragment_ar_mass"],
            "pocket_feat": pocket_feat,
            "fp_edge": fp_edge,
            "fp_weight": fp_weight,
            "null_weight": null_weight,
            "fragment_evidence_weight": fragment_evidence_weight,
            "mean_fp_entropy": mean_fp_entropy,
            "mean_null_weight": mean_null_weight,
            "mean_fragment_ar_mass": mean_fragment_ar_mass,
        }


class E2GuidedThreeGranularityDTA(nn.Module):
    """
    Stage-A model:
        frozen E2(Global + AR) + trainable FP residual.

    The pretrained E2 branch is strictly frozen and kept in eval mode.
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
        pocket_top_k: int = 64,
        interaction_heads: int = 4,
        freeze_e2: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        # The original E2 class is used as-is. This protects its parameter names,
        # checkpoint compatibility, and original forward computation.
        self.e2 = MyModelMDTAP13DFineGrained(
            drug_1d_in_dim=drug_1d_in_dim,
            drug_3d_node_in_dim=drug_3d_node_in_dim,
            protein_1d_in_dim=protein_1d_in_dim,
            protein_3d_node_s_dim=protein_3d_node_s_dim,
            protein_3d_node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
            pocket_top_k=pocket_top_k,
            interaction_heads=interaction_heads,
        )

        self.fp_refiner = E2GuidedFragmentPocketRefiner(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Residual correction to E2, not a second independent affinity predictor.
        self.fp_delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.fp_delta_head[-1].weight)
        nn.init.zeros_(self.fp_delta_head[-1].bias)

        self.freeze_e2 = bool(freeze_e2)
        self.set_e2_frozen(self.freeze_e2)

    def set_e2_frozen(self, frozen: bool = True) -> None:
        """Freeze/unfreeze the complete E2 branch.

        Stage A requires frozen=True. When frozen, E2 parameters receive no
        gradients and E2 is always kept in eval mode so Dropout cannot perturb
        the pretrained prediction/features.
        """
        self.freeze_e2 = bool(frozen)
        for parameter in self.e2.parameters():
            parameter.requires_grad_(not self.freeze_e2)
        if self.freeze_e2:
            self.e2.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_e2:
            self.e2.eval()
        return self

    def load_e2_checkpoint_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Strictly load an original E2 checkpoint into the unchanged E2 submodule."""
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

        fp = self.fp_refiner(e2, batch)

        fp_delta = self.fp_delta_head(
            torch.cat(
                [
                    e2["drug_feat"],
                    e2["protein_feat"],
                    e2["local_interaction_feat"],
                    fp["fp_feat"],
                ],
                dim=-1,
            )
        )
        fp_delta = fp_delta.view_as(e2["pred"])

        pred = e2["pred"] + fp_delta

        if not return_details:
            return pred

        return {
            "pred": pred,
            "e2_pred": e2["pred"],
            "base_pred": e2["base_pred"],
            "ar_delta": e2["local_delta"],
            "fp_delta": fp_delta,
            "drug_feat": e2["drug_feat"],
            "protein_feat": e2["protein_feat"],
            "ar_feat": e2["local_interaction_feat"],
            "fp_feat": fp["fp_feat"],
            "fragment_feat": fp["fragment_feat"],
            "fragment_ar_feat": fp["fragment_ar_feat"],
            "fragment_ar_mass": fp["fragment_ar_mass"],
            "pocket_feat": fp["pocket_feat"],
            "fp_weight": fp["fp_weight"],
            "null_weight": fp["null_weight"],
            "fragment_evidence_weight": fp["fragment_evidence_weight"],
            "mean_fp_entropy": fp["mean_fp_entropy"],
            "mean_null_weight": fp["mean_null_weight"],
            "mean_fragment_ar_mass": fp["mean_fragment_ar_mass"],
        }
