from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.model_p13d_triscale import (
    MyModelMDTAP13DTriScale,
    _masked_softmax,
    parse_active_scales,
)


BRANCH_NAMES = ("global", "fp", "ar", "joint")


class MyModelMDTAP13DProtectedTriScale(MyModelMDTAP13DTriScale):
    """From-scratch TriScale with a gradient-protected global predictor.

    The inherited class supplies the tested global path, tensor-index helpers and
    local modules. This class replaces the delta heads and old forward semantics
    with four complete predictions and a masked convex fusion.
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
        active_scales="global,fp,ar",
        cross_scale_mode="bottom_up",
        local_hidden_dim=128,
        local_dropout=0.1,
        max_fragments=12,
        max_atoms_per_fragment=24,
        max_pockets=3,
        max_residues_per_pocket=32,
        ar_region_chunk_size=32,
        top_k_fp_for_ar: Optional[int] = None,
        joint_recompute_fp_score: bool = False,
    ):
        super().__init__(
            drug_1d_in_dim=drug_1d_in_dim,
            drug_3d_node_in_dim=drug_3d_node_in_dim,
            protein_1d_in_dim=protein_1d_in_dim,
            protein_3d_node_s_dim=protein_3d_node_s_dim,
            protein_3d_node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
            active_scales=active_scales,
            cross_scale_mode=cross_scale_mode,
            local_hidden_dim=local_hidden_dim,
            local_dropout=local_dropout,
            max_fragments=max_fragments,
            max_atoms_per_fragment=max_atoms_per_fragment,
            max_pockets=max_pockets,
            max_residues_per_pocket=max_residues_per_pocket,
            ar_region_chunk_size=ar_region_chunk_size,
            top_k_fp_for_ar=top_k_fp_for_ar,
        )
        # Re-register the old joint MLP under an unambiguous complete-prediction
        # name, and remove every alpha/delta-era trainable parameter.
        self.joint_head = self.multiscale_head
        del self.multiscale_head
        del self.fp_alpha
        del self.ar_alpha
        del self.multiscale_alpha

        self.joint_recompute_fp_score = bool(joint_recompute_fp_score)
        self.joint_fp_score = (
            nn.Linear(local_hidden_dim, 1)
            if self.joint_recompute_fp_score
            else None
        )
        self.fusion_logits = nn.Parameter(
            torch.log(torch.tensor([0.55, 0.15, 0.15, 0.15]))
        )

    def _local_batched_protected(self, atom_nodes, protein_nodes, batch):
        """Batched FP plus region-chunked AR, preserving raw and joint FP."""
        local = self._get_local_tensors(batch)
        fragment_indices = local["fragment_atom_indices"]
        fragment_atom_mask = local["fragment_atom_mask"]
        fragment_mask = local["fragment_mask"]
        pocket_sequence = local["pocket_sequence_indices"]
        pocket_graph = local["pocket_graph_indices"]
        pocket_residue_mask = local["pocket_residue_mask"]
        pocket_mask = local["pocket_mask"]
        batch_size, num_fragments, num_atoms = fragment_indices.shape
        num_pockets, num_residues = pocket_sequence.shape[1:]

        atom_values, valid_atoms = self._gather_grouped(
            atom_nodes,
            fragment_indices,
            fragment_atom_mask,
            batch["drug_3d"]["batch"],
        )
        fragment_mask = fragment_mask & valid_atoms.any(dim=-1)
        atom_repr = self.atom_proj(atom_values)
        fragment_mean = (
            atom_repr * valid_atoms.unsqueeze(-1).to(atom_repr.dtype)
        ).sum(dim=2) / valid_atoms.sum(dim=2, keepdim=True).clamp_min(1).to(
            atom_repr.dtype
        )
        fragment_repr = self.fragment_projection(fragment_mean)

        token_values = local["protein_1d_token_values"]
        token_offsets = local["protein_1d_token_offsets"]
        token_lengths = local["protein_1d_token_lengths"]
        token_valid = (
            pocket_residue_mask
            & (pocket_sequence >= 0)
            & (pocket_sequence < token_lengths[:, None, None])
        )
        token_indices = (pocket_sequence + token_offsets[:, None, None]).clamp(
            0, max(token_values.shape[0] - 1, 0)
        )
        token_repr = token_values[token_indices]
        protein_3d_repr, graph_valid = self._gather_grouped(
            protein_nodes,
            pocket_graph,
            pocket_residue_mask,
            batch["protein_3d"]["batch"],
        )
        residue_mask = token_valid & graph_valid
        pocket_mask = pocket_mask & residue_mask.any(dim=-1)
        residue_repr = self.residue_norm(
            self.residue_fusion(
                torch.cat(
                    [self.res1_proj(token_repr), self.res3_proj(protein_3d_repr)],
                    dim=-1,
                )
            )
        )
        pocket_mean = (
            residue_repr * residue_mask.unsqueeze(-1).to(residue_repr.dtype)
        ).sum(dim=2) / residue_mask.sum(dim=2, keepdim=True).clamp_min(1).to(
            residue_repr.dtype
        )
        pocket_repr = self.pocket_projection(pocket_mean)

        fragment_grid = fragment_repr[:, :, None, :].expand(
            -1, -1, num_pockets, -1
        )
        pocket_grid = pocket_repr[:, None, :, :].expand(
            -1, num_fragments, -1, -1
        )
        fp_pair_input = self.pair_input(fragment_grid, pocket_grid)
        fp_pair_features_raw = self.fp_feature_mlp(fp_pair_input)
        fp_pair_scores = self.fp_score_mlp(fp_pair_input).squeeze(-1)
        region_mask = fragment_mask[:, :, None] & pocket_mask[:, None, :]
        fp_weights = _masked_softmax(
            fp_pair_scores.flatten(1), region_mask.flatten(1), dim=1
        )
        fp_repr_raw = (
            fp_weights.unsqueeze(-1) * fp_pair_features_raw.flatten(1, 2)
        ).sum(dim=1)
        fp_valid_mask = region_mask.flatten(1).any(dim=1)

        need_ar = "ar" in self.active_scales
        selected_ar_regions = (
            self._select_ar_regions(fp_pair_scores, region_mask)
            if need_ar
            else torch.zeros_like(region_mask)
        )
        region_indices = selected_ar_regions.nonzero(as_tuple=False)
        ar_indices = []
        ar_features = []
        ar_scores = []
        for start in range(0, region_indices.shape[0], self.ar_region_chunk_size):
            index_chunk = region_indices[start : start + self.ar_region_chunk_size]
            sample_index, fragment_index, pocket_index = index_chunk.unbind(dim=1)
            atom_chunk = atom_repr[sample_index, fragment_index]
            atom_mask_chunk = valid_atoms[sample_index, fragment_index]
            residue_chunk = residue_repr[sample_index, pocket_index]
            residue_mask_chunk = residue_mask[sample_index, pocket_index]
            pair_mask = atom_mask_chunk[:, :, None] & residue_mask_chunk[:, None, :]
            ar_input = self.pair_input(
                atom_chunk[:, :, None, :].expand(-1, -1, num_residues, -1),
                residue_chunk[:, None, :, :].expand(-1, num_atoms, -1, -1),
            )
            flat_input = ar_input.flatten(0, 2)
            flat_features = self.ar_feature_mlp(flat_input).view(
                len(index_chunk), num_atoms * num_residues, -1
            )
            flat_scores = self.ar_score_mlp(flat_input).view(
                len(index_chunk), num_atoms * num_residues
            )
            pair_weights = _masked_softmax(
                flat_scores, pair_mask.flatten(1), dim=1
            )
            region_feature = (pair_weights.unsqueeze(-1) * flat_features).sum(dim=1)
            region_score = self.ar_pair_score(region_feature).squeeze(-1)
            flat_region_index = (
                (sample_index * num_fragments + fragment_index) * num_pockets
                + pocket_index
            )
            ar_indices.append(flat_region_index)
            ar_features.append(region_feature)
            ar_scores.append(region_score)
            del ar_input, flat_input, flat_features, flat_scores

        flat_region_count = batch_size * num_fragments * num_pockets
        ar_feature_dtype = (
            ar_features[0].dtype if ar_features else fp_pair_features_raw.dtype
        )
        ar_score_dtype = ar_scores[0].dtype if ar_scores else fp_pair_scores.dtype
        ar_stack_flat = torch.zeros(
            (flat_region_count, self.fragment_projection.out_features),
            device=fp_pair_features_raw.device,
            dtype=ar_feature_dtype,
        )
        ar_pair_scores_flat = torch.full(
            (flat_region_count,),
            torch.finfo(ar_score_dtype).min,
            device=fp_pair_scores.device,
            dtype=ar_score_dtype,
        )
        if ar_indices:
            all_indices = torch.cat(ar_indices)
            ar_stack_flat = ar_stack_flat.index_copy(
                0, all_indices, torch.cat(ar_features)
            )
            ar_pair_scores_flat = ar_pair_scores_flat.index_copy(
                0, all_indices, torch.cat(ar_scores)
            )
        ar_pair_repr = ar_stack_flat.view(
            batch_size, num_fragments, num_pockets, -1
        )
        ar_pair_scores = ar_pair_scores_flat.view(
            batch_size, num_fragments, num_pockets
        )
        ar_weights = _masked_softmax(
            ar_pair_scores.flatten(1), selected_ar_regions.flatten(1), dim=1
        )
        ar_repr = (
            ar_weights.unsqueeze(-1) * ar_pair_repr.flatten(1, 2)
        ).sum(dim=1)
        ar_valid_mask = selected_ar_regions.flatten(1).any(dim=1)

        fp_pair_features_joint = fp_pair_features_raw
        fp_repr_joint = fp_repr_raw
        if self.cross_scale_mode == "bottom_up":
            updated_fp = self.fp_norm(
                fp_pair_features_raw
                + torch.sigmoid(self.ar_to_fp_gate) * self.ar_to_fp(ar_pair_repr)
            )
            fp_pair_features_joint = torch.where(
                selected_ar_regions.unsqueeze(-1), updated_fp, fp_pair_features_raw
            )
            joint_scores = (
                self.joint_fp_score(fp_pair_features_joint).squeeze(-1)
                if self.joint_fp_score is not None
                else fp_pair_scores
            )
            joint_weights = _masked_softmax(
                joint_scores.flatten(1), region_mask.flatten(1), dim=1
            )
            fp_repr_joint = (
                joint_weights.unsqueeze(-1)
                * fp_pair_features_joint.flatten(1, 2)
            ).sum(dim=1)

        fragment_count = fragment_mask.sum(dim=1)
        pocket_count = pocket_mask.sum(dim=1)
        stats = {
            "fragments_per_sample": fragment_count.to(atom_nodes.dtype).mean(),
            "pockets_per_sample": pocket_count.to(atom_nodes.dtype).mean(),
            "fp_pairs_per_sample": region_mask.sum(dim=(1, 2))
            .to(atom_nodes.dtype)
            .mean(),
            "ar_regions_per_sample": selected_ar_regions.sum(dim=(1, 2))
            .to(atom_nodes.dtype)
            .mean(),
            "local_missing_rate": (~(fp_valid_mask | ar_valid_mask))
            .to(atom_nodes.dtype)
            .mean(),
            "sample_count": torch.tensor(batch_size, device=atom_nodes.device),
            "local_missing_count": (~(fp_valid_mask | ar_valid_mask)).sum(),
            "fp_pairs_total": region_mask.sum(),
            "ar_regions_total": selected_ar_regions.sum(),
        }
        return {
            "fp_pair_features_raw": fp_pair_features_raw,
            "fp_pair_scores": fp_pair_scores,
            "fp_repr_raw": fp_repr_raw,
            "fp_valid_mask": fp_valid_mask,
            "ar_pair_repr": ar_pair_repr,
            "ar_repr": ar_repr,
            "ar_valid_mask": ar_valid_mask,
            "fp_pair_features_joint": fp_pair_features_joint,
            "fp_repr_joint": fp_repr_joint,
            "stats": stats,
        }

    def forward(self, batch):
        need_local = bool({"fp", "ar"} & self.active_scales)
        global_pred, global_repr, drug_nodes, protein_nodes = (
            self._encode_global_and_nodes(batch, need_nodes=need_local)
        )
        global_pred = global_pred.reshape(-1)
        batch_size = global_pred.shape[0]
        local_dim = self.fragment_projection.out_features
        zero_pred = global_pred.new_zeros(batch_size)
        zero_repr = global_pred.new_zeros((batch_size, local_dim))
        false_mask = torch.zeros(batch_size, dtype=torch.bool, device=global_pred.device)
        zero_stat = global_pred.new_zeros(())
        stats = {
            "fragments_per_sample": zero_stat,
            "pockets_per_sample": zero_stat,
            "fp_pairs_per_sample": zero_stat,
            "ar_regions_per_sample": zero_stat,
            "local_missing_rate": zero_stat,
            "sample_count": torch.tensor(batch_size, device=global_pred.device),
            "local_missing_count": torch.zeros(
                (), dtype=torch.long, device=global_pred.device
            ),
            "fp_pairs_total": torch.zeros(
                (), dtype=torch.long, device=global_pred.device
            ),
            "ar_regions_total": torch.zeros(
                (), dtype=torch.long, device=global_pred.device
            ),
        }

        # These detached copies are the only Global-derived tensors visible to
        # Local, Joint and the final fusion losses.
        global_repr_local = global_repr.detach()
        fp_repr_raw = zero_repr
        ar_repr = zero_repr
        fp_repr_for_joint = zero_repr
        global_repr_for_joint = global_repr_local
        fp_valid_mask = false_mask
        ar_valid_mask = false_mask
        if need_local:
            local = self._local_batched_protected(
                drug_nodes.detach(), protein_nodes.detach(), batch
            )
            fp_repr_raw = local["fp_repr_raw"]
            ar_repr = local["ar_repr"]
            fp_valid_mask = local["fp_valid_mask"]
            ar_valid_mask = local["ar_valid_mask"]
            stats = local["stats"]
            fp_repr_for_joint = local["fp_repr_joint"]
            if self.cross_scale_mode == "bottom_up":
                global_repr_for_joint = self.global_norm(
                    global_repr_local
                    + torch.sigmoid(self.fp_to_global_gate)
                    * self.fp_to_global(fp_repr_for_joint)
                )

        fp_active = "fp" in self.active_scales
        ar_active = "ar" in self.active_scales
        joint_active = fp_active and ar_active
        fp_branch_mask = fp_valid_mask & fp_active
        ar_branch_mask = ar_valid_mask & ar_active
        joint_valid_mask = fp_valid_mask & ar_valid_mask & joint_active

        fp_pred = (
            self.fp_head(torch.cat([global_repr_local, fp_repr_raw], dim=-1))
            .reshape(-1)
            if fp_active
            else zero_pred
        )
        ar_pred = (
            self.ar_head(torch.cat([global_repr_local, ar_repr], dim=-1)).reshape(-1)
            if ar_active
            else zero_pred
        )
        joint_pred = (
            self.joint_head(
                torch.cat(
                    [global_repr_for_joint, fp_repr_for_joint, ar_repr], dim=-1
                )
            ).reshape(-1)
            if joint_active
            else zero_pred
        )
        fp_pred = torch.where(fp_branch_mask, fp_pred, zero_pred)
        ar_pred = torch.where(ar_branch_mask, ar_pred, zero_pred)
        joint_pred = torch.where(joint_valid_mask, joint_pred, zero_pred)

        branch_valid_mask = torch.stack(
            [
                torch.ones_like(false_mask),
                fp_branch_mask,
                ar_branch_mask,
                joint_valid_mask,
            ],
            dim=-1,
        )
        masked_logits = self.fusion_logits[None, :].expand(batch_size, -1)
        masked_logits = masked_logits.masked_fill(
            ~branch_valid_mask, torch.finfo(masked_logits.dtype).min
        )
        fusion_weights = torch.softmax(masked_logits, dim=-1)
        branch_predictions = torch.stack(
            [global_pred.detach(), fp_pred, ar_pred, joint_pred], dim=-1
        )
        final_pred = (fusion_weights * branch_predictions.float()).sum(dim=-1)

        self.last_stats = stats
        return {
            "pred": final_pred,
            "global_pred": global_pred,
            "fp_pred": fp_pred,
            "ar_pred": ar_pred,
            "joint_pred": joint_pred,
            "fusion_weights": fusion_weights,
            "fp_valid_mask": fp_branch_mask,
            "ar_valid_mask": ar_branch_mask,
            "joint_valid_mask": joint_valid_mask,
            "fp_repr": fp_repr_raw,
            "ar_repr": ar_repr,
            "joint_fp_repr": fp_repr_for_joint,
            "joint_global_repr": global_repr_for_joint,
            "ar_to_fp_gate": torch.sigmoid(self.ar_to_fp_gate),
            "fp_to_global_gate": torch.sigmoid(self.fp_to_global_gate),
            "branch_valid_mask": branch_valid_mask,
            "active_scales": sorted(self.active_scales),
            "stats": stats,
        }


__all__ = [
    "BRANCH_NAMES",
    "MyModelMDTAP13DProtectedTriScale",
    "parse_active_scales",
]
