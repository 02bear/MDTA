# -*- coding: utf-8 -*-
"""
Clean Global + Strict AR2 (all-residue version)

This model deliberately does NOT depend on E2, the old E2 selector, or the old
E2 atom-residue branch.

Frozen independent Global baseline:
    drug_1d + drug_3d -> drug fusion
    protein_1d + protein_3d -> protein fusion
    concat -> decoder -> y_global

Strict AR2 residual:
    frozen Drug3D node tokens (heavy atoms only)
        x
    frozen Protein3D residue node tokens (ALL residues)
        -> strict multiplicative pair interaction
        -> double-center pair score
        -> softmax - uniform null
        -> double-center pair embedding
        -> z_AR2
        -> bias-free residual head -> delta_AR2

Final:
    y = y_global + delta_AR2
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.model_p13d import MyModelMDTAP13D


def _double_center_scalar(x: torch.Tensor) -> torch.Tensor:
    return (
        x
        - x.mean(dim=1, keepdim=True)
        - x.mean(dim=0, keepdim=True)
        + x.mean()
    )


def _double_center_vector(x: torch.Tensor) -> torch.Tensor:
    return (
        x
        - x.mean(dim=1, keepdim=True)
        - x.mean(dim=0, keepdim=True)
        + x.mean(dim=(0, 1), keepdim=True)
    )


class StrictCenteredAllResidueAR2(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        interaction_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.interaction_dim = int(interaction_dim)

        self.atom_proj = nn.Sequential(
            nn.Linear(hidden_dim, interaction_dim, bias=False),
            nn.LayerNorm(interaction_dim, elementwise_affine=False),
        )
        self.residue_proj = nn.Sequential(
            nn.Linear(hidden_dim, interaction_dim, bias=False),
            nn.LayerNorm(interaction_dim, elementwise_affine=False),
        )

        self.pair_score_mlp = nn.Sequential(
            nn.Linear(interaction_dim, interaction_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, 1, bias=False),
        )

        self.pair_edge_mlp = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

    def _one_pair(
        self,
        atom_h: torch.Tensor,
        residue_h: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if atom_h.size(0) == 0:
            raise ValueError("AR2 received a sample with zero heavy atoms.")
        if residue_h.size(0) == 0:
            raise ValueError("AR2 received a sample with zero protein residues.")

        q = self.atom_proj(atom_h)
        k = self.residue_proj(residue_h)

        product = q[:, None, :] * k[None, :, :]

        raw_score = (
            product.sum(dim=-1) / math.sqrt(self.interaction_dim)
            + self.pair_score_mlp(product).squeeze(-1)
        )

        interaction_score = _double_center_scalar(raw_score)

        alpha = torch.softmax(
            interaction_score.reshape(-1),
            dim=0,
        ).reshape_as(interaction_score)

        n_pair = interaction_score.numel()
        uniform = interaction_score.new_full(
            interaction_score.shape,
            1.0 / float(n_pair),
        )
        centered_weight = alpha - uniform

        raw_edge = self.pair_edge_mlp(product)
        interaction_edge = _double_center_vector(raw_edge)

        z = (
            centered_weight.unsqueeze(-1)
            * interaction_edge
        ).sum(dim=(0, 1))

        entropy = -(
            alpha.clamp_min(1e-12)
            * alpha.clamp_min(1e-12).log()
        ).sum()
        normalized_entropy = (
            entropy / math.log(float(n_pair))
            if n_pair > 1
            else entropy.new_zeros(())
        )

        tv_from_uniform = 0.5 * centered_weight.abs().sum()
        raw_std = raw_score.std(unbiased=False)
        interaction_std = interaction_score.std(unbiased=False)

        center_err = torch.maximum(
            interaction_score.mean(dim=1).abs().max(),
            interaction_score.mean(dim=0).abs().max(),
        )

        residue_importance = centered_weight.abs().sum(dim=0)
        atom_importance = centered_weight.abs().sum(dim=1)

        return {
            "z": z,
            "raw_score_std": raw_std,
            "interaction_score_std": interaction_std,
            "normalized_entropy": normalized_entropy,
            "tv_from_uniform": tv_from_uniform,
            "z_norm": z.norm(),
            "center_err": center_err,
            "num_atoms": raw_score.new_tensor(float(atom_h.size(0))),
            "num_residues": raw_score.new_tensor(float(residue_h.size(0))),
            "num_pairs": raw_score.new_tensor(float(n_pair)),
            "residue_importance": residue_importance,
            "atom_importance": atom_importance,
        }

    def forward(
        self,
        atom_tokens: torch.Tensor,
        atom_batch: torch.Tensor,
        heavy_atom_indices: torch.Tensor,
        heavy_atom_mask: torch.Tensor,
        residue_tokens: torch.Tensor,
        residue_batch: torch.Tensor,
    ) -> Dict[str, object]:
        heavy_atom_indices = heavy_atom_indices.long()
        heavy_atom_mask = heavy_atom_mask.bool() & (heavy_atom_indices >= 0)

        batch_size = heavy_atom_indices.size(0)
        z_list: List[torch.Tensor] = []
        scalar_lists = {
            "raw_score_std": [],
            "interaction_score_std": [],
            "normalized_entropy": [],
            "tv_from_uniform": [],
            "z_norm": [],
            "center_err": [],
            "num_atoms": [],
            "num_residues": [],
            "num_pairs": [],
        }
        residue_importance_list = []
        atom_importance_list = []

        for b in range(batch_size):
            valid_atom_idx = heavy_atom_indices[b][heavy_atom_mask[b]]
            if valid_atom_idx.numel() == 0:
                raise ValueError(f"sample {b}: no heavy atoms")

            owners = atom_batch[valid_atom_idx]
            if not torch.all(owners == b):
                raise ValueError(
                    f"sample {b}: heavy atom index points to another batch item"
                )

            residue_idx = torch.nonzero(
                residue_batch == b,
                as_tuple=False,
            ).view(-1)
            if residue_idx.numel() == 0:
                raise ValueError(f"sample {b}: no protein residues")

            out = self._one_pair(
                atom_tokens[valid_atom_idx],
                residue_tokens[residue_idx],
            )

            z_list.append(out["z"])
            for key in scalar_lists:
                scalar_lists[key].append(out[key])
            residue_importance_list.append(out["residue_importance"])
            atom_importance_list.append(out["atom_importance"])

        result: Dict[str, object] = {
            "z_ar2": torch.stack(z_list, dim=0),
            "residue_importance_list": residue_importance_list,
            "atom_importance_list": atom_importance_list,
        }
        for key, values in scalar_lists.items():
            result[key] = torch.stack(values, dim=0)

        return result


class GlobalStrictAR2StageA(nn.Module):
    def __init__(
        self,
        drug_1d_in_dim: int = 768,
        drug_3d_node_in_dim: int = 10,
        protein_1d_in_dim: int = 1280,
        protein_3d_node_s_dim: int = 6,
        protein_3d_node_v_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        interaction_dim: int = 128,
        task: str = "regression",
        freeze_global: bool = True,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        self.global_model = MyModelMDTAP13D(
            drug_1d_in_dim=drug_1d_in_dim,
            drug_3d_node_in_dim=drug_3d_node_in_dim,
            protein_1d_in_dim=protein_1d_in_dim,
            protein_3d_node_s_dim=protein_3d_node_s_dim,
            protein_3d_node_v_dim=protein_3d_node_v_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
        )

        self.ar2 = StrictCenteredAllResidueAR2(
            hidden_dim=hidden_dim,
            interaction_dim=interaction_dim,
            dropout=dropout,
        )

        self.ar2_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )
        nn.init.zeros_(self.ar2_delta_head[-1].weight)

        self.freeze_global = bool(freeze_global)
        self.set_global_frozen(self.freeze_global)

    def set_global_frozen(self, frozen: bool = True) -> None:
        self.freeze_global = bool(frozen)
        for p in self.global_model.parameters():
            p.requires_grad_(not frozen)
        if frozen:
            self.global_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_global:
            self.global_model.eval()
        return self

    def load_global_checkpoint_state(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        self.global_model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _drug3d_forward_with_nodes(encoder, data: dict):
        h = encoder.input_proj(data["x"])
        x = data["pos"]
        edge_index = data["edge_index"]
        batch = data["batch"]

        for layer in encoder.layers:
            h, x = layer(h, x, edge_index)

        graph_feat = encoder.out_proj(global_mean_pool(h, batch))
        node_feat = encoder.out_proj(h)

        return {
            "node_feat": node_feat,
            "graph_feat": graph_feat,
            "batch": batch,
        }

    def _global_forward_with_nodes(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        gm = self.global_model

        drug_1d_feat = gm.drug_1d_encoder(batch["drug_1d"])
        drug_3d = self._drug3d_forward_with_nodes(
            gm.drug_3d_encoder,
            batch["drug_3d"],
        )
        drug_feat = gm.drug_fusion(
            [drug_1d_feat, drug_3d["graph_feat"]]
        )

        protein_1d_feat = gm.protein_1d_encoder(batch["protein_1d"])
        protein_3d = gm.protein_3d_encoder(
            batch["protein_3d"],
            return_node=True,
        )
        protein_feat = gm.protein_fusion(
            [protein_1d_feat, protein_3d["graph_feat"]]
        )

        pair_feat = torch.cat([drug_feat, protein_feat], dim=-1)
        global_pred = gm.decoder(pair_feat)

        return {
            "global_pred": global_pred,
            "drug_atom_tokens": drug_3d["node_feat"],
            "drug_atom_batch": drug_3d["batch"],
            "protein_residue_tokens": protein_3d["node_feat"],
            "protein_residue_batch": protein_3d["batch"],
        }

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
        if self.freeze_global:
            with torch.no_grad():
                g = self._global_forward_with_nodes(batch)
        else:
            g = self._global_forward_with_nodes(batch)

        ar = self.ar2(
            atom_tokens=g["drug_atom_tokens"],
            atom_batch=g["drug_atom_batch"],
            heavy_atom_indices=batch["heavy_atom_indices"],
            heavy_atom_mask=batch["heavy_atom_mask"],
            residue_tokens=g["protein_residue_tokens"],
            residue_batch=g["protein_residue_batch"],
        )

        delta = self.ar2_delta_head(ar["z_ar2"])
        delta = delta.view_as(g["global_pred"])
        pred = g["global_pred"] + delta

        if not return_details:
            return pred

        return {
            "pred": pred,
            "global_pred": g["global_pred"],
            "ar2_delta": delta,
            "ar2_feat": ar["z_ar2"],
            "ar2_raw_score_std": ar["raw_score_std"],
            "ar2_interaction_score_std": ar["interaction_score_std"],
            "ar2_normalized_entropy": ar["normalized_entropy"],
            "ar2_tv_from_uniform": ar["tv_from_uniform"],
            "ar2_z_norm": ar["z_norm"],
            "ar2_center_err": ar["center_err"],
            "ar2_num_atoms": ar["num_atoms"],
            "ar2_num_residues": ar["num_residues"],
            "ar2_num_pairs": ar["num_pairs"],
            "residue_importance_list": ar["residue_importance_list"],
            "atom_importance_list": ar["atom_importance_list"],
        }
