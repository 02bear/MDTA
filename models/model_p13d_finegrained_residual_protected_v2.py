# -*- coding: utf-8 -*-
"""Protected Residual E2 v2.

The global branch is trained by its own baseline loss.  The local branch
consumes detached copies of shared representations, so a residual loss cannot
change the global encoders or decoder.  The residual head has no direct access
to the full drug/protein global representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.decoder import Decoder
from models.drug_1d_encoder import Drug1DEncoder
from models.drug_3d_egnn_encoder import Drug3DEGNNEncoder
from models.fusion import ConcatFusion
from models.model_p13d_finegrained_residual import (
    AtomResidueInteraction,
    DrugConditionedPocketSelector,
)
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder


GLOBAL_MODULE_NAMES = (
    "drug_1d_encoder",
    "drug_3d_encoder",
    "drug_fusion",
    "protein_1d_encoder",
    "protein_3d_encoder",
    "protein_fusion",
    "decoder",
)

LOCAL_MODULE_NAMES = (
    "pocket_selector",
    "atom_residue_interaction",
    "local_delta_head",
)


class MyModelMDTAP13DFineGrainedProtectedV2(nn.Module):
    """Baseline plus a bounded, gradient-isolated atom-residue residual."""

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
        delta_limit: float = 1.0,
        include_base_pred_in_delta_head: bool = False,
    ):
        super().__init__()
        if delta_limit <= 0:
            raise ValueError(f"delta_limit must be positive, got {delta_limit}")

        self.hidden_dim = int(hidden_dim)
        self.pocket_top_k = int(pocket_top_k)
        self.interaction_heads = int(interaction_heads)
        self.delta_limit = float(delta_limit)
        self.include_base_pred_in_delta_head = bool(
            include_base_pred_in_delta_head
        )

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
        self.drug_fusion = ConcatFusion(
            input_dims=[hidden_dim, hidden_dim],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
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

        self.pocket_selector = DrugConditionedPocketSelector(
            hidden_dim=hidden_dim,
            top_k=pocket_top_k,
            dropout=dropout,
        )
        self.atom_residue_interaction = AtomResidueInteraction(
            hidden_dim=hidden_dim,
            num_heads=interaction_heads,
            dropout=dropout,
        )

        delta_input_dim = hidden_dim + (
            1 if self.include_base_pred_in_delta_head else 0
        )
        self.delta_input_dim = delta_input_dim
        self.local_delta_head = nn.Sequential(
            nn.Linear(delta_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.local_delta_head[-1].weight)
        nn.init.zeros_(self.local_delta_head[-1].bias)

    @staticmethod
    def _empty_local_debug(base_pred: torch.Tensor, hidden_dim: int):
        batch_size = base_pred.shape[0]
        device = base_pred.device
        dtype = base_pred.dtype
        return {
            "raw_delta": torch.zeros_like(base_pred),
            "local_delta": torch.zeros_like(base_pred),
            "local_interaction_feat": torch.zeros(
                batch_size, hidden_dim, device=device, dtype=dtype
            ),
            "pocket_scores": torch.empty(
                batch_size, 0, device=device, dtype=dtype
            ),
            "pocket_indices": torch.empty(
                batch_size, 0, device=device, dtype=torch.long
            ),
            "interaction_scores": torch.empty(
                batch_size, 0, 0, device=device, dtype=dtype
            ),
            "atom_mask": torch.empty(
                batch_size, 0, device=device, dtype=torch.bool
            ),
            "local_valid_mask": torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
        }

    def _global_forward(self, batch, return_node: bool):
        drug_1d_feat = self.drug_1d_encoder(batch["drug_1d"])
        protein_1d_feat = self.protein_1d_encoder(batch["protein_1d"])

        if return_node:
            drug_3d_output = self.drug_3d_encoder(
                batch["drug_3d"], return_node=True
            )
            protein_3d_output = self.protein_3d_encoder(
                batch["protein_3d"], return_node=True
            )
            drug_3d_feat = drug_3d_output["graph_feat"]
            protein_3d_feat = protein_3d_output["graph_feat"]
        else:
            drug_3d_output = protein_3d_output = None
            drug_3d_feat = self.drug_3d_encoder(batch["drug_3d"])
            protein_3d_feat = self.protein_3d_encoder(batch["protein_3d"])

        drug_feat = self.drug_fusion([drug_1d_feat, drug_3d_feat])
        protein_feat = self.protein_fusion(
            [protein_1d_feat, protein_3d_feat]
        )
        base_pred = self.decoder(
            torch.cat([drug_feat, protein_feat], dim=-1)
        )
        return (
            base_pred,
            drug_feat,
            protein_feat,
            drug_3d_output,
            protein_3d_output,
        )

    def forward(
        self,
        batch,
        return_debug: bool = False,
        disable_local: bool = False,
        residual_target: torch.Tensor | None = None,
    ):
        (
            base_pred,
            drug_feat,
            protein_feat,
            drug_3d_output,
            protein_3d_output,
        ) = self._global_forward(batch, return_node=not disable_local)

        if disable_local:
            local = self._empty_local_debug(base_pred, self.hidden_dim)
        else:
            # Residual gradients stop at these shared representations.  The
            # local modules below remain fully differentiable.
            drug_feat_local = drug_feat.detach()
            protein_feat_local = protein_feat.detach()
            del protein_feat_local  # Kept intentionally as a protected copy.
            drug_atom_tokens_local = drug_3d_output["node_feat"].detach()
            protein_residue_tokens_local = protein_3d_output[
                "node_feat"
            ].detach()
            drug_atom_batch = drug_3d_output["batch"]
            protein_residue_batch = protein_3d_output["batch"]

            (
                pocket_tokens,
                pocket_mask,
                pocket_scores,
                pocket_indices,
            ) = self.pocket_selector(
                protein_tokens=protein_residue_tokens_local,
                protein_batch=protein_residue_batch,
                drug_context=drug_feat_local,
            )

            batch_size = base_pred.shape[0]
            atom_counts = torch.bincount(
                drug_atom_batch, minlength=batch_size
            )
            local_valid_mask = (atom_counts > 0) & pocket_mask.any(dim=1)
            explicit_mask = batch.get("local_valid_mask")
            if explicit_mask is not None:
                explicit_mask = torch.as_tensor(
                    explicit_mask,
                    device=base_pred.device,
                    dtype=torch.bool,
                ).reshape(-1)
                if explicit_mask.numel() != batch_size:
                    raise ValueError(
                        "local_valid_mask must have one value per sample"
                    )
                local_valid_mask = local_valid_mask & explicit_mask

            # MultiheadAttention cannot consume a row whose keys are all
            # masked.  Give invalid rows one zero-valued safe key; their delta
            # is forced to exact zero below.
            safe_pocket_tokens = pocket_tokens.clone()
            safe_pocket_mask = pocket_mask.clone()
            invalid = ~local_valid_mask
            if invalid.any():
                safe_pocket_tokens[invalid, 0] = 0
                safe_pocket_mask[invalid, 0] = True

            (
                local_interaction_feat,
                interaction_scores,
                atom_mask,
            ) = self.atom_residue_interaction(
                atom_tokens=drug_atom_tokens_local,
                atom_batch=drug_atom_batch,
                pocket_tokens=safe_pocket_tokens,
                pocket_mask=safe_pocket_mask,
            )

            delta_input = local_interaction_feat
            if self.include_base_pred_in_delta_head:
                delta_input = torch.cat(
                    [delta_input, base_pred.detach().reshape(batch_size, 1)],
                    dim=-1,
                )
            raw_delta = self.local_delta_head(delta_input).reshape_as(
                base_pred
            )
            bounded_delta = self.delta_limit * torch.tanh(
                raw_delta / self.delta_limit
            )
            local_delta = torch.where(
                local_valid_mask.reshape_as(base_pred),
                bounded_delta,
                torch.zeros_like(bounded_delta),
            )
            local = {
                "raw_delta": raw_delta,
                "local_delta": local_delta,
                "local_interaction_feat": local_interaction_feat,
                "pocket_scores": pocket_scores,
                "pocket_indices": pocket_indices,
                "interaction_scores": interaction_scores,
                "atom_mask": atom_mask,
                "local_valid_mask": local_valid_mask,
            }

        final_pred = base_pred + local["local_delta"]
        if not return_debug:
            return final_pred

        if residual_target is None and batch.get("label") is not None:
            residual_target = (
                batch["label"].reshape_as(base_pred).float()
                - base_pred.detach().float()
            )
        return {
            "pred": final_pred,
            "base_pred": base_pred,
            "raw_delta": local["raw_delta"],
            "local_delta": local["local_delta"],
            "residual_target": residual_target,
            "drug_feat": drug_feat,
            "protein_feat": protein_feat,
            "local_interaction_feat": local["local_interaction_feat"],
            "pocket_scores": local["pocket_scores"],
            "pocket_indices": local["pocket_indices"],
            "interaction_scores": local["interaction_scores"],
            "atom_mask": local["atom_mask"],
            "local_valid_mask": local["local_valid_mask"],
        }

