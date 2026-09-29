# -*- coding: utf-8 -*-
"""
No-FP residual control for the E2-guided three-granularity study.

Purpose
-------
This is a strict control for Stage A.

- Load the same fold-specific trained E2 checkpoint.
- Freeze the entire E2 branch and keep it in eval mode.
- Do NOT read BRICS fragments.
- Do NOT read CAVIAR pockets.
- Do NOT construct any fragment-pocket feature.
- Train only a generic residual head from existing frozen E2 representations:

      delta_control = H([drug_feat, protein_feat, z_AR])
      y_final = y_E2 + delta_control

The final linear layer is zero initialized, so before training:
      y_final == y_E2

This control asks whether Stage-A gains can be obtained merely by adding another
residual MLP on top of the already-learned E2 features, without FP information.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.model_p13d_finegrained_residual import (
    MyModelMDTAP13DFineGrained,
)


class E2NoFPResidualControlDTA(nn.Module):
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

        # Generic residual control: ONLY frozen E2 features.
        self.control_delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.control_delta_head[-1].weight)
        nn.init.zeros_(self.control_delta_head[-1].bias)

        self.freeze_e2 = bool(freeze_e2)
        self.set_e2_frozen(self.freeze_e2)

    def set_e2_frozen(self, frozen: bool = True) -> None:
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
        self.e2.load_state_dict(state_dict, strict=True)

    def forward(self, batch, return_details: bool = False):
        if self.freeze_e2:
            with torch.no_grad():
                e2 = self.e2(batch, return_debug=True)
        else:
            e2 = self.e2(batch, return_debug=True)

        control_input = torch.cat(
            [
                e2["drug_feat"],
                e2["protein_feat"],
                e2["local_interaction_feat"],
            ],
            dim=-1,
        )
        control_delta = self.control_delta_head(control_input)
        control_delta = control_delta.view_as(e2["pred"])

        pred = e2["pred"] + control_delta

        if not return_details:
            return pred

        return {
            "pred": pred,
            "e2_pred": e2["pred"],
            "base_pred": e2["base_pred"],
            "ar_delta": e2["local_delta"],
            "control_delta": control_delta,
            "drug_feat": e2["drug_feat"],
            "protein_feat": e2["protein_feat"],
            "ar_feat": e2["local_interaction_feat"],
        }
