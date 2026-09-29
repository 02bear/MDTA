# -*- coding: utf-8 -*-
"""
Generic I2MoE / IMoE-style interaction-aware fusion for regression.
"""

import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class I2MoEFusionRegressor(nn.Module):
    """
    Interaction-aware mixture-of-experts fusion for any number of modalities.
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

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
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
                raise ValueError(
                    f"Batch size mismatch at zs[{idx}]: {z.size(0)} vs {batch_size}"
                )
            if z.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"Hidden dim mismatch at zs[{idx}]: {z.size(-1)} vs {self.hidden_dim}"
                )

    def _expert_predictions(self, zs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(zs, dim=-1)
        preds = [expert(x).squeeze(-1) for expert in self.experts]
        return torch.stack(preds, dim=-1)

    def _interaction_loss(self, zs: List[torch.Tensor], full_preds: torch.Tensor) -> torch.Tensor:
        losses = []
        masked_preds = []

        for mask_idx in range(self.num_modalities):
            z_masked = list(zs)
            z_masked[mask_idx] = torch.randn_like(zs[mask_idx])
            masked_preds.append(self._expert_predictions(z_masked))

        masked_preds = torch.stack(masked_preds, dim=0)

        for expert_idx in range(self.num_modalities):
            anchor = full_preds[:, expert_idx]
            for mask_idx in range(self.num_modalities):
                other = masked_preds[mask_idx, :, expert_idx]
                if mask_idx == expert_idx:
                    dist = (anchor - other).pow(2)
                    losses.append(F.relu(self.interaction_margin - dist).mean())
                else:
                    losses.append(F.mse_loss(anchor, other))

        syn_anchor = full_preds[:, self.syn_idx]
        red_anchor = full_preds[:, self.red_idx]
        for mask_idx in range(self.num_modalities):
            syn_other = masked_preds[mask_idx, :, self.syn_idx]
            syn_dist = (syn_anchor - syn_other).pow(2)
            losses.append(F.relu(self.interaction_margin - syn_dist).mean())

            red_other = masked_preds[mask_idx, :, self.red_idx]
            losses.append(F.mse_loss(red_anchor, red_other))

        if not losses:
            return torch.zeros((), device=zs[0].device)
        return torch.stack(losses).mean()

    def forward(
        self,
        zs: List[torch.Tensor],
        compute_interaction_loss: bool = True,
    ):
        self._check_inputs(zs)

        gate_input = torch.cat(zs, dim=-1)
        gate_logits = self.gate(gate_input)
        weights = F.softmax(gate_logits / self.gate_temperature, dim=-1)

        expert_preds = self._expert_predictions(zs)
        pred = (weights * expert_preds).sum(dim=-1)

        if self.training and compute_interaction_loss:
            interaction_loss = self._interaction_loss(zs, expert_preds)
        else:
            interaction_loss = torch.zeros((), device=zs[0].device)

        aux = {
            "i2moe_loss": interaction_loss,
            "i2moe_weights": weights.detach(),
            "i2moe_expert_preds": expert_preds.detach(),
            "i2moe_w_synergy": weights[:, self.syn_idx].mean().detach(),
            "i2moe_w_redundancy": weights[:, self.red_idx].mean().detach(),
        }

        for idx, key in enumerate(self._uni_aux_names):
            aux[key] = weights[:, idx].mean().detach()

        return pred, aux
