"""Two-part Davis affinity model for the censored floor at pKd=5.

The model first estimates whether an interaction is active (label > 5) from
global drug/protein evidence.  Conditional affinity above the floor is then
predicted from global evidence, optionally corrected by atom-residue (AR)
evidence.  The final prediction is the conditional expectation

    floor + P(active) * E[label - floor | active].

Unlike the earlier scale gates, P(active) has a directly observed target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from models.drug_1d_encoder import Drug1DEncoder
from models.fusion import ConcatFusion
from models.model_p13d_hierarchical_multiscale_v41 import (
    AtomResidueRegionBuilder,
    BondAwareDrugEGNNEncoder,
)
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder


class HurdleGlobalARDTA(nn.Module):
    """Global activity classifier plus conditional Global(+AR) regressor."""

    def __init__(
        self,
        drug_1d_in_dim=768,
        atom_v2_dim=52,
        protein_1d_in_dim=1280,
        protein_node_s_dim=6,
        hidden_dim=128,
        dropout=0.1,
        affinity_floor=5.0,
        residual_head="global_ar",
        ar_delta_max=1.0,
    ):
        super().__init__()
        if residual_head not in {"global", "global_ar"}:
            raise ValueError("residual_head must be 'global' or 'global_ar'")
        if ar_delta_max <= 0:
            raise ValueError("ar_delta_max must be positive")

        self.affinity_floor = float(affinity_floor)
        self.residual_head = residual_head
        self.use_ar = residual_head == "global_ar"
        self.ar_delta_max = float(ar_delta_max)

        self.drug_1d_encoder = Drug1DEncoder(drug_1d_in_dim, hidden_dim)
        self.drug_atom_encoder = BondAwareDrugEGNNEncoder(
            atom_v2_dim, 14, hidden_dim, dropout,
        )
        self.drug_fusion = ConcatFusion(
            [hidden_dim, hidden_dim],
            hidden_dim,
            hidden_dim * 2,
            dropout,
        )

        self.protein_1d_encoder = Protein1DEncoder(
            protein_1d_in_dim, hidden_dim,
        )
        self.protein_3d_encoder = Protein3DEGNNEncoder(
            protein_node_s_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            n_layers=3,
        )
        self.residue_type_embedding = nn.Embedding(32, 32)
        self.residue_aux_proj = nn.Sequential(
            nn.Linear(32 + 3 + 1280 + 1 + 5, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residue_norm = nn.LayerNorm(hidden_dim)
        self.protein_fusion = ConcatFusion(
            [hidden_dim, hidden_dim],
            hidden_dim,
            hidden_dim * 2,
            dropout,
        )

        global_dim = hidden_dim * 2
        self.active_head = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.global_residual_head = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        if self.use_ar:
            self.region_builder = AtomResidueRegionBuilder(
                hidden_dim, dropout,
            )
            self.ar_pool_score = nn.Linear(hidden_dim, 1)
            self.ar_residual_head = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            # Start exactly at the Global hurdle model.  AR must earn a
            # non-zero correction from conditional active-pair supervision.
            nn.init.zeros_(self.ar_residual_head[-1].weight)
            nn.init.zeros_(self.ar_residual_head[-1].bias)

    def encode(self, batch):
        drug_1d = self.drug_1d_encoder(batch["drug_1d"])
        drug_out = self.drug_atom_encoder(
            batch["drug_atom_v2"], return_node=True,
        )
        drug_feat = self.drug_fusion(
            [drug_1d, drug_out["graph_feat"]],
        )

        protein_1d = self.protein_1d_encoder(batch["protein_1d"])
        protein_out = self.protein_3d_encoder(
            batch["protein_3d"], return_node=True,
        )
        residue = batch["protein_residue_v2"]
        residue_aux = torch.cat(
            [
                self.residue_type_embedding(
                    residue["residue_type_index"].clamp(0, 31),
                ),
                residue["sequence_scalar3"],
                residue["esm_per_tok"],
                residue["esm_mask"].float().unsqueeze(-1),
                residue["structure_extra5"],
            ],
            dim=-1,
        )
        residue_tokens = self.residue_norm(
            protein_out["node_feat"] + self.residue_aux_proj(residue_aux),
        )
        protein_local_global = global_mean_pool(
            residue_tokens, protein_out["batch"],
        )
        protein_feat = self.protein_fusion(
            [
                protein_1d,
                protein_out["graph_feat"] + protein_local_global,
            ],
        )
        return (
            drug_feat,
            protein_feat,
            drug_out["node_feat"],
            residue_tokens,
        )

    def forward(self, batch, return_details=False):
        (
            drug_feat,
            protein_feat,
            atom_tokens,
            residue_tokens,
        ) = self.encode(batch)
        global_feat = torch.cat([drug_feat, protein_feat], dim=-1)

        active_logit = self.active_head(global_feat)
        active_probability = torch.sigmoid(active_logit)
        global_residual_raw = self.global_residual_head(global_feat)
        global_conditional_residual = F.softplus(global_residual_raw)

        batch_size = global_feat.size(0)
        ar_residual_raw = global_residual_raw.new_zeros((batch_size, 1))
        ar_residual_correction = global_residual_raw.new_zeros(
            (batch_size, 1),
        )
        ar_feat = global_residual_raw.new_zeros(
            (batch_size, drug_feat.size(-1)),
        )
        region_mask = torch.zeros(
            (batch_size, 1),
            dtype=torch.bool,
            device=global_feat.device,
        )
        ar_region_weight = global_residual_raw.new_zeros((batch_size, 1))

        if self.use_ar:
            regions, region_mask, _ = self.region_builder(
                atom_tokens, residue_tokens, batch,
            )
            ar_score = self.ar_pool_score(regions).squeeze(-1)
            ar_score = ar_score.masked_fill(~region_mask, -1e9)
            ar_region_weight = torch.softmax(ar_score, dim=-1)
            ar_feat = (
                ar_region_weight[:, :, None] * regions
            ).sum(dim=1)
            ar_residual_raw = self.ar_residual_head(
                torch.cat([drug_feat, protein_feat, ar_feat], dim=-1),
            )
            ar_residual_correction = self.ar_delta_max * torch.tanh(
                ar_residual_raw,
            )

        conditional_residual = F.softplus(
            global_residual_raw + ar_residual_correction,
        )
        global_hurdle_pred = (
            self.affinity_floor
            + active_probability * global_conditional_residual
        )
        pred = (
            self.affinity_floor
            + active_probability * conditional_residual
        )

        if not return_details:
            return pred
        return {
            "pred": pred,
            "global_hurdle_pred": global_hurdle_pred,
            "active_logit": active_logit,
            "active_probability": active_probability,
            "global_residual_raw": global_residual_raw,
            "global_conditional_residual": global_conditional_residual,
            "conditional_residual": conditional_residual,
            "ar_residual_raw": ar_residual_raw,
            "ar_residual_correction": ar_residual_correction,
            "ar_feat": ar_feat,
            "region_mask": region_mask,
            "ar_region_weight": ar_region_weight,
        }
