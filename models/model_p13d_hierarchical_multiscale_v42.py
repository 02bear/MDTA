"""V4.2 hierarchical DTA with local-global cross-granularity consistency.

The three scales do not make three independent affinity predictions. Atom-residue
and fragment-pocket evidence provide gated residual corrections to one global
baseline. Training-only projection/prediction heads align pooled AR+FP evidence
with the global drug-protein representation without using in-batch negatives.
"""

import math

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add

from models.decoder import Decoder
from models.drug_1d_encoder import Drug1DEncoder
from models.fusion import ConcatFusion
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder


class BondAwareDrugEGNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim=14, dropout=0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pos, edge_index, edge_attr):
        src, dst = edge_index
        # Molecular coordinates are physical input evidence, not latent states.
        # Keeping them fixed prevents unconstrained EGNN coordinate drift.  Compute
        # geometry in FP32 even when the surrounding network uses BF16 autocast.
        difference = pos[src].float() - pos[dst].float()
        distance2 = difference.square().sum(-1, keepdim=True).clamp_(0.0, 100.0)
        distance2 = distance2.to(dtype=h.dtype)
        message = self.edge_mlp(torch.cat([h[src], h[dst], distance2, edge_attr], dim=-1))
        h = self.norm(h + self.node_mlp(torch.cat([
            h, scatter_add(message, dst, dim=0, dim_size=h.size(0)),
        ], dim=-1)))
        return h, pos


class BondAwareDrugEGNNEncoder(nn.Module):
    def __init__(self, node_dim=52, edge_dim=14, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList([
            BondAwareDrugEGNNLayer(hidden_dim, edge_dim, dropout) for _ in range(3)
        ])
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout))

    def forward(self, data, return_node=False):
        h, pos = self.input_proj(data["x"]), data["pos"]
        for layer in self.layers:
            h, pos = layer(h, pos, data["edge_index"], data["edge_attr"])
        node_feat = self.out_proj(h)
        graph_feat = global_mean_pool(node_feat, data["batch"])
        if return_node:
            return {"node_feat": node_feat, "graph_feat": graph_feat, "batch": data["batch"]}
        return graph_feat


class AtomResidueRegionBuilder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residue_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_proj = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU())
        self.region_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_tokens, residue_tokens, batch):
        fragment_indices = batch["fragment_atom_indices"]
        fragment_atom_mask = batch["fragment_atom_mask"]
        fragment_mask = batch["fragment_mask"]
        pocket_indices = batch["pocket_residue_indices"]
        pocket_residue_mask = batch["pocket_residue_mask"]
        pocket_mask = batch["pocket_mask"]
        pocket_context = batch["pocket_context6"]
        batch_size, num_fragments = fragment_mask.shape
        num_pockets = pocket_mask.size(1)
        # Gather all padded fragment atoms and pocket residues once.  The old
        # implementation launched hundreds of tiny GPU kernels from three
        # nested Python loops for every batch.
        safe_atom_indices = fragment_indices.clamp_min(0)
        safe_residue_indices = pocket_indices.clamp_min(0)
        atoms = atom_tokens[safe_atom_indices]                       # B,F,A,H
        residues = residue_tokens[safe_residue_indices]             # B,P,R,H
        atom_valid = fragment_atom_mask
        residue_valid = pocket_residue_mask
        pair_valid = (
            atom_valid[:, :, None, :, None]
            & residue_valid[:, None, :, None, :]
        )                                                            # B,F,P,A,R
        region_valid = fragment_mask[:, :, None] & pocket_mask[:, None, :]

        atom_q = self.atom_q(atoms)
        residue_k = self.residue_k(residues)
        scores = torch.einsum("bfah,bprh->bfpar", atom_q, residue_k)
        scores = scores / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~pair_valid, -1e4)

        flat_scores = scores.flatten(-2)
        weights = torch.softmax(flat_scores, dim=-1).view_as(scores)
        weights = weights * pair_valid.to(weights.dtype)
        weights = weights / weights.sum((-2, -1), keepdim=True).clamp_min(1e-8)

        atom_weight = weights.sum(-1)                                # B,F,P,A
        residue_weight = weights.sum(-2)                             # B,F,P,R
        atom_summary = torch.einsum("bfpa,bfah->bfph", atom_weight, atoms)
        residue_summary = torch.einsum("bfpr,bprh->bfph", residue_weight, residues)
        pair_product = torch.einsum(
            "bfpar,bfah,bprh->bfph", weights, atoms, residues,
        )
        atom_square = torch.einsum(
            "bfpa,bfah->bfph", atom_weight, atoms.square(),
        )
        residue_square = torch.einsum(
            "bfpr,bprh->bfph", residue_weight, residues.square(),
        )
        # Smooth squared feature difference retains discrepancy information without
        # materialising a B*F*P*A*R*H tensor.
        pair_difference_squared = (
            atom_square + residue_square - 2.0 * pair_product
        ).clamp_min(0.0)
        # log1p has a bounded derivative at zero, unlike sqrt, which caused
        # extreme pre-clipping gradient norms in v3.
        pair_difference = torch.log1p(pair_difference_squared)

        context_weight = residue_valid.to(pocket_context.dtype)
        context = (
            pocket_context * context_weight[..., None]
        ).sum(2) / context_weight.sum(2, keepdim=True).clamp_min(1.0)
        context = self.context_proj(context)[:, None].expand(
            -1, num_fragments, -1, -1,
        )
        token_input = torch.cat([
            atom_summary, residue_summary, pair_product, pair_difference, context,
        ], dim=-1)
        region_tokens = self.norm(self.region_mlp(token_input))
        region_tokens = region_tokens * region_valid[..., None].to(region_tokens.dtype)
        region_strength = scores.flatten(-2).amax(-1)
        region_strength = region_strength.masked_fill(~region_valid, 0.0)
        return (
            region_tokens.flatten(1, 2),
            region_valid.flatten(1, 2),
            region_strength.flatten(1, 2),
        )


class RegionGraphPropagation(nn.Module):
    def __init__(self, hidden_dim, heads=4, rounds=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=rounds)
        self.pool_score = nn.Linear(hidden_dim, 1)

    def forward(self, tokens, mask):
        propagated = self.encoder(tokens, src_key_padding_mask=~mask)
        score = self.pool_score(propagated).squeeze(-1).masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=-1)
        pooled = (weight[:, :, None] * propagated).sum(1)
        return propagated, pooled, weight


class HierarchicalMultiScaleDTA(nn.Module):
    def __init__(
        self, drug_1d_in_dim=768, atom_v2_dim=52, protein_1d_in_dim=1280,
        protein_node_s_dim=6, hidden_dim=128, dropout=0.1,
        interaction_heads=4, region_rounds=2, task="regression",
        delta_max=1.0, ar_gate_epsilon=0.05, fp_gate_max=0.4,
        consistency_projection_dim=32,
    ):
        super().__init__()
        self.delta_max = float(delta_max)
        self.ar_gate_epsilon = float(ar_gate_epsilon)
        self.fp_gate_max = float(fp_gate_max)
        if self.delta_max <= 0:
            raise ValueError("delta_max must be positive")
        if not 0 <= self.ar_gate_epsilon < 0.5:
            raise ValueError("ar_gate_epsilon must be in [0, 0.5)")
        if not 0 < self.fp_gate_max <= 1.0:
            raise ValueError("fp_gate_max must be in (0, 1]")
        self.drug_1d_encoder = Drug1DEncoder(drug_1d_in_dim, hidden_dim)
        self.drug_atom_encoder = BondAwareDrugEGNNEncoder(atom_v2_dim, 14, hidden_dim, dropout)
        self.drug_fusion = ConcatFusion([hidden_dim, hidden_dim], hidden_dim, hidden_dim * 2, dropout)
        self.protein_1d_encoder = Protein1DEncoder(protein_1d_in_dim, hidden_dim)
        self.protein_3d_encoder = Protein3DEGNNEncoder(
            protein_node_s_dim, hidden_dim, hidden_dim, dropout=dropout, n_layers=3,
        )
        self.residue_type_embedding = nn.Embedding(32, 32)
        self.residue_aux_proj = nn.Sequential(
            nn.Linear(32 + 3 + 1280 + 1 + 5, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.residue_norm = nn.LayerNorm(hidden_dim)
        self.protein_fusion = ConcatFusion([hidden_dim, hidden_dim], hidden_dim, hidden_dim * 2, dropout)
        self.region_builder = AtomResidueRegionBuilder(hidden_dim, dropout)
        self.region_propagation = RegionGraphPropagation(
            hidden_dim, heads=interaction_heads, rounds=region_rounds, dropout=dropout,
        )
        self.ar_pool_score = nn.Linear(hidden_dim, 1)
        self.global_decoder = Decoder(
            input_dim=hidden_dim * 2, hidden_dim=hidden_dim,
            dropout=dropout, task=task,
        )
        self.ar_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.fp_gate = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 1, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.ar_delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.fp_delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        projection_dim = int(consistency_projection_dim)
        if projection_dim <= 0:
            raise ValueError("consistency_projection_dim must be positive")
        # These heads are used only when return_details=True during training and
        # validation. They do not alter the affinity prediction path.
        self.local_consistency_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, projection_dim, bias=False),
        )
        self.global_consistency_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, projection_dim, bias=False),
        )
        # SimSiam-style predictors make stop-gradient alignment less prone to a
        # trivial constant solution while keeping the target encoders trainable
        # through the opposite direction of the symmetric loss.
        self.local_consistency_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim),
            nn.SiLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.global_consistency_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim),
            nn.SiLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        for head in (self.ar_delta_head, self.fp_delta_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, batch, return_details=False):
        drug_1d = self.drug_1d_encoder(batch["drug_1d"])
        drug_out = self.drug_atom_encoder(batch["drug_atom_v2"], return_node=True)
        drug_feat = self.drug_fusion([drug_1d, drug_out["graph_feat"]])

        protein_1d = self.protein_1d_encoder(batch["protein_1d"])
        protein_out = self.protein_3d_encoder(batch["protein_3d"], return_node=True)
        residue = batch["protein_residue_v2"]
        residue_aux = torch.cat([
            self.residue_type_embedding(residue["residue_type_index"].clamp(0, 31)),
            residue["sequence_scalar3"], residue["esm_per_tok"],
            residue["esm_mask"].float().unsqueeze(-1), residue["structure_extra5"],
        ], dim=-1)
        residue_tokens = self.residue_norm(protein_out["node_feat"] + self.residue_aux_proj(residue_aux))
        protein_local_global = global_mean_pool(residue_tokens, protein_out["batch"])
        protein_feat = self.protein_fusion([protein_1d, protein_out["graph_feat"] + protein_local_global])

        regions, region_mask, region_strength = self.region_builder(
            drug_out["node_feat"], residue_tokens, batch,
        )
        ar_score = self.ar_pool_score(regions).squeeze(-1).masked_fill(~region_mask, -1e9)
        ar_weight = torch.softmax(ar_score, dim=-1)
        ar_feat = (ar_weight[:, :, None] * regions).sum(1)
        propagated, fp_feat, fp_weight = self.region_propagation(regions, region_mask)

        base_pred = self.global_decoder(torch.cat([drug_feat, protein_feat], dim=-1))
        ar_delta_raw = self.ar_delta_head(
            torch.cat([drug_feat, protein_feat, ar_feat], dim=-1)
        ).view_as(base_pred)
        fp_increment_features = torch.cat([
            drug_feat, protein_feat, ar_feat, fp_feat, fp_feat - ar_feat,
        ], dim=-1)
        fp_delta_raw = self.fp_delta_head(fp_increment_features).view_as(base_pred)
        ar_delta = self.delta_max * torch.tanh(ar_delta_raw)
        fp_delta = self.delta_max * torch.tanh(fp_delta_raw)
        ar_gate_logit = self.ar_gate(
            torch.cat([drug_feat, protein_feat, ar_feat], dim=-1)
        )
        ar_gate = self.ar_gate_epsilon + (
            1.0 - 2.0 * self.ar_gate_epsilon
        ) * torch.sigmoid(ar_gate_logit)
        gated_ar = ar_gate.view_as(base_pred) * ar_delta
        pred_after_ar = base_pred + gated_ar
        fp_gate_logit = self.fp_gate(torch.cat([
            fp_increment_features, gated_ar.detach(),
        ], dim=-1))
        gate_logits = torch.cat([ar_gate_logit, fp_gate_logit], dim=-1)
        fp_gate = self.fp_gate_max * torch.sigmoid(fp_gate_logit)
        gates = torch.cat([ar_gate, fp_gate], dim=-1)
        gated_fp = fp_gate.view_as(base_pred) * fp_delta
        pred = pred_after_ar + gated_fp
        ar_candidate = base_pred + ar_delta
        fp_candidate = pred_after_ar + fp_delta

        if not return_details:
            return pred
        local_projection = self.local_consistency_projector(
            torch.cat([ar_feat, fp_feat], dim=-1)
        )
        global_projection = self.global_consistency_projector(
            torch.cat([drug_feat, protein_feat], dim=-1)
        )
        return {
            "pred": pred,
            "base_pred": base_pred,
            "ar_candidate": ar_candidate,
            "pred_after_ar": pred_after_ar,
            "fp_candidate": fp_candidate,
            "ar_delta": ar_delta,
            "fp_delta": fp_delta,
            "ar_delta_raw": ar_delta_raw,
            "fp_delta_raw": fp_delta_raw,
            "scale_gates": gates,
            "scale_gate_logits": gate_logits,
            "region_mask": region_mask,
            "region_strength": region_strength,
            "ar_region_weight": ar_weight,
            "fp_region_weight": fp_weight,
            "region_tokens": propagated,
            "local_projection": local_projection,
            "global_projection": global_projection,
            "local_prediction": self.local_consistency_predictor(local_projection),
            "global_prediction": self.global_consistency_predictor(global_projection),
        }
