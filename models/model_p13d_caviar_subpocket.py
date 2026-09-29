"""CAVIAR subpocket model adapted from the V4.1 residual hierarchy.

Key differences from V4.1:
  1. shared-weight spatial GCN inside every CAVIAR subpocket;
  2. graph propagation between overlapping/neighboring subpockets;
  3. sparse fragment-subpocket selection before atom-residue interaction.
"""

import math

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.decoder import Decoder
from models.drug_1d_encoder import Drug1DEncoder
from models.fusion import ConcatFusion
from models.model_p13d_hierarchical_multiscale_v41 import (
    BondAwareDrugEGNNEncoder,
)
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder


class SharedSubpocketGCN(nn.Module):
    """One set of weights is reused for every subpocket in the batch."""

    def __init__(self, hidden_dim, rounds=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(rounds)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(rounds)])
        self.pool_score = nn.Linear(hidden_dim, 1)

    def forward(self, residue_tokens, batch):
        indices = batch["pocket_residue_indices"].clamp_min(0)
        mask = batch["pocket_residue_mask"]
        adjacency = batch["pocket_residue_adjacency"].to(residue_tokens.dtype)
        x = residue_tokens[indices] * mask[..., None].to(residue_tokens.dtype)
        valid_pair = mask[..., :, None] & mask[..., None, :]
        adjacency = adjacency * valid_pair.to(adjacency.dtype)
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        for layer, norm in zip(self.layers, self.norms):
            message = torch.matmul(adjacency, x) / degree
            x = norm(x + layer(torch.cat([x, message], dim=-1)))
            x = x * mask[..., None].to(x.dtype)
        score = self.pool_score(x).squeeze(-1).masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=-1) * mask.to(score.dtype)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (weight[..., None] * x).sum(-2)
        pooled = pooled * batch["pocket_mask"][..., None].to(pooled.dtype)
        return x, pooled, weight


class SubpocketGraphPropagation(nn.Module):
    def __init__(self, hidden_dim, rounds=2, edge_dim=3, dropout=0.1):
        super().__init__()
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(rounds)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(rounds)])

    def forward(self, x, batch):
        mask = batch["pocket_mask"]
        adjacency = batch["pocket_adjacency"].to(x.dtype)
        edge_weight = 0.5 + torch.sigmoid(
            self.edge_gate(batch["pocket_edge_features"]).squeeze(-1)
        )
        valid_pair = mask[:, :, None] & mask[:, None, :]
        adjacency = adjacency * edge_weight * valid_pair.to(x.dtype)
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        for layer, norm in zip(self.layers, self.norms):
            message = torch.matmul(adjacency, x) / degree
            x = norm(x + layer(torch.cat([x, message], dim=-1)))
            x = x * mask[..., None].to(x.dtype)
        return x


class SparseAtomResidueRegionBuilder(nn.Module):
    def __init__(self, hidden_dim, pockets_per_fragment=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pockets_per_fragment = int(pockets_per_fragment)
        self.fragment_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pocket_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atom_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residue_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_proj = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU())
        self.soft_selection_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.region_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_tokens, pocket_residue_tokens, pocket_tokens, batch):
        fragment_indices = batch["fragment_atom_indices"].clamp_min(0)
        fragment_atom_mask = batch["fragment_atom_mask"]
        fragment_mask = batch["fragment_mask"]
        pocket_mask = batch["pocket_mask"]
        atoms = atom_tokens[fragment_indices]
        atom_weight = fragment_atom_mask.to(atoms.dtype)
        fragment_tokens = (
            atoms * atom_weight[..., None]
        ).sum(-2) / atom_weight.sum(-1, keepdim=True).clamp_min(1.0)

        compatibility = torch.einsum(
            "bfh,bph->bfp",
            self.fragment_query(fragment_tokens),
            self.pocket_key(pocket_tokens),
        ) / math.sqrt(self.hidden_dim)
        compatibility = compatibility.masked_fill(~pocket_mask[:, None, :], -1e4)
        compatibility = compatibility.masked_fill(~fragment_mask[:, :, None], -1e4)
        soft_selection_weight = torch.softmax(compatibility, dim=-1)
        soft_selection_weight = (
            soft_selection_weight
            * pocket_mask[:, None, :].to(soft_selection_weight.dtype)
            * fragment_mask[:, :, None].to(soft_selection_weight.dtype)
        )
        soft_selection_weight = soft_selection_weight / soft_selection_weight.sum(
            -1, keepdim=True
        ).clamp_min(1e-8)
        soft_pocket_context = torch.einsum(
            "bfp,bph->bfh", soft_selection_weight, pocket_tokens
        )
        select_k = min(self.pockets_per_fragment, pocket_tokens.size(1))
        selected_score, selected = torch.topk(
            compatibility, k=select_k, dim=-1
        )
        batch_index = torch.arange(atoms.size(0), device=atoms.device)[:, None, None]
        selected_residues = pocket_residue_tokens[batch_index, selected]
        selected_residue_mask = batch["pocket_residue_mask"][batch_index, selected]
        selected_context = batch["pocket_context6"][batch_index, selected]
        selected_pocket_mask = pocket_mask[batch_index, selected]
        region_valid = fragment_mask[:, :, None] & selected_pocket_mask

        atom_q = self.atom_q(atoms)
        residue_k = self.residue_k(selected_residues)
        scores = torch.einsum("bfah,bfkrh->bfkar", atom_q, residue_k)
        scores = scores / math.sqrt(self.hidden_dim)
        pair_valid = (
            fragment_atom_mask[:, :, None, :, None]
            & selected_residue_mask[:, :, :, None, :]
        )
        scores = scores.masked_fill(~pair_valid, -1e4)
        weights = torch.softmax(scores.flatten(-2), dim=-1).view_as(scores)
        weights = weights * pair_valid.to(weights.dtype)
        weights = weights / weights.sum((-2, -1), keepdim=True).clamp_min(1e-8)

        atom_attention = weights.sum(-1)
        residue_attention = weights.sum(-2)
        atom_summary = torch.einsum("bfka,bfah->bfkh", atom_attention, atoms)
        residue_summary = torch.einsum(
            "bfkr,bfkrh->bfkh", residue_attention, selected_residues
        )
        product = torch.einsum(
            "bfkar,bfah,bfkrh->bfkh", weights, atoms, selected_residues
        )
        atom_square = torch.einsum(
            "bfka,bfah->bfkh", atom_attention, atoms.square()
        )
        residue_square = torch.einsum(
            "bfkr,bfkrh->bfkh", residue_attention, selected_residues.square()
        )
        difference = torch.log1p(
            (atom_square + residue_square - 2.0 * product).clamp_min(0.0)
        )
        context_weight = selected_residue_mask.to(selected_context.dtype)
        context = (
            selected_context * context_weight[..., None]
        ).sum(-2) / context_weight.sum(-1, keepdim=True).clamp_min(1.0)
        context = self.context_proj(context)
        region = self.region_mlp(
                torch.cat(
                    [atom_summary, residue_summary, product, difference, context],
                    dim=-1,
                )
        )
        region = region + self.soft_selection_context(soft_pocket_context)[
            :, :, None, :
        ]
        selection_confidence = 0.5 + torch.sigmoid(selected_score)[..., None]
        region = self.norm(region) * selection_confidence
        region = region * region_valid[..., None].to(region.dtype)
        strength = scores.flatten(-2).amax(-1).masked_fill(~region_valid, 0.0)
        return (
            region.flatten(1, 2),
            region_valid.flatten(1, 2),
            strength.flatten(1, 2),
            selected,
            selected_score,
        )


class SparseRegionGraphPropagation(nn.Module):
    def __init__(self, hidden_dim, rounds=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(rounds)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(rounds)])
        self.pool_score = nn.Linear(hidden_dim, 1)

    def forward(self, tokens, mask, selected_pockets, batch):
        batch_size, num_fragments, select_k = selected_pockets.shape
        num_regions = num_fragments * select_k
        fragment_ids = torch.arange(
            num_fragments, device=tokens.device
        )[:, None].expand(-1, select_k).reshape(-1)
        pocket_ids = selected_pockets.reshape(batch_size, num_regions)

        fi = fragment_ids[None, :, None].expand(batch_size, -1, num_regions)
        fj = fragment_ids[None, None, :].expand(batch_size, num_regions, -1)
        pi = pocket_ids[:, :, None].expand(-1, -1, num_regions)
        pj = pocket_ids[:, None, :].expand(-1, num_regions, -1)
        batch_index = torch.arange(batch_size, device=tokens.device)[:, None, None]
        fragment_link = batch["fragment_adjacency"][batch_index, fi, fj]
        pocket_link = batch["pocket_adjacency"][batch_index, pi, pj]
        same_fragment = fi == fj
        same_pocket = pi == pj
        adjacency = (
            same_fragment.to(tokens.dtype) * pocket_link
            + same_pocket.to(tokens.dtype) * fragment_link
            + fragment_link * pocket_link
        ).clamp_max(1.0)
        valid_pair = mask[:, :, None] & mask[:, None, :]
        adjacency = adjacency * valid_pair.to(adjacency.dtype)
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        x = tokens
        for layer, norm in zip(self.layers, self.norms):
            message = torch.matmul(adjacency, x) / degree
            x = norm(x + layer(torch.cat([x, message], dim=-1)))
            x = x * mask[..., None].to(x.dtype)
        score = self.pool_score(x).squeeze(-1).masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=-1) * mask.to(score.dtype)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (weight[..., None] * x).sum(1)
        return x, pooled, weight, adjacency


class CaviarSubpocketDTA(nn.Module):
    def __init__(
        self,
        drug_1d_in_dim=768,
        atom_v2_dim=52,
        protein_1d_in_dim=1280,
        protein_node_s_dim=6,
        hidden_dim=128,
        dropout=0.1,
        subpocket_rounds=2,
        region_rounds=2,
        pockets_per_fragment=4,
        task="regression",
        delta_max=1.0,
        ar_gate_epsilon=0.05,
        fp_gate_max=0.4,
    ):
        super().__init__()
        self.delta_max = float(delta_max)
        self.ar_gate_epsilon = float(ar_gate_epsilon)
        self.fp_gate_max = float(fp_gate_max)
        self.drug_1d_encoder = Drug1DEncoder(drug_1d_in_dim, hidden_dim)
        self.drug_atom_encoder = BondAwareDrugEGNNEncoder(
            atom_v2_dim, 14, hidden_dim, dropout
        )
        self.drug_fusion = ConcatFusion(
            [hidden_dim, hidden_dim], hidden_dim, hidden_dim * 2, dropout
        )
        self.protein_1d_encoder = Protein1DEncoder(protein_1d_in_dim, hidden_dim)
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
            [hidden_dim, hidden_dim], hidden_dim, hidden_dim * 2, dropout
        )
        self.subpocket_gcn = SharedSubpocketGCN(
            hidden_dim, subpocket_rounds, dropout
        )
        self.subpocket_graph = SubpocketGraphPropagation(
            hidden_dim, subpocket_rounds, 3, dropout
        )
        self.region_builder = SparseAtomResidueRegionBuilder(
            hidden_dim, pockets_per_fragment, dropout
        )
        self.region_propagation = SparseRegionGraphPropagation(
            hidden_dim, region_rounds, dropout
        )
        self.ar_pool_score = nn.Linear(hidden_dim, 1)
        self.global_decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
        )
        self.ar_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.fp_gate = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.ar_delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.fp_delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
        residue_aux = torch.cat(
            [
                self.residue_type_embedding(
                    residue["residue_type_index"].clamp(0, 31)
                ),
                residue["sequence_scalar3"],
                residue["esm_per_tok"],
                residue["esm_mask"].float().unsqueeze(-1),
                residue["structure_extra5"],
            ],
            dim=-1,
        )
        residue_tokens = self.residue_norm(
            protein_out["node_feat"] + self.residue_aux_proj(residue_aux)
        )
        protein_local_global = global_mean_pool(
            residue_tokens, protein_out["batch"]
        )
        protein_feat = self.protein_fusion(
            [protein_1d, protein_out["graph_feat"] + protein_local_global]
        )

        pocket_residue, pocket_tokens, pocket_residue_weight = self.subpocket_gcn(
            residue_tokens, batch
        )
        pocket_tokens = self.subpocket_graph(pocket_tokens, batch)
        regions, region_mask, region_strength, selected, selected_score = (
            self.region_builder(
                drug_out["node_feat"],
                pocket_residue,
                pocket_tokens,
                batch,
            )
        )
        ar_score = self.ar_pool_score(regions).squeeze(-1).masked_fill(
            ~region_mask, -1e9
        )
        ar_weight = torch.softmax(ar_score, dim=-1)
        ar_feat = (ar_weight[:, :, None] * regions).sum(1)
        propagated, fp_feat, fp_weight, region_adjacency = self.region_propagation(
            regions, region_mask, selected, batch
        )

        base_pred = self.global_decoder(
            torch.cat([drug_feat, protein_feat], dim=-1)
        )
        ar_features = torch.cat([drug_feat, protein_feat, ar_feat], dim=-1)
        ar_delta_raw = self.ar_delta_head(ar_features).view_as(base_pred)
        fp_features = torch.cat(
            [drug_feat, protein_feat, ar_feat, fp_feat, fp_feat - ar_feat],
            dim=-1,
        )
        fp_delta_raw = self.fp_delta_head(fp_features).view_as(base_pred)
        ar_delta = self.delta_max * torch.tanh(ar_delta_raw)
        fp_delta = self.delta_max * torch.tanh(fp_delta_raw)
        ar_gate_logit = self.ar_gate(ar_features)
        ar_gate = self.ar_gate_epsilon + (
            1.0 - 2.0 * self.ar_gate_epsilon
        ) * torch.sigmoid(ar_gate_logit)
        gated_ar = ar_gate.view_as(base_pred) * ar_delta
        pred_after_ar = base_pred + gated_ar
        fp_gate_logit = self.fp_gate(
            torch.cat([fp_features, gated_ar.detach()], dim=-1)
        )
        fp_gate = self.fp_gate_max * torch.sigmoid(fp_gate_logit)
        gated_fp = fp_gate.view_as(base_pred) * fp_delta
        pred = pred_after_ar + gated_fp

        if not return_details:
            return pred
        return {
            "pred": pred,
            "base_pred": base_pred,
            "ar_candidate": base_pred + ar_delta,
            "pred_after_ar": pred_after_ar,
            "fp_candidate": pred_after_ar + fp_delta,
            "ar_delta": ar_delta,
            "fp_delta": fp_delta,
            "ar_delta_raw": ar_delta_raw,
            "fp_delta_raw": fp_delta_raw,
            "scale_gates": torch.cat([ar_gate, fp_gate], dim=-1),
            "scale_gate_logits": torch.cat(
                [ar_gate_logit, fp_gate_logit], dim=-1
            ),
            "region_mask": region_mask,
            "region_strength": region_strength,
            "ar_region_weight": ar_weight,
            "fp_region_weight": fp_weight,
            "region_tokens": propagated,
            "selected_subpockets": selected,
            "selection_score": selected_score,
            "subpocket_residue_weight": pocket_residue_weight,
            "region_adjacency": region_adjacency,
        }
