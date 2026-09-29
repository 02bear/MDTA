"""Joint AR -> FP -> Global affinity model using BRICS and CAVIAR graphs.

The model has one affinity prediction head. Atom-residue interactions create
fragment-subpocket edge evidence; those edges drive cross-modal propagation
between a BRICS fragment graph and a CAVIAR subpocket graph; the resulting FP
evidence is aggregated together with the original whole-drug/whole-protein
representation for the final prediction.
"""

import math

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.drug_1d_encoder import Drug1DEncoder
from models.fusion import ConcatFusion
from models.model_p13d_caviar_subpocket import (
    SharedSubpocketGCN,
    SubpocketGraphPropagation,
)
from models.model_p13d_hierarchical_multiscale_v41 import (
    BondAwareDrugEGNNEncoder,
)
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder


class DenseGraphPropagation(nn.Module):
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
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(rounds)]
        )

    def forward(self, x, adjacency, mask):
        adjacency = adjacency.to(x.dtype)
        valid_pair = mask[:, :, None] & mask[:, None, :]
        adjacency = adjacency * valid_pair.to(adjacency.dtype)
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        for layer, norm in zip(self.layers, self.norms):
            message = torch.matmul(adjacency, x) / degree
            x = norm(x + layer(torch.cat([x, message], dim=-1)))
            x = x * mask[..., None].to(x.dtype)
        return x


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask, dim=1):
        score = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=dim) * mask.to(score.dtype)
        weight = weight / weight.sum(dim, keepdim=True).clamp_min(1e-8)
        pooled = (weight[..., None] * x).sum(dim)
        return pooled, weight


class AtomResidueEdgeBuilder(nn.Module):
    """Create FP cross-edge evidence from atoms and residues."""

    def __init__(self, hidden_dim, pockets_per_fragment=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pockets_per_fragment = int(pockets_per_fragment)
        self.fragment_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pocket_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.global_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atom_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residue_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fragment_atom_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pocket_residue_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_proj = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU())
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.soft_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atom_tokens,
        fragment_tokens,
        pocket_residue_tokens,
        pocket_tokens,
        global_context,
        batch,
    ):
        fragment_indices = batch["fragment_atom_indices"].clamp_min(0)
        fragment_atom_mask = batch["fragment_atom_mask"]
        fragment_mask = batch["fragment_mask"]
        pocket_mask = batch["pocket_mask"]
        atoms = atom_tokens[fragment_indices]

        compatibility = torch.einsum(
            "bfh,bph->bfp",
            self.fragment_query(fragment_tokens)
            + self.global_query(global_context)[:, None, :],
            self.pocket_key(pocket_tokens),
        ) / math.sqrt(self.hidden_dim)
        valid_fp = fragment_mask[:, :, None] & pocket_mask[:, None, :]
        compatibility = compatibility.masked_fill(~valid_fp, -1e4)
        soft_weight = torch.softmax(compatibility, dim=-1)
        soft_weight = soft_weight * valid_fp.to(soft_weight.dtype)
        soft_weight = soft_weight / soft_weight.sum(-1, keepdim=True).clamp_min(
            1e-8
        )
        soft_pocket_context = torch.einsum(
            "bfp,bph->bfh", soft_weight, pocket_tokens
        )

        select_k = min(self.pockets_per_fragment, pocket_tokens.size(1))
        selected_score, selected = torch.topk(
            compatibility, k=select_k, dim=-1
        )
        batch_index = torch.arange(
            atoms.size(0), device=atoms.device
        )[:, None, None]
        selected_pocket = pocket_tokens[batch_index, selected]
        selected_residues = pocket_residue_tokens[batch_index, selected]
        selected_residue_mask = batch["pocket_residue_mask"][
            batch_index, selected
        ]
        selected_context = batch["pocket_context6"][batch_index, selected]
        selected_pocket_mask = pocket_mask[batch_index, selected]
        edge_mask = fragment_mask[:, :, None] & selected_pocket_mask

        atom_query = (
            self.atom_q(atoms)
            + self.fragment_atom_context(fragment_tokens)[:, :, None, :]
        )
        residue_key = self.residue_k(selected_residues) + (
            self.pocket_residue_context(selected_pocket)[:, :, :, None, :]
        )
        interaction_score = torch.einsum(
            "bfah,bfkrh->bfkar", atom_query, residue_key
        ) / math.sqrt(self.hidden_dim)
        pair_mask = (
            fragment_atom_mask[:, :, None, :, None]
            & selected_residue_mask[:, :, :, None, :]
            & edge_mask[:, :, :, None, None]
        )
        interaction_score = interaction_score.masked_fill(~pair_mask, -1e4)
        interaction_weight = torch.softmax(
            interaction_score.flatten(-2), dim=-1
        ).view_as(interaction_score)
        interaction_weight = interaction_weight * pair_mask.to(
            interaction_weight.dtype
        )
        interaction_weight = interaction_weight / interaction_weight.sum(
            (-2, -1), keepdim=True
        ).clamp_min(1e-8)

        atom_attention = interaction_weight.sum(-1)
        residue_attention = interaction_weight.sum(-2)
        atom_summary = torch.einsum(
            "bfka,bfah->bfkh", atom_attention, atoms
        )
        residue_summary = torch.einsum(
            "bfkr,bfkrh->bfkh", residue_attention, selected_residues
        )
        product = torch.einsum(
            "bfkar,bfah,bfkrh->bfkh",
            interaction_weight,
            atoms,
            selected_residues,
        )
        atom_square = torch.einsum(
            "bfka,bfah->bfkh", atom_attention, atoms.square()
        )
        residue_square = torch.einsum(
            "bfkr,bfkrh->bfkh",
            residue_attention,
            selected_residues.square(),
        )
        distance = torch.log1p(
            (atom_square + residue_square - 2.0 * product).clamp_min(0.0)
        )
        context_weight = selected_residue_mask.to(selected_context.dtype)
        context = (
            selected_context * context_weight[..., None]
        ).sum(-2) / context_weight.sum(-1, keepdim=True).clamp_min(1.0)
        context = self.context_proj(context)
        fragment_expanded = fragment_tokens[:, :, None, :].expand(
            -1, -1, select_k, -1
        )
        global_expanded = global_context[:, None, None, :].expand(
            -1, fragment_tokens.size(1), select_k, -1
        )
        edge = self.edge_mlp(
            torch.cat(
                [
                    fragment_expanded,
                    selected_pocket,
                    atom_summary,
                    residue_summary,
                    product,
                    distance,
                    context,
                    global_expanded,
                ],
                dim=-1,
            )
        )
        edge = edge + self.soft_context(soft_pocket_context)[:, :, None, :]
        selection_confidence = 0.5 + torch.sigmoid(selected_score)[..., None]
        edge = self.edge_norm(edge) * selection_confidence
        edge = edge * edge_mask[..., None].to(edge.dtype)
        return {
            "edge_tokens": edge,
            "edge_mask": edge_mask,
            "selected_pockets": selected,
            "selected_score": selected_score,
            "compatibility": compatibility,
            "soft_selection_weight": soft_weight,
            "interaction_weight": interaction_weight,
            "interaction_score": interaction_score,
        }


class FragmentPocketCrossPropagation(nn.Module):
    """Alternating messages over AR-supported fragment-subpocket edges."""

    def __init__(self, hidden_dim, rounds=2, dropout=0.1):
        super().__init__()
        self.rounds = int(rounds)
        self.edge_gate = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(rounds)]
        )
        self.fragment_update = nn.ModuleList(
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
        self.pocket_update = nn.ModuleList(
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
        self.edge_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(rounds)
            ]
        )
        self.fragment_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(rounds)]
        )
        self.pocket_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(rounds)]
        )
        self.edge_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(rounds)]
        )

    def forward(
        self,
        fragment,
        pocket,
        edge,
        edge_mask,
        selected_pockets,
        fragment_mask,
        pocket_mask,
    ):
        batch_size, num_fragments, select_k, hidden_dim = edge.shape
        batch_index = torch.arange(
            batch_size, device=edge.device
        )[:, None, None]
        for round_index in range(self.rounds):
            gate = torch.sigmoid(
                self.edge_gate[round_index](edge).squeeze(-1)
            ) * edge_mask.to(edge.dtype)
            fragment_denominator = gate.sum(-1, keepdim=True).clamp_min(1e-8)
            fragment_message = (
                gate[..., None] * edge
            ).sum(2) / fragment_denominator
            fragment = self.fragment_norm[round_index](
                fragment
                + self.fragment_update[round_index](
                    torch.cat([fragment, fragment_message], dim=-1)
                )
            )
            fragment = fragment * fragment_mask[..., None].to(fragment.dtype)

            pocket_message = torch.zeros_like(pocket)
            pocket_weight = torch.zeros(
                pocket.shape[:2], dtype=edge.dtype, device=edge.device
            )
            for k in range(select_k):
                index = selected_pockets[:, :, k]
                weighted_edge = edge[:, :, k] * gate[:, :, k, None]
                pocket_message.scatter_add_(
                    1, index[..., None].expand(-1, -1, hidden_dim), weighted_edge
                )
                pocket_weight.scatter_add_(1, index, gate[:, :, k])
            pocket_message = pocket_message / pocket_weight[..., None].clamp_min(
                1e-8
            )
            pocket = self.pocket_norm[round_index](
                pocket
                + self.pocket_update[round_index](
                    torch.cat([pocket, pocket_message], dim=-1)
                )
            )
            pocket = pocket * pocket_mask[..., None].to(pocket.dtype)

            selected_pocket = pocket[batch_index, selected_pockets]
            selected_fragment = fragment[:, :, None, :].expand(
                -1, -1, select_k, -1
            )
            edge = self.edge_norm[round_index](
                edge
                + self.edge_update[round_index](
                    torch.cat(
                        [edge, selected_fragment, selected_pocket], dim=-1
                    )
                )
            )
            edge = edge * edge_mask[..., None].to(edge.dtype)
        return fragment, pocket, edge


class ThreeGranularityCaviarDTA(nn.Module):
    def __init__(
        self,
        drug_1d_in_dim=768,
        atom_v2_dim=52,
        protein_1d_in_dim=1280,
        protein_node_s_dim=6,
        hidden_dim=128,
        dropout=0.1,
        fragment_rounds=2,
        subpocket_rounds=2,
        cross_rounds=2,
        pockets_per_fragment=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drug_1d_encoder = Drug1DEncoder(drug_1d_in_dim, hidden_dim)
        self.drug_atom_encoder = BondAwareDrugEGNNEncoder(
            atom_v2_dim, 14, hidden_dim, dropout
        )
        self.drug_fusion = ConcatFusion(
            [hidden_dim, hidden_dim], hidden_dim, hidden_dim * 2, dropout
        )
        self.protein_1d_encoder = Protein1DEncoder(
            protein_1d_in_dim, hidden_dim
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
            [hidden_dim, hidden_dim], hidden_dim, hidden_dim * 2, dropout
        )

        self.fragment_atom_pool = MaskedAttentionPool(hidden_dim)
        self.fragment_graph = DenseGraphPropagation(
            hidden_dim, fragment_rounds, dropout
        )
        self.subpocket_gcn = SharedSubpocketGCN(
            hidden_dim, subpocket_rounds, dropout
        )
        self.subpocket_graph = SubpocketGraphPropagation(
            hidden_dim, subpocket_rounds, 3, dropout
        )
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fragment_global_context = nn.Linear(hidden_dim, hidden_dim)
        self.pocket_global_context = nn.Linear(hidden_dim, hidden_dim)
        self.fragment_context_norm = nn.LayerNorm(hidden_dim)
        self.pocket_context_norm = nn.LayerNorm(hidden_dim)
        self.ar_edge_builder = AtomResidueEdgeBuilder(
            hidden_dim, pockets_per_fragment, dropout
        )
        self.cross_propagation = FragmentPocketCrossPropagation(
            hidden_dim, cross_rounds, dropout
        )

        self.fragment_pool = MaskedAttentionPool(hidden_dim)
        self.pocket_pool = MaskedAttentionPool(hidden_dim)
        self.ar_edge_pool = MaskedAttentionPool(hidden_dim)
        self.fp_edge_pool = MaskedAttentionPool(hidden_dim)
        self.final_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.affinity_head = nn.Linear(hidden_dim, 1)
        self.activity_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.affinity_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.affinity_head.bias, 5.0)
        nn.init.zeros_(self.activity_head.weight)
        nn.init.zeros_(self.activity_head.bias)

    def encode_global(self, batch):
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
        return (
            drug_feat,
            protein_feat,
            drug_out["node_feat"],
            residue_tokens,
        )

    def forward(self, batch, return_details=False):
        drug_feat, protein_feat, atom_tokens, residue_tokens = (
            self.encode_global(batch)
        )
        fragment_indices = batch["fragment_atom_indices"].clamp_min(0)
        fragment_atoms = atom_tokens[fragment_indices]
        fragment_tokens, fragment_atom_weight = self.fragment_atom_pool(
            fragment_atoms, batch["fragment_atom_mask"], dim=2
        )
        fragment_tokens = self.fragment_graph(
            fragment_tokens,
            batch["fragment_adjacency"],
            batch["fragment_mask"],
        )
        pocket_residue, pocket_tokens, pocket_residue_weight = (
            self.subpocket_gcn(residue_tokens, batch)
        )
        pocket_tokens = self.subpocket_graph(pocket_tokens, batch)

        global_context = self.global_context(
            torch.cat([drug_feat, protein_feat], dim=-1)
        )
        fragment_tokens = self.fragment_context_norm(
            fragment_tokens + self.fragment_global_context(global_context)[:, None]
        ) * batch["fragment_mask"][..., None].to(fragment_tokens.dtype)
        pocket_tokens = self.pocket_context_norm(
            pocket_tokens + self.pocket_global_context(global_context)[:, None]
        ) * batch["pocket_mask"][..., None].to(pocket_tokens.dtype)

        ar = self.ar_edge_builder(
            atom_tokens,
            fragment_tokens,
            pocket_residue,
            pocket_tokens,
            global_context,
            batch,
        )
        ar_edge_tokens = ar["edge_tokens"]
        fragment_tokens, pocket_tokens, fp_edge_tokens = (
            self.cross_propagation(
                fragment_tokens,
                pocket_tokens,
                ar_edge_tokens,
                ar["edge_mask"],
                ar["selected_pockets"],
                batch["fragment_mask"],
                batch["pocket_mask"],
            )
        )

        fragment_evidence, fragment_weight = self.fragment_pool(
            fragment_tokens, batch["fragment_mask"]
        )
        pocket_evidence, pocket_weight = self.pocket_pool(
            pocket_tokens, batch["pocket_mask"]
        )
        edge_mask_flat = ar["edge_mask"].flatten(1, 2)
        ar_evidence, ar_edge_weight = self.ar_edge_pool(
            ar_edge_tokens.flatten(1, 2), edge_mask_flat
        )
        fp_evidence, fp_edge_weight = self.fp_edge_pool(
            fp_edge_tokens.flatten(1, 2), edge_mask_flat
        )
        final_latent = self.final_trunk(
            torch.cat(
                [
                    drug_feat,
                    protein_feat,
                    fragment_evidence,
                    pocket_evidence,
                    ar_evidence,
                    fp_evidence,
                ],
                dim=-1,
            )
        )
        affinity_mean = self.affinity_head(final_latent)
        activity_logit = self.activity_head(final_latent)
        if not return_details:
            return affinity_mean
        return {
            "pred": affinity_mean,
            "affinity_mean": affinity_mean,
            "activity_logit": activity_logit,
            "fragment_tokens": fragment_tokens,
            "pocket_tokens": pocket_tokens,
            "ar_edge_tokens": ar_edge_tokens,
            "fp_edge_tokens": fp_edge_tokens,
            "edge_mask": ar["edge_mask"],
            "selected_subpockets": ar["selected_pockets"],
            "selection_score": ar["selected_score"],
            "compatibility": ar["compatibility"],
            "soft_selection_weight": ar["soft_selection_weight"],
            "interaction_weight": ar["interaction_weight"],
            "fragment_atom_weight": fragment_atom_weight,
            "subpocket_residue_weight": pocket_residue_weight,
            "fragment_weight": fragment_weight,
            "pocket_weight": pocket_weight,
            "ar_edge_weight": ar_edge_weight,
            "fp_edge_weight": fp_edge_weight,
        }
