"""Three-grain CAVIAR DTA v3.

The model keeps three explicit interaction grains:

1. atom--residue (AR) interaction restricted to atoms of one BRICS fragment
   and residues of one CAVIAR subpocket;
2. fragment--pocket (FP) interaction over the BRICS and CAVIAR graphs;
3. whole-drug--whole-protein (Global) interaction.

AR and FP are alternated twice (AR1 -> FP1 -> AR2 -> FP2).  Original atom
and residue tokens are immutable while a separate working stream is updated
with gated residuals.  Every valid fragment--pocket pair is a candidate, but
independent sigmoid gates can turn every candidate off; there is no fixed
top-k and no learned null node.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch.utils.checkpoint import checkpoint

from models.drug_1d_encoder import Drug1DEncoder
from models.fusion import ConcatFusion
from models.model_p13d_hierarchical_multiscale_v41 import (
    BondAwareDrugEGNNEncoder,
)
from models.protein_1d_encoder import Protein1DEncoder
from models.protein_3d_egnn_encoder import Protein3DEGNNEncoder


def masked_mean(x, mask, dim):
    weight = mask.to(x.dtype)
    numerator = (x * weight[..., None]).sum(dim)
    denominator = weight.sum(dim).clamp_min(1.0)[..., None]
    return numerator / denominator


def masked_softmax(score, mask, dim):
    score = score.masked_fill(~mask, -1e4)
    weight = torch.softmax(score, dim=dim) * mask.to(score.dtype)
    return weight / weight.sum(dim=dim, keepdim=True).clamp_min(1e-8)


def scatter_padded_mean(values, indices, mask, total_nodes):
    """Scatter padded [.., H] values back to a packed node tensor."""
    hidden_dim = values.size(-1)
    output = values.new_zeros((total_nodes, hidden_dim))
    counts = values.new_zeros((total_nodes, 1))
    flat_mask = mask.reshape(-1)
    if flat_mask.any():
        flat_indices = indices.reshape(-1)[flat_mask]
        flat_values = values.reshape(-1, hidden_dim)[flat_mask]
        output.index_add_(0, flat_indices, flat_values)
        counts.index_add_(
            0,
            flat_indices,
            values.new_ones((flat_indices.numel(), 1)),
        )
    valid = counts.squeeze(-1) > 0
    return output / counts.clamp_min(1.0), valid


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask, dim=1):
        score = self.score(x).squeeze(-1)
        weight = masked_softmax(score, mask, dim)
        return (weight[..., None] * x).sum(dim), weight


class GatedMaskedPool(nn.Module):
    """Attention pooling whose magnitude vanishes when all gates vanish."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask, gate):
        score = self.score(x).squeeze(-1)
        attention = masked_softmax(score, mask, dim=1)
        gate = gate.to(attention.dtype) * mask.to(attention.dtype)
        effective = attention * gate
        content_weight = effective / effective.sum(1, keepdim=True).clamp_min(1e-8)
        content = (content_weight[..., None] * x).sum(1)
        activity = gate.sum(1, keepdim=True) / (1.0 + gate.sum(1, keepdim=True))
        return content * activity, content_weight


class DualTokenFuse(nn.Module):
    """Expose both immutable base tokens and the current working tokens."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, base, work):
        return self.norm(base + self.net(torch.cat([base, work], dim=-1)))


class GatedResidualUpdate(nn.Module):
    """Residual update with a conservative, per-feature learned gate."""

    def __init__(self, hidden_dim, dropout, gate_bias=-2.0):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x, message, valid=None, activity=None):
        # Pre-norm residual: activity=0 is an exact identity path.  This is
        # important for independently closable AR/FP edges; post-norm would
        # still alter x even when the cross message is exactly zero.
        pair = torch.cat([self.input_norm(x), message], dim=-1)
        gate = torch.sigmoid(self.gate(pair))
        delta = self.delta(pair)
        if valid is not None:
            valid_float = valid[..., None].to(x.dtype)
            gate = gate * valid_float
            delta = delta * valid_float
        if activity is None:
            activity_float = 1.0
        else:
            activity_float = activity[..., None].to(x.dtype)
        effective_gate = activity_float * gate
        updated = x + effective_gate * delta
        if valid is not None:
            updated = torch.where(valid[..., None], updated, x)
        return updated, effective_gate


class DenseGraphResidual(nn.Module):
    """One residual propagation step over a supplied dense adjacency mask."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.update = GatedResidualUpdate(hidden_dim, dropout, gate_bias=-1.0)

    def forward(self, x, adjacency, mask):
        adjacency = adjacency.to(x.dtype)
        valid_pair = mask[:, :, None] & mask[:, None, :]
        adjacency = adjacency * valid_pair.to(adjacency.dtype)
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        message = torch.matmul(adjacency, x) / degree
        updated, gate = self.update(x, message, mask)
        return updated * mask[..., None].to(updated.dtype), gate


class PocketResidueEncoder(nn.Module):
    """Shared spatial message passing inside every CAVIAR subpocket."""

    def __init__(self, hidden_dim, dropout, message_passing=True):
        super().__init__()
        self.message_passing = bool(message_passing)
        if self.message_passing:
            self.update = GatedResidualUpdate(hidden_dim, dropout, gate_bias=-1.0)
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(self, residue_tokens, batch):
        indices = batch["pocket_residue_indices"].clamp_min(0)
        mask = batch["pocket_residue_mask"]
        x = residue_tokens[indices] * mask[..., None].to(residue_tokens.dtype)
        if self.message_passing:
            adjacency = batch["pocket_residue_adjacency"].to(x.dtype)
            valid_pair = mask[..., :, None] & mask[..., None, :]
            adjacency = adjacency * valid_pair.to(adjacency.dtype)
            degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
            message = torch.matmul(adjacency, x) / degree
            x, update_gate = self.update(x, message, mask)
        else:
            update_gate = torch.zeros_like(x)
        pooled, weight = self.pool(x, mask, dim=2)
        pocket_mask = batch["pocket_mask"]
        pooled = pooled * pocket_mask[..., None].to(pooled.dtype)
        return x, pooled, weight, update_gate


class PocketGraphResidual(nn.Module):
    """CAVIAR graph propagation where non-self weak edges can reach zero."""

    def __init__(self, hidden_dim, edge_dim, dropout):
        super().__init__()
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update = GatedResidualUpdate(hidden_dim, dropout, gate_bias=-1.0)

    def forward(self, x, batch):
        mask = batch["pocket_mask"]
        adjacency_mask = batch["pocket_adjacency"].to(x.dtype)
        learned = torch.sigmoid(
            self.edge_gate(batch["pocket_edge_features"]).squeeze(-1)
        )
        n_pocket = x.size(1)
        eye = torch.eye(n_pocket, dtype=torch.bool, device=x.device)[None]
        edge_weight = torch.where(eye, torch.ones_like(learned), learned)
        valid_pair = mask[:, :, None] & mask[:, None, :]
        adjacency = (
            adjacency_mask
            * edge_weight.to(x.dtype)
            * valid_pair.to(x.dtype)
        )
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        message = torch.matmul(adjacency, x) / degree
        updated, node_gate = self.update(x, message, mask)
        return (
            updated * mask[..., None].to(updated.dtype),
            edge_weight * adjacency_mask * valid_pair.to(edge_weight.dtype),
            node_gate,
        )


class LocalNodeBuilder(nn.Module):
    """Pool atom/residue streams into fragment and subpocket nodes."""

    def __init__(self, hidden_dim, dropout, pocket_message_passing=True):
        super().__init__()
        self.atom_fuse = DualTokenFuse(hidden_dim, dropout)
        self.residue_fuse = DualTokenFuse(hidden_dim, dropout)
        self.fragment_pool = MaskedAttentionPool(hidden_dim)
        self.pocket_residue = PocketResidueEncoder(
            hidden_dim, dropout, message_passing=pocket_message_passing
        )

    def forward(self, atom_base, atom_work, residue_base, residue_work, batch):
        atom_state = self.atom_fuse(atom_base, atom_work)
        atom_indices = batch["fragment_atom_indices"].clamp_min(0)
        fragment_atoms = atom_state[atom_indices]
        fragment, fragment_weight = self.fragment_pool(
            fragment_atoms,
            batch["fragment_atom_mask"],
            dim=2,
        )
        fragment = fragment * batch["fragment_mask"][..., None].to(fragment.dtype)

        residue_state = self.residue_fuse(residue_base, residue_work)
        pocket_residue, pocket, pocket_weight, pocket_update_gate = (
            self.pocket_residue(residue_state, batch)
        )
        return {
            "fragment": fragment,
            "pocket": pocket,
            "fragment_atom_weight": fragment_weight,
            "pocket_residue_weight": pocket_weight,
            "pocket_residue_tokens": pocket_residue,
            "pocket_residue_update_gate": pocket_update_gate,
        }


class AtomResidueInteractionBlock(nn.Module):
    """Full-candidate, chunked AR interaction with packed-token updates."""

    def __init__(self, hidden_dim, dropout, pocket_chunk_size=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pocket_chunk_size = max(int(pocket_chunk_size), 1)
        self.atom_fuse = DualTokenFuse(hidden_dim, dropout)
        self.residue_fuse = DualTokenFuse(hidden_dim, dropout)
        self.atom_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atom_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residue_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residue_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fragment_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pocket_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context6 = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU())
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.edge_gate = nn.Linear(hidden_dim, 1)
        self.atom_message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residue_message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.atom_update = GatedResidualUpdate(hidden_dim, dropout, gate_bias=-2.0)
        self.residue_update = GatedResidualUpdate(
            hidden_dim, dropout, gate_bias=-2.0
        )
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.constant_(self.edge_gate.bias, -1.0)

    def _forward_chunk(
        self,
        atom_query,
        atom_value,
        fragment_atom_mask,
        fragment,
        residue_chunk,
        residue_mask_chunk,
        pocket_chunk,
        pair_mask_chunk,
        context6_chunk,
    ):
        """Compute one pocket chunk; checkpointing avoids saving dense AR maps."""
        batch_size, num_fragments, num_atoms, _ = atom_query.shape
        chunk_size = pocket_chunk.size(1)
        num_residues = residue_chunk.size(2)
        residue_key = self.residue_key(residue_chunk) + self.pocket_context(
            pocket_chunk
        )[:, :, None, :]
        residue_value = self.residue_value(residue_chunk)

        # Compute both the dot products and the normalizations in FP32.  Casting
        # only after einsum would still form BF16/FP16 logits under autocast.
        score = torch.einsum(
            "bfah,bcrh->bfcar", atom_query.float(), residue_key.float()
        ) / math.sqrt(self.hidden_dim)
        full_mask = (
            fragment_atom_mask[:, :, None, :, None]
            & residue_mask_chunk[:, None, :, None, :]
            & pair_mask_chunk[:, :, :, None, None]
        )
        atom_to_residue = masked_softmax(score, full_mask, dim=-1)
        residue_to_atom = masked_softmax(score, full_mask, dim=-2)
        atom_context = torch.einsum(
            "bfcar,bcrh->bfcah",
            atom_to_residue.to(residue_value.dtype),
            residue_value,
        )
        residue_context = torch.einsum(
            "bfcar,bfah->bfcrh",
            residue_to_atom.to(atom_value.dtype),
            atom_value,
        )

        atom_summary = masked_mean(
            atom_context,
            fragment_atom_mask[:, :, None, :].expand(
                -1, -1, chunk_size, -1
            ),
            dim=3,
        )
        residue_summary = masked_mean(
            residue_context,
            residue_mask_chunk[:, None, :, :].expand(
                -1, num_fragments, -1, -1
            ),
            dim=3,
        )
        fragment_expanded = fragment[:, :, None, :].expand(
            -1, -1, chunk_size, -1
        )
        pocket_expanded = pocket_chunk[:, None, :, :].expand(
            -1, num_fragments, -1, -1
        )
        context = masked_mean(
            context6_chunk,
            residue_mask_chunk,
            dim=2,
        )
        context = self.context6(context)[:, None, :, :].expand(
            -1, num_fragments, -1, -1
        )
        edge = self.edge_norm(
            self.edge_mlp(
                torch.cat(
                    [
                        fragment_expanded,
                        pocket_expanded,
                        atom_summary,
                        residue_summary,
                        fragment_expanded * pocket_expanded,
                        torch.abs(fragment_expanded - pocket_expanded),
                        context,
                    ],
                    dim=-1,
                )
            )
        )
        edge = edge * pair_mask_chunk[..., None].to(edge.dtype)
        gate = torch.sigmoid(self.edge_gate(edge).squeeze(-1))
        gate = gate * pair_mask_chunk.to(gate.dtype)

        edge_for_atom = edge[:, :, :, None, :].expand(
            -1, -1, -1, num_atoms, -1
        )
        atom_candidate = self.atom_message(
            torch.cat([atom_context, edge_for_atom], dim=-1)
        )
        atom_valid = fragment_atom_mask[:, :, None, :].to(gate.dtype)
        atom_effective = gate[..., None] * atom_valid
        atom_numerator = (
            atom_candidate * atom_effective[..., None]
        ).sum(2)
        atom_denominator = atom_effective.sum(2)

        edge_for_residue = edge[:, :, :, None, :].expand(
            -1, -1, -1, num_residues, -1
        )
        residue_candidate = self.residue_message(
            torch.cat([residue_context, edge_for_residue], dim=-1)
        )
        residue_valid = residue_mask_chunk[:, None, :, :].to(gate.dtype)
        residue_effective = gate[..., None] * residue_valid
        residue_numerator = (
            residue_candidate * residue_effective[..., None]
        ).sum(1)
        residue_denominator = residue_effective.sum(1)

        # Diagnostic only.  It must not keep the dense attention graph alive.
        with torch.no_grad():
            joint = atom_to_residue * full_mask.to(atom_to_residue.dtype)
            entropy = -(joint.clamp_min(1e-8).log() * joint).sum((-2, -1))
            entropy = entropy / fragment_atom_mask.sum(-1).clamp_min(1)[
                :, :, None
            ].to(entropy.dtype)
        return (
            atom_numerator,
            atom_denominator,
            residue_numerator,
            residue_denominator,
            edge,
            gate,
            entropy,
        )

    def forward(
        self,
        atom_base,
        atom_work,
        residue_base,
        residue_work,
        fragment,
        pocket,
        batch,
    ):
        fragment_indices = batch["fragment_atom_indices"].clamp_min(0)
        fragment_atom_mask = batch["fragment_atom_mask"]
        pocket_indices = batch["pocket_residue_indices"].clamp_min(0)
        pocket_residue_mask = batch["pocket_residue_mask"]
        fragment_mask = batch["fragment_mask"]
        pocket_mask = batch["pocket_mask"]
        edge_mask = fragment_mask[:, :, None] & pocket_mask[:, None, :]

        atom_state_packed = self.atom_fuse(atom_base, atom_work)
        residue_state_packed = self.residue_fuse(residue_base, residue_work)
        atoms = atom_state_packed[fragment_indices]
        residues = residue_state_packed[pocket_indices]

        atom_query = self.atom_query(atoms) + self.fragment_context(fragment)[
            :, :, None, :
        ]
        atom_value = self.atom_value(atoms)

        batch_size, num_fragments, num_atoms, _ = atoms.shape
        num_pockets = pocket.size(1)
        num_residues = residues.size(2)
        atom_numerator = atoms.new_zeros(
            (batch_size, num_fragments, num_atoms, self.hidden_dim)
        )
        atom_denominator = atoms.new_zeros(
            (batch_size, num_fragments, num_atoms)
        )
        residue_numerator_chunks = []
        residue_denominator_chunks = []

        edge_chunks = []
        gate_chunks = []
        interaction_entropy_chunks = []
        for start in range(0, num_pockets, self.pocket_chunk_size):
            end = min(start + self.pocket_chunk_size, num_pockets)
            residue_chunk = residues[:, start:end]
            residue_mask_chunk = pocket_residue_mask[:, start:end]
            pocket_chunk = pocket[:, start:end]
            pair_mask_chunk = edge_mask[:, :, start:end]
            chunk_inputs = (
                atom_query,
                atom_value,
                fragment_atom_mask,
                fragment,
                residue_chunk,
                residue_mask_chunk,
                pocket_chunk,
                pair_mask_chunk,
                batch["pocket_context6"][:, start:end],
            )
            if self.training and torch.is_grad_enabled():
                chunk_outputs = checkpoint(
                    self._forward_chunk,
                    *chunk_inputs,
                    use_reentrant=False,
                )
            else:
                chunk_outputs = self._forward_chunk(*chunk_inputs)
            (
                atom_numerator_chunk,
                atom_denominator_chunk,
                residue_numerator_chunk,
                residue_denominator_chunk,
                edge,
                gate,
                entropy,
            ) = chunk_outputs
            atom_numerator = atom_numerator + atom_numerator_chunk
            atom_denominator = atom_denominator + atom_denominator_chunk
            residue_numerator_chunks.append(residue_numerator_chunk)
            residue_denominator_chunks.append(residue_denominator_chunk)
            interaction_entropy_chunks.append(entropy)
            edge_chunks.append(edge)
            gate_chunks.append(gate)

        # Normalize message content once, then apply total edge activity once in
        # the residual updater.  Using (1 + sum_gate) here as well would square
        # the attenuation and make sparse AR evidence almost disappear.
        atom_message = atom_numerator / atom_denominator[..., None].clamp_min(1e-8)
        residue_numerator = torch.cat(residue_numerator_chunks, dim=1)
        residue_denominator = torch.cat(residue_denominator_chunks, dim=1)
        residue_message = residue_numerator / residue_denominator[..., None].clamp_min(
            1e-8
        )
        atom_packed_message, atom_valid = scatter_padded_mean(
            atom_message,
            fragment_indices,
            fragment_atom_mask,
            atom_work.size(0),
        )
        residue_packed_message, residue_valid = scatter_padded_mean(
            residue_message,
            pocket_indices,
            pocket_residue_mask,
            residue_work.size(0),
        )
        atom_activity_padded = atom_denominator / (1.0 + atom_denominator)
        atom_activity, _ = scatter_padded_mean(
            atom_activity_padded[..., None],
            fragment_indices,
            fragment_atom_mask,
            atom_work.size(0),
        )
        residue_activity_padded = residue_denominator / (
            1.0 + residue_denominator
        )
        residue_activity, _ = scatter_padded_mean(
            residue_activity_padded[..., None],
            pocket_indices,
            pocket_residue_mask,
            residue_work.size(0),
        )
        atom_work, atom_update_gate = self.atom_update(
            atom_work,
            atom_packed_message,
            atom_valid,
            atom_activity.squeeze(-1),
        )
        residue_work, residue_update_gate = self.residue_update(
            residue_work,
            residue_packed_message,
            residue_valid,
            residue_activity.squeeze(-1),
        )

        return {
            "atom_work": atom_work,
            "residue_work": residue_work,
            "edge_tokens": torch.cat(edge_chunks, dim=2),
            "edge_gate": torch.cat(gate_chunks, dim=2),
            "edge_mask": edge_mask,
            # Diagnostics only: detach so logging does not retain the AR graph.
            "interaction_entropy": torch.cat(
                interaction_entropy_chunks, dim=2
            ).detach(),
            "atom_update_gate": atom_update_gate,
            "residue_update_gate": residue_update_gate,
            "atom_update_valid": atom_valid,
            "residue_update_valid": residue_valid,
        }


class FragmentPocketStage(nn.Module):
    """One FP stage plus token-context broadcast for the next AR stage."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.node_builder = LocalNodeBuilder(hidden_dim, dropout)
        self.fragment_seed_update = GatedResidualUpdate(
            hidden_dim, dropout, gate_bias=-1.0
        )
        self.pocket_seed_update = GatedResidualUpdate(
            hidden_dim, dropout, gate_bias=-1.0
        )
        self.fragment_graph = DenseGraphResidual(hidden_dim, dropout)
        self.pocket_graph = PocketGraphResidual(hidden_dim, 3, dropout)
        self.edge_seed = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.edge_gate = nn.Linear(hidden_dim, 1)
        self.edge_to_fragment = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.edge_to_pocket = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.fragment_cross_update = GatedResidualUpdate(
            hidden_dim, dropout, gate_bias=-1.0
        )
        self.pocket_cross_update = GatedResidualUpdate(
            hidden_dim, dropout, gate_bias=-1.0
        )
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update_norm = nn.LayerNorm(hidden_dim)
        self.atom_context_update = GatedResidualUpdate(
            hidden_dim, dropout, gate_bias=-2.0
        )
        self.residue_context_update = GatedResidualUpdate(
            hidden_dim, dropout, gate_bias=-2.0
        )
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.constant_(self.edge_gate.bias, -1.0)

    def forward(
        self,
        atom_base,
        atom_work,
        residue_base,
        residue_work,
        previous_fragment,
        previous_pocket,
        ar_edge,
        edge_mask,
        batch,
    ):
        local = self.node_builder(
            atom_base, atom_work, residue_base, residue_work, batch
        )
        fragment_mask = batch["fragment_mask"]
        pocket_mask = batch["pocket_mask"]
        fragment, fragment_seed_gate = self.fragment_seed_update(
            previous_fragment, local["fragment"], fragment_mask
        )
        pocket, pocket_seed_gate = self.pocket_seed_update(
            previous_pocket, local["pocket"], pocket_mask
        )
        fragment, fragment_graph_gate = self.fragment_graph(
            fragment, batch["fragment_adjacency"], fragment_mask
        )
        pocket, pocket_edge_weight, pocket_graph_gate = self.pocket_graph(
            pocket, batch
        )

        fragment_expanded = fragment[:, :, None, :].expand(
            -1, -1, pocket.size(1), -1
        )
        pocket_expanded = pocket[:, None, :, :].expand(
            -1, fragment.size(1), -1, -1
        )
        edge = self.edge_norm(
            ar_edge
            + self.edge_seed(
                torch.cat(
                    [
                        ar_edge,
                        fragment_expanded,
                        pocket_expanded,
                        fragment_expanded * pocket_expanded,
                        torch.abs(fragment_expanded - pocket_expanded),
                    ],
                    dim=-1,
                )
            )
        )
        edge = edge * edge_mask[..., None].to(edge.dtype)
        gate = torch.sigmoid(self.edge_gate(edge).squeeze(-1))
        gate = gate * edge_mask.to(gate.dtype)

        fragment_gate_sum = gate.sum(2)
        pocket_gate_sum = gate.sum(1)
        fragment_message = (
            gate[..., None] * self.edge_to_fragment(edge)
        ).sum(2) / fragment_gate_sum[..., None].clamp_min(1e-8)
        pocket_message = (
            gate[..., None] * self.edge_to_pocket(edge)
        ).sum(1) / pocket_gate_sum[..., None].clamp_min(1e-8)
        fragment_activity = fragment_gate_sum / (1.0 + fragment_gate_sum)
        pocket_activity = pocket_gate_sum / (1.0 + pocket_gate_sum)
        fragment, fragment_cross_gate = self.fragment_cross_update(
            fragment,
            fragment_message,
            fragment_mask,
            fragment_activity,
        )
        pocket, pocket_cross_gate = self.pocket_cross_update(
            pocket,
            pocket_message,
            pocket_mask,
            pocket_activity,
        )

        fragment_expanded = fragment[:, :, None, :].expand_as(edge)
        pocket_expanded = pocket[:, None, :, :].expand_as(edge)
        edge = self.edge_update_norm(
            edge
            + self.edge_update(
                torch.cat([edge, fragment_expanded, pocket_expanded], dim=-1)
            )
        )
        edge = edge * edge_mask[..., None].to(edge.dtype)
        return {
            "fragment": fragment,
            "pocket": pocket,
            "edge": edge,
            "edge_gate": gate,
            "pocket_edge_weight": pocket_edge_weight,
            "fragment_atom_weight": local["fragment_atom_weight"],
            "pocket_residue_weight": local["pocket_residue_weight"],
            "fragment_seed_gate": fragment_seed_gate,
            "pocket_seed_gate": pocket_seed_gate,
            "fragment_graph_gate": fragment_graph_gate,
            "pocket_graph_gate": pocket_graph_gate,
            "fragment_cross_gate": fragment_cross_gate,
            "pocket_cross_gate": pocket_cross_gate,
        }

    def broadcast(
        self, atom_work, residue_work, fragment, pocket, edge_gate, batch
    ):
        fragment_indices = batch["fragment_atom_indices"].clamp_min(0)
        fragment_atom_mask = batch["fragment_atom_mask"]
        fragment_message = fragment[:, :, None, :].expand(
            -1, -1, fragment_indices.size(2), -1
        )
        atom_message, atom_valid = scatter_padded_mean(
            fragment_message,
            fragment_indices,
            fragment_atom_mask,
            atom_work.size(0),
        )
        fragment_activity = edge_gate.sum(2) / (1.0 + edge_gate.sum(2))
        atom_activity, _ = scatter_padded_mean(
            fragment_activity[:, :, None, None].expand(
                -1, -1, fragment_indices.size(2), 1
            ),
            fragment_indices,
            fragment_atom_mask,
            atom_work.size(0),
        )
        atom_work, atom_gate = self.atom_context_update(
            atom_work,
            atom_message,
            atom_valid,
            atom_activity.squeeze(-1),
        )

        pocket_indices = batch["pocket_residue_indices"].clamp_min(0)
        pocket_residue_mask = batch["pocket_residue_mask"]
        pocket_message = pocket[:, :, None, :].expand(
            -1, -1, pocket_indices.size(2), -1
        )
        residue_message, residue_valid = scatter_padded_mean(
            pocket_message,
            pocket_indices,
            pocket_residue_mask,
            residue_work.size(0),
        )
        pocket_activity = edge_gate.sum(1) / (1.0 + edge_gate.sum(1))
        residue_activity, _ = scatter_padded_mean(
            pocket_activity[:, :, None, None].expand(
                -1, -1, pocket_indices.size(2), 1
            ),
            pocket_indices,
            pocket_residue_mask,
            residue_work.size(0),
        )
        residue_work, residue_gate = self.residue_context_update(
            residue_work,
            residue_message,
            residue_valid,
            residue_activity.squeeze(-1),
        )
        return atom_work, residue_work, atom_gate, residue_gate


class GlobalPairInteraction(nn.Module):
    """Symmetric whole-drug--whole-protein interaction without local leakage."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        pair_dim = hidden_dim * 4
        self.drug_message = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.protein_message = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.drug_update = GatedResidualUpdate(hidden_dim, dropout, -1.0)
        self.protein_update = GatedResidualUpdate(hidden_dim, dropout, -1.0)
        self.evidence = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def pair_features(drug, protein):
        return torch.cat(
            [drug, protein, drug * protein, torch.abs(drug - protein)], dim=-1
        )

    def forward(self, drug, protein):
        pair = self.pair_features(drug, protein)
        drug, drug_gate = self.drug_update(drug, self.drug_message(pair))
        protein, protein_gate = self.protein_update(
            protein, self.protein_message(pair)
        )
        evidence = self.evidence(self.pair_features(drug, protein))
        return drug, protein, evidence, drug_gate, protein_gate


class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.net[-1].bias, 5.0)

    def forward(self, x):
        return self.net(x)


class ThreeGranularityCaviarDTA(nn.Module):
    def __init__(
        self,
        drug_1d_in_dim=768,
        atom_v2_dim=52,
        protein_1d_in_dim=1280,
        protein_node_s_dim=6,
        hidden_dim=128,
        dropout=0.1,
        interaction_rounds=2,
        ar_pocket_chunk_size=4,
        contrast_projection_dim=128,
        **deprecated_kwargs,
    ):
        super().__init__()
        if int(interaction_rounds) != 2:
            raise ValueError("v3 currently requires interaction_rounds=2")
        self.hidden_dim = int(hidden_dim)
        self.interaction_rounds = int(interaction_rounds)
        self.deprecated_kwargs = dict(deprecated_kwargs)

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

        # Initial AR seeds are pure fragment/pocket pools.  Pocket-residue graph
        # propagation is performed exactly once inside each of the two FP stages.
        self.initial_nodes = LocalNodeBuilder(
            hidden_dim, dropout, pocket_message_passing=False
        )
        self.ar_blocks = nn.ModuleList(
            [
                AtomResidueInteractionBlock(
                    hidden_dim, dropout, ar_pocket_chunk_size
                )
                for _ in range(self.interaction_rounds)
            ]
        )
        self.fp_stages = nn.ModuleList(
            [
                FragmentPocketStage(hidden_dim, dropout)
                for _ in range(self.interaction_rounds)
            ]
        )
        self.global_interaction = GlobalPairInteraction(hidden_dim, dropout)

        self.fragment_pool = MaskedAttentionPool(hidden_dim)
        self.pocket_pool = MaskedAttentionPool(hidden_dim)
        self.ar_edge_pool = GatedMaskedPool(hidden_dim)
        self.fp_edge_pool = GatedMaskedPool(hidden_dim)
        self.ar_evidence_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fp_evidence_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.final_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.ar_head = PredictionHead(hidden_dim, hidden_dim // 2, dropout)
        self.fp_head = PredictionHead(hidden_dim, hidden_dim // 2, dropout)
        self.global_head = PredictionHead(hidden_dim, hidden_dim // 2, dropout)
        self.main_head = PredictionHead(hidden_dim, hidden_dim, dropout)
        self.contrast_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, contrast_projection_dim),
                )
                for _ in range(3)
            ]
        )

    def encode_inputs(self, batch):
        drug_1d = self.drug_1d_encoder(batch["drug_1d"])
        drug_out = self.drug_atom_encoder(batch["drug_atom_v2"], return_node=True)
        drug_global = self.drug_fusion([drug_1d, drug_out["graph_feat"]])

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
        residue_base = self.residue_norm(
            protein_out["node_feat"] + self.residue_aux_proj(residue_aux)
        )
        protein_local_global = global_mean_pool(
            residue_base, protein_out["batch"]
        )
        protein_global = self.protein_fusion(
            [protein_1d, protein_out["graph_feat"] + protein_local_global]
        )
        return (
            drug_global,
            protein_global,
            drug_out["node_feat"],
            residue_base,
        )

    def forward(self, batch, return_details=False, compute_contrastive=False):
        (
            drug_global,
            protein_global,
            atom_base,
            residue_base,
        ) = self.encode_inputs(batch)
        atom_work = atom_base
        residue_work = residue_base

        initial = self.initial_nodes(
            atom_base, atom_work, residue_base, residue_work, batch
        )
        fragment = initial["fragment"]
        pocket = initial["pocket"]
        broadcast_atom_gate = None
        broadcast_residue_gate = None
        final_ar = None
        final_fp = None
        for round_index, (ar_block, fp_stage) in enumerate(
            zip(self.ar_blocks, self.fp_stages)
        ):
            ar = ar_block(
                atom_base,
                atom_work,
                residue_base,
                residue_work,
                fragment,
                pocket,
                batch,
            )
            atom_work = ar["atom_work"]
            residue_work = ar["residue_work"]
            fp = fp_stage(
                atom_base,
                atom_work,
                residue_base,
                residue_work,
                fragment,
                pocket,
                ar["edge_tokens"],
                ar["edge_mask"],
                batch,
            )
            fragment, pocket = fp["fragment"], fp["pocket"]
            final_ar, final_fp = ar, fp
            if round_index + 1 < self.interaction_rounds:
                (
                    atom_work,
                    residue_work,
                    broadcast_atom_gate,
                    broadcast_residue_gate,
                ) = fp_stage.broadcast(
                    atom_work,
                    residue_work,
                    fragment,
                    pocket,
                    fp["edge_gate"],
                    batch,
                )

        assert final_ar is not None and final_fp is not None
        fragment_evidence, fragment_weight = self.fragment_pool(
            fragment, batch["fragment_mask"]
        )
        pocket_evidence, pocket_weight = self.pocket_pool(
            pocket, batch["pocket_mask"]
        )
        edge_mask_flat = final_ar["edge_mask"].flatten(1, 2)
        ar_edge_flat = final_ar["edge_tokens"].flatten(1, 2)
        ar_gate_flat = final_ar["edge_gate"].flatten(1, 2)
        fp_edge_flat = final_fp["edge"].flatten(1, 2)
        fp_gate_flat = final_fp["edge_gate"].flatten(1, 2)
        ar_edge_evidence, ar_edge_weight = self.ar_edge_pool(
            ar_edge_flat, edge_mask_flat, ar_gate_flat
        )
        fp_edge_evidence, fp_edge_weight = self.fp_edge_pool(
            fp_edge_flat, edge_mask_flat, fp_gate_flat
        )
        ar_evidence = self.ar_evidence_proj(ar_edge_evidence)
        fp_evidence = self.fp_evidence_proj(
            torch.cat(
                [fragment_evidence, pocket_evidence, fp_edge_evidence], dim=-1
            )
        )
        (
            drug_global_updated,
            protein_global_updated,
            global_evidence,
            global_drug_gate,
            global_protein_gate,
        ) = self.global_interaction(drug_global, protein_global)

        final_latent = self.final_trunk(
            torch.cat(
                [
                    ar_evidence,
                    fp_evidence,
                    global_evidence,
                    ar_evidence * fp_evidence,
                    ar_evidence * global_evidence,
                    fp_evidence * global_evidence,
                ],
                dim=-1,
            )
        )
        ar_prediction = self.ar_head(ar_evidence)
        fp_prediction = self.fp_head(fp_evidence)
        global_prediction = self.global_head(global_evidence)
        affinity_mean = self.main_head(final_latent)

        if not return_details:
            return affinity_mean
        if compute_contrastive:
            contrast_embeddings = torch.stack(
                [
                    self.contrast_projectors[0](ar_evidence),
                    self.contrast_projectors[1](fp_evidence),
                    self.contrast_projectors[2](global_evidence),
                ],
                dim=1,
            )
        else:
            # No projection-matrix work or unused projection graph when the
            # default contrastive weight is zero.
            contrast_embeddings = torch.stack(
                [ar_evidence, fp_evidence, global_evidence], dim=1
            ).detach()
        return {
            "pred": affinity_mean,
            "affinity_mean": affinity_mean,
            "main_prediction": affinity_mean,
            "ar_prediction": ar_prediction,
            "fp_prediction": fp_prediction,
            "global_prediction": global_prediction,
            "ar_embedding": ar_evidence,
            "fp_embedding": fp_evidence,
            "global_embedding": global_evidence,
            "contrast_embeddings": contrast_embeddings,
            "fragment_tokens": fragment,
            "pocket_tokens": pocket,
            "ar_edge_tokens": final_ar["edge_tokens"],
            "fp_edge_tokens": final_fp["edge"],
            "edge_mask": final_ar["edge_mask"],
            "ar_edge_gate": final_ar["edge_gate"],
            "fp_edge_gate": final_fp["edge_gate"],
            "edge_confidence": final_fp["edge_gate"],
            "pocket_edge_weight": final_fp["pocket_edge_weight"],
            "pocket_edge_mask": batch["pocket_adjacency"].bool(),
            "interaction_entropy": final_ar["interaction_entropy"],
            "atom_update_gate": final_ar["atom_update_gate"],
            "residue_update_gate": final_ar["residue_update_gate"],
            "atom_update_valid": final_ar["atom_update_valid"],
            "residue_update_valid": final_ar["residue_update_valid"],
            "broadcast_atom_gate": broadcast_atom_gate,
            "broadcast_residue_gate": broadcast_residue_gate,
            "global_drug_gate": global_drug_gate,
            "global_protein_gate": global_protein_gate,
            "fragment_atom_weight": final_fp["fragment_atom_weight"],
            "subpocket_residue_weight": final_fp["pocket_residue_weight"],
            "fragment_weight": fragment_weight,
            "pocket_weight": pocket_weight,
            "ar_edge_weight": ar_edge_weight,
            "fp_edge_weight": fp_edge_weight,
            "drug_global_updated": drug_global_updated,
            "protein_global_updated": protein_global_updated,
        }
