"""Typed atom-residue predictor with a capacity-controlled shared encoder."""

from __future__ import annotations

import torch
from torch import nn

from chemical_masks import INTERACTION_TYPES
from rich_ligand_features import ATOM_DIM, EDGE_DIM


class LigandMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node, edge_index, edge_attr):
        src, dst = edge_index
        msg = self.message(torch.cat([node[src], edge_attr], dim=-1))
        aggregate = node.new_zeros(node.shape)
        aggregate.index_add_(0, dst, msg)
        degree = node.new_zeros((node.shape[0], 1))
        degree.index_add_(0, dst, node.new_ones((dst.shape[0], 1)))
        aggregate = aggregate / degree.clamp_min(1.0)
        return self.norm(node + self.update(torch.cat([node, aggregate], dim=-1)))


class RichLigandEncoder(nn.Module):
    def __init__(self, hidden_dim=128, layers=2):
        super().__init__()
        self.atom_projection = nn.Sequential(
            nn.Linear(ATOM_DIM, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.layers = nn.ModuleList(
            LigandMessageLayer(hidden_dim, EDGE_DIM) for _ in range(layers)
        )

    def forward_nodes(self, graph):
        node = self.atom_projection(graph.x.float())
        for layer in self.layers:
            node = layer(node, graph.edge_index.long(), graph.edge_attr.float())
        return node


class TypedPairwiseContactPredictor(nn.Module):
    def __init__(self, protein_dim=1024, hidden_dim=128, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.ligand_encoder = RichLigandEncoder(hidden_dim=hidden_dim)
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.atom_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.base_pair_bias = nn.Parameter(torch.zeros(1))
        self.residue_bias = nn.Parameter(torch.tensor(-3.0))
        self.type_scale = nn.Parameter(torch.ones(len(INTERACTION_TYPES), hidden_dim))
        self.type_bias = nn.Parameter(torch.zeros(len(INTERACTION_TYPES)))
        self.scale = hidden_dim**-0.5

    def forward(self, protein_graph, ligand_graph):
        residues = self.protein_projection(protein_graph.x.float())
        atoms = self.atom_projection(self.ligand_encoder.forward_nodes(ligand_graph))
        base_pair_logits = residues @ atoms.transpose(0, 1) * self.scale + self.base_pair_bias
        typed_pair_logits = torch.einsum(
            "rh,th,ah->tra", residues, self.type_scale, atoms
        ) * self.scale + self.type_bias[:, None, None]
        tau = self.temperature
        residue_logits = tau * torch.logsumexp(base_pair_logits / tau, dim=1)
        residue_logits -= tau * torch.log(
            residue_logits.new_tensor(float(base_pair_logits.shape[1]))
        )
        residue_logits += self.residue_bias
        return residue_logits, base_pair_logits, typed_pair_logits


def initialize_from_binary(model: TypedPairwiseContactPredictor, checkpoint):
    """Warm-start all compatible parameters; append rich atom features at zero weight."""
    old = torch.load(checkpoint, map_location="cpu", weights_only=True)
    new = model.state_dict()
    copied, expanded = [], []
    for key, value in old.items():
        destination = key
        if key == "pair_bias":
            destination = "base_pair_bias"
        if destination not in new:
            continue
        value = value.to(new[destination].device)
        if new[destination].shape == value.shape:
            new[destination] = value.clone()
            copied.append(destination)
        elif destination == "ligand_encoder.atom_projection.0.weight" and value.shape[1] < new[destination].shape[1]:
            expanded_weight = new[destination].clone()
            expanded_weight.zero_()
            # Cached PDBbind graphs had aromaticity cleared, so old column 81
            # never carried trained signal.  Do not copy that random column.
            expanded_weight[:, :81] = value[:, :81]
            new[destination] = expanded_weight
            expanded.append(destination)
    old_bias = old.get("pair_bias", torch.zeros(1)).reshape(-1)[0].to(new["type_bias"].device)
    new["type_bias"] = torch.full_like(new["type_bias"], old_bias)
    new["type_scale"] = torch.ones_like(new["type_scale"])
    model.load_state_dict(new)
    return {"copied": sorted(copied), "expanded": sorted(expanded)}
