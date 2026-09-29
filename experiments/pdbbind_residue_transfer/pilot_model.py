"""Small pose-free ligand-conditioned residue contact predictor."""

from __future__ import annotations

import torch
from torch import nn


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
        agg = node.new_zeros(node.shape)
        agg.index_add_(0, dst, msg)
        degree = node.new_zeros((node.shape[0], 1))
        degree.index_add_(0, dst, node.new_ones((dst.shape[0], 1)))
        agg = agg / degree.clamp_min(1.0)
        return self.norm(node + self.update(torch.cat([node, agg], dim=-1)))


class LigandEncoder(nn.Module):
    def __init__(self, atom_dim=82, edge_dim=6, hidden_dim=128, layers=2):
        super().__init__()
        self.atom_projection = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.layers = nn.ModuleList(
            LigandMessageLayer(hidden_dim, edge_dim) for _ in range(layers)
        )

    def forward_nodes(self, graph):
        node = self.atom_projection(graph.x.float())
        edge_attr = graph.edge_attr.float()
        for layer in self.layers:
            node = layer(node, graph.edge_index.long(), edge_attr)
        return node

    def forward(self, graph):
        return self.forward_nodes(graph).mean(dim=0)


class ResidueContactPredictor(nn.Module):
    def __init__(self, protein_dim=1024, hidden_dim=128, protein_only=False):
        super().__init__()
        self.protein_only = protein_only
        self.ligand_encoder = LigandEncoder(hidden_dim=hidden_dim)
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.ligand_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.protein_only_context = nn.Parameter(torch.zeros(hidden_dim))
        self.bias = nn.Parameter(torch.tensor(-3.0))
        self.scale = hidden_dim**-0.5

    def forward(self, protein_graph, ligand_graph):
        residues = self.protein_projection(protein_graph.x.float())
        if self.protein_only:
            ligand = self.protein_only_context
        else:
            ligand = self.ligand_projection(self.ligand_encoder(ligand_graph))
        return (residues * ligand.unsqueeze(0)).sum(dim=-1) * self.scale + self.bias


class PairwiseResidueContactPredictor(nn.Module):
    """Atom-residue pair scorer with residue labels as weak supervision."""

    def __init__(
        self,
        protein_dim=1024,
        hidden_dim=128,
        protein_only=False,
        temperature=0.5,
        atom_dropout=0.1,
    ):
        super().__init__()
        self.protein_only = protein_only
        self.temperature = temperature
        self.atom_dropout = atom_dropout
        self.ligand_encoder = LigandEncoder(hidden_dim=hidden_dim)
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.atom_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.protein_only_atom = nn.Parameter(torch.zeros(1, hidden_dim))
        self.pair_bias = nn.Parameter(torch.zeros(1))
        self.residue_bias = nn.Parameter(torch.tensor(-3.0))
        self.scale = hidden_dim**-0.5

    def forward(self, protein_graph, ligand_graph, return_pair_scores=False):
        residues = self.protein_projection(protein_graph.x.float())
        if self.protein_only:
            atoms = self.protein_only_atom
        else:
            atoms = self.atom_projection(self.ligand_encoder.forward_nodes(ligand_graph))
        pair_scores = residues @ atoms.transpose(0, 1) * self.scale + self.pair_bias
        if self.training and not self.protein_only and self.atom_dropout > 0:
            keep = torch.rand(atoms.shape[0], device=atoms.device) >= self.atom_dropout
            if not keep.any():
                keep[torch.randint(atoms.shape[0], (1,), device=atoms.device)] = True
            aggregation_scores = pair_scores[:, keep]
        else:
            aggregation_scores = pair_scores
        tau = self.temperature
        residue_logits = tau * torch.logsumexp(aggregation_scores / tau, dim=1)
        residue_logits = residue_logits - tau * torch.log(
            residue_logits.new_tensor(float(aggregation_scores.shape[1]))
        )
        residue_logits = residue_logits + self.residue_bias
        if return_pair_scores:
            return residue_logits, pair_scores
        return residue_logits
