#!/usr/bin/env python3
"""BRICS hierarchy v2 with explicit fragment chemistry and masked-chemistry recovery."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_brics_hierarchical_stage1 as base


class ChemStore(base.Store):
    def collate(self, rows):
        output = super().collate(rows)
        output["fragment_chem_feat"] = torch.cat([
            self.drugs[self.drug_ids[row]]["fragment_chem_feat"].float()
            for row in rows
        ])
        if output["fragment_chem_feat"].shape[0] != output["fragment_batch"].numel():
            raise ValueError("fragment chemistry count does not match fragment graph")
        return output


class BRICSChemHierarchicalP13D(base.BRICSHierarchicalP13D):
    def __init__(self, project, checkpoint, hidden=128, dropout=0.1, mask_rate=0.15):
        super().__init__(project, checkpoint, hidden, dropout, mask_rate)
        self.fragment_input = nn.Sequential(
            nn.Linear(256 + 282, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )
        self.fragment_reconstruction = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, 282)
        )

    def forward(self, batch, use_mask=False, return_debug=False):
        atoms = batch["atom_node_feat"]
        assignment = batch["atom_to_fragment"]
        fragment_count = batch["fragment_batch"].numel()
        fragment_mean = scatter(atoms, assignment, dim=0, dim_size=fragment_count, reduce="mean")
        fragment_max = scatter(atoms, assignment, dim=0, dim_size=fragment_count, reduce="max")
        atom_summary = torch.cat([fragment_mean, fragment_max], dim=-1)
        chemistry = batch["fragment_chem_feat"]
        fragment = self.fragment_input(torch.cat([atom_summary, chemistry], dim=-1))

        masked = torch.zeros(fragment_count, dtype=torch.bool, device=fragment.device)
        if use_mask and fragment_count:
            masked = torch.rand(fragment_count, device=fragment.device) < self.mask_rate
            if not masked.any():
                masked[torch.randint(fragment_count, (1,), device=fragment.device)] = True
            fragment = torch.where(masked[:, None], self.mask_token[None, :], fragment)
        for block in self.fragment_blocks:
            fragment = block(fragment, batch["fragment_edge_index"], batch["fragment_edge_attr"])

        fragment_global = global_mean_pool(fragment, batch["fragment_batch"])
        broadcast = self.broadcast(fragment[assignment])
        gate = torch.sigmoid(self.atom_gate(torch.cat([atoms, broadcast], dim=-1)))
        updated_atoms = atoms + gate * broadcast
        updated_graph = global_mean_pool(updated_atoms, batch["atom_batch"])
        baseline_3d = batch["drug_3d_graph_feat"]
        delta = self.delta_3d(torch.cat(
            [baseline_3d, updated_graph, updated_graph - baseline_3d], dim=-1
        ))
        drug_3d = baseline_3d + delta
        drug_fused = self.drug_fusion([batch["drug_1d_feat"], drug_3d])
        pair_feature = torch.cat([drug_fused, batch["pair_feature"][:, 128:]], dim=-1)
        prediction = self.decoder(pair_feature).view(-1)

        align = 1.0 - F.cosine_similarity(
            self.fragment_alignment(fragment_global), baseline_3d.detach(), dim=-1
        ).mean()
        if masked.any():
            reconstructed = self.fragment_reconstruction(fragment[masked])
            target = chemistry[masked].detach()
            reconstruction = (
                F.binary_cross_entropy_with_logits(reconstructed[:, :272], target[:, :272])
                + 0.2 * F.mse_loss(reconstructed[:, 272:], target[:, 272:])
            )
        else:
            reconstruction = prediction.new_zeros(())
        debug = {
            "delta": delta,
            "gate": gate,
            "alignment_loss": align,
            "reconstruction_loss": reconstruction,
        }
        return (prediction, debug) if return_debug else prediction


if __name__ == "__main__":
    base.Store = ChemStore
    base.BRICSHierarchicalP13D = BRICSChemHierarchicalP13D
    base.main()
