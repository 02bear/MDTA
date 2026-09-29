#!/usr/bin/env python3
"""Clean BRICS chemistry-graph view without atom-to-fragment writeback."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_brics_hierarchical_stage1 as base
from train_brics_chem_stage1 import BRICSChemHierarchicalP13D, ChemStore


class BRICSChemGraphP13D(BRICSChemHierarchicalP13D):
    def __init__(self, project, checkpoint, hidden=128, dropout=0.1, mask_rate=0.15):
        super().__init__(project, checkpoint, hidden, dropout, mask_rate)
        self.fragment_input = nn.Sequential(
            nn.Linear(282, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )
        self.graph_gate = nn.Linear(256, 1)
        self.delta_3d = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
        )
        nn.init.zeros_(self.delta_3d[-1].weight)
        nn.init.zeros_(self.delta_3d[-1].bias)

    def forward(self, batch, use_mask=False, return_debug=False):
        chemistry = batch["fragment_chem_feat"]
        fragment_count = batch["fragment_batch"].numel()
        fragment = self.fragment_input(chemistry)
        masked = torch.zeros(fragment_count, dtype=torch.bool, device=fragment.device)
        if use_mask and fragment_count:
            masked = torch.rand(fragment_count, device=fragment.device) < self.mask_rate
            if not masked.any():
                masked[torch.randint(fragment_count, (1,), device=fragment.device)] = True
            fragment = torch.where(masked[:, None], self.mask_token[None, :], fragment)
        for block in self.fragment_blocks:
            fragment = block(fragment, batch["fragment_edge_index"], batch["fragment_edge_attr"])

        fragment_global = global_mean_pool(fragment, batch["fragment_batch"])
        baseline_3d = batch["drug_3d_graph_feat"]
        gate = torch.sigmoid(self.graph_gate(torch.cat([baseline_3d, fragment_global], dim=-1)))
        raw_delta = self.delta_3d(torch.cat(
            [baseline_3d, fragment_global, baseline_3d * fragment_global], dim=-1
        ))
        delta = gate * raw_delta
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
    base.BRICSHierarchicalP13D = BRICSChemGraphP13D
    base.main()
