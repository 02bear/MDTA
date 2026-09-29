from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.model_p13d import MyModelMDTAP13D


def parse_active_scales(value):
    scales = {
        item.strip().lower()
        for item in (value.split(",") if isinstance(value, str) else value)
        if item.strip()
    }
    allowed = {"global", "fp", "ar"}
    if not scales <= allowed:
        raise ValueError(f"unknown scales: {sorted(scales - allowed)}")
    if "global" not in scales:
        raise ValueError("global must always be active")
    return scales


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax over valid entries, returning all-zero weights for empty rows."""
    mask = mask.to(torch.bool)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=dim) * mask.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).eps
    )


class MyModelMDTAP13DTriScale(nn.Module):
    def __init__(
        self,
        drug_1d_in_dim=768,
        drug_3d_node_in_dim=10,
        protein_1d_in_dim=1280,
        protein_3d_node_s_dim=6,
        protein_3d_node_v_dim=3,
        hidden_dim=128,
        dropout=0.1,
        task="regression",
        active_scales="global",
        cross_scale_mode="none",
        local_hidden_dim=128,
        local_dropout=0.1,
        max_fragments=12,
        max_atoms_per_fragment=24,
        max_pockets=3,
        max_residues_per_pocket=32,
        ar_region_chunk_size=32,
        top_k_fp_for_ar: Optional[int] = None,
    ):
        super().__init__()
        self.active_scales = parse_active_scales(active_scales)
        self.cross_scale_mode = cross_scale_mode
        if cross_scale_mode not in {"none", "bottom_up"}:
            raise ValueError("cross_scale_mode must be none or bottom_up")
        if cross_scale_mode == "bottom_up" and not {"fp", "ar"} <= self.active_scales:
            raise ValueError("bottom_up requires both fp and ar")
        if ar_region_chunk_size <= 0:
            raise ValueError("ar_region_chunk_size must be positive")
        if top_k_fp_for_ar is not None and top_k_fp_for_ar <= 0:
            raise ValueError("top_k_fp_for_ar must be positive or None")

        self.global_model = MyModelMDTAP13D(
            drug_1d_in_dim,
            drug_3d_node_in_dim,
            protein_1d_in_dim,
            protein_3d_node_s_dim,
            protein_3d_node_v_dim,
            hidden_dim,
            dropout,
            task,
        )
        h = local_hidden_dim
        self.hidden_dim = hidden_dim
        self.max_fragments = max_fragments
        self.max_atoms_per_fragment = max_atoms_per_fragment
        self.max_pockets = max_pockets
        self.max_residues_per_pocket = max_residues_per_pocket
        self.ar_region_chunk_size = ar_region_chunk_size
        self.top_k_fp_for_ar = top_k_fp_for_ar

        self.atom_proj = nn.Linear(hidden_dim, h)
        self.res1_proj = nn.Linear(protein_1d_in_dim, h)
        self.res3_proj = nn.Linear(hidden_dim, h)
        self.residue_fusion = nn.Sequential(
            nn.Linear(2 * h, h), nn.SiLU(), nn.Dropout(local_dropout)
        )
        self.residue_norm = nn.LayerNorm(h)
        self.fragment_projection = nn.Linear(h, h)
        self.pocket_projection = nn.Linear(h, h)

        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(4 * h, h),
                nn.SiLU(),
                nn.Dropout(local_dropout),
                nn.Linear(h, out_dim),
            )

        self.fp_feature_mlp = mlp(h)
        self.fp_score_mlp = mlp(1)
        self.ar_feature_mlp = mlp(h)
        self.ar_score_mlp = mlp(1)
        self.ar_pair_score = nn.Linear(h, 1)
        self.fp_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + h, h),
            nn.SiLU(),
            nn.Dropout(local_dropout),
            nn.Linear(h, 1),
        )
        self.ar_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + h, h),
            nn.SiLU(),
            nn.Dropout(local_dropout),
            nn.Linear(h, 1),
        )
        self.multiscale_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2 * h, h),
            nn.SiLU(),
            nn.Dropout(local_dropout),
            nn.Linear(h, 1),
        )
        self.fp_alpha = nn.Parameter(torch.tensor(0.0))
        self.ar_alpha = nn.Parameter(torch.tensor(0.0))
        self.multiscale_alpha = nn.Parameter(torch.tensor(0.0))
        self.ar_to_fp = nn.Linear(h, h)
        self.fp_to_global = nn.Linear(h, hidden_dim * 2)
        self.ar_to_fp_gate = nn.Parameter(torch.tensor(-2.0))
        self.fp_to_global_gate = nn.Parameter(torch.tensor(-2.0))
        self.fp_norm = nn.LayerNorm(h)
        self.global_norm = nn.LayerNorm(hidden_dim * 2)
        self.last_stats = {}

    @staticmethod
    def pair_input(x, y):
        return torch.cat([x, y, x * y, torch.abs(x - y)], dim=-1)

    def _encode_global_and_nodes(self, batch, need_nodes):
        """Run each 3D encoder exactly once and reuse graph/node outputs."""
        model = self.global_model
        drug_1d = model.drug_1d_encoder(batch["drug_1d"])
        device_type = batch["drug_3d"]["x"].device.type
        # EGNN uses in-place index_add/scatter reductions whose destination dtype
        # must match the source. Keep these reductions in FP32 under outer AMP.
        with torch.autocast(device_type=device_type, enabled=False):
            drug_3d = model.drug_3d_encoder(
                batch["drug_3d"], return_node=need_nodes
            )
        if need_nodes:
            drug_graph = drug_3d["graph_feat"]
            drug_nodes = drug_3d["node_feat"]
        else:
            drug_graph = drug_3d
        drug = model.drug_fusion([drug_1d, drug_graph])

        protein_1d = model.protein_1d_encoder(batch["protein_1d"])
        with torch.autocast(device_type=device_type, enabled=False):
            protein_3d = model.protein_3d_encoder(
                batch["protein_3d"], return_node=need_nodes
            )
        if need_nodes:
            protein_graph = protein_3d["graph_feat"]
            protein_nodes = protein_3d["node_feat"]
        else:
            protein_graph = protein_3d
            drug_nodes = protein_nodes = None
        protein = model.protein_fusion([protein_1d, protein_graph])
        global_repr = torch.cat([drug, protein], dim=-1)
        return model.decoder(global_repr), global_repr, drug_nodes, protein_nodes

    @staticmethod
    def _group_offsets(batch_index: torch.Tensor, batch_size: int):
        counts = torch.bincount(batch_index, minlength=batch_size)
        return counts.cumsum(0) - counts, counts

    @staticmethod
    def _gather_grouped(features, local_indices, requested_mask, batch_index):
        batch_size = local_indices.shape[0]
        offsets, counts = MyModelMDTAP13DTriScale._group_offsets(
            batch_index, batch_size
        )
        view_shape = (batch_size,) + (1,) * (local_indices.ndim - 1)
        valid = (
            requested_mask
            & (local_indices >= 0)
            & (local_indices < counts.view(view_shape))
        )
        global_indices = local_indices + offsets.view(view_shape)
        global_indices = global_indices.clamp(0, max(features.shape[0] - 1, 0))
        return features[global_indices], valid

    def _legacy_local_tensors(self, batch):
        """Compatibility path for old hand-built batches used by tests/tools."""
        device = batch["drug_3d"]["batch"].device
        batch_size = len(batch["drug_id"])
        f_max, a_max = self.max_fragments, self.max_atoms_per_fragment
        p_max, r_max = self.max_pockets, self.max_residues_per_pocket
        fragment_indices = torch.zeros(
            (batch_size, f_max, a_max), dtype=torch.long, device=device
        )
        fragment_atom_mask = torch.zeros_like(fragment_indices, dtype=torch.bool)
        fragment_mask = torch.zeros((batch_size, f_max), dtype=torch.bool, device=device)
        pocket_sequence = torch.zeros(
            (batch_size, p_max, r_max), dtype=torch.long, device=device
        )
        pocket_graph = torch.zeros_like(pocket_sequence)
        pocket_residue_mask = torch.zeros_like(pocket_sequence, dtype=torch.bool)
        pocket_mask = torch.zeros((batch_size, p_max), dtype=torch.bool, device=device)

        token_values = []
        token_offsets = []
        token_lengths = []
        running_offset = 0
        for sample_index in range(batch_size):
            tokens = batch["protein_1d_tokens"][sample_index].to(device)
            token_values.append(tokens)
            token_offsets.append(running_offset)
            token_lengths.append(len(tokens))
            running_offset += len(tokens)

            fragments = batch["drug_fragment_atom_indices"][sample_index]
            if batch["drug_fragment_valid"][sample_index]:
                fragments = sorted(
                    fragments,
                    key=lambda value: (
                        -len(value),
                        int(value.min()) if len(value) else 10**9,
                    ),
                )[:f_max]
                for fragment_index, atom_indices in enumerate(fragments):
                    atom_indices = atom_indices[:a_max].to(device)
                    count = len(atom_indices)
                    if count:
                        fragment_indices[sample_index, fragment_index, :count] = atom_indices
                        fragment_atom_mask[sample_index, fragment_index, :count] = True
                        fragment_mask[sample_index, fragment_index] = True

            pockets = batch["protein_pockets"][sample_index]
            if batch["protein_pocket_valid"][sample_index]:
                pockets = sorted(pockets, key=lambda value: -value["score"])[:p_max]
                for pocket_index, pocket in enumerate(pockets):
                    sequence_indices = pocket["sequence_indices"][:r_max].to(device)
                    graph_indices = pocket["protein_graph_node_indices"][:r_max].to(device)
                    count = min(len(sequence_indices), len(graph_indices))
                    if count:
                        pocket_sequence[sample_index, pocket_index, :count] = sequence_indices[:count]
                        pocket_graph[sample_index, pocket_index, :count] = graph_indices[:count]
                        pocket_residue_mask[sample_index, pocket_index, :count] = True
                        pocket_mask[sample_index, pocket_index] = True

        return {
            "fragment_atom_indices": fragment_indices,
            "fragment_atom_mask": fragment_atom_mask,
            "fragment_mask": fragment_mask,
            "pocket_sequence_indices": pocket_sequence,
            "pocket_graph_indices": pocket_graph,
            "pocket_residue_mask": pocket_residue_mask,
            "pocket_mask": pocket_mask,
            "protein_1d_token_values": torch.cat(token_values, dim=0),
            "protein_1d_token_offsets": torch.tensor(
                token_offsets, dtype=torch.long, device=device
            ),
            "protein_1d_token_lengths": torch.tensor(
                token_lengths, dtype=torch.long, device=device
            ),
        }

    def _get_local_tensors(self, batch):
        if "fragment_atom_indices" in batch:
            return batch
        return self._legacy_local_tensors(batch)

    def _select_ar_regions(self, fp_scores, region_mask):
        if self.top_k_fp_for_ar is None:
            return region_mask
        batch_size = fp_scores.shape[0]
        flat_scores = fp_scores.flatten(1).masked_fill(
            ~region_mask.flatten(1), torch.finfo(fp_scores.dtype).min
        )
        k = min(self.top_k_fp_for_ar, flat_scores.shape[1])
        top_indices = torch.topk(flat_scores, k=k, dim=1).indices
        selected = torch.zeros_like(flat_scores, dtype=torch.bool)
        selected.scatter_(1, top_indices, True)
        return selected.view_as(region_mask) & region_mask

    def _local_batched(self, atom_nodes, protein_nodes, batch):
        local = self._get_local_tensors(batch)
        fragment_indices = local["fragment_atom_indices"]
        fragment_atom_mask = local["fragment_atom_mask"]
        fragment_mask = local["fragment_mask"]
        pocket_sequence = local["pocket_sequence_indices"]
        pocket_graph = local["pocket_graph_indices"]
        pocket_residue_mask = local["pocket_residue_mask"]
        pocket_mask = local["pocket_mask"]
        batch_size, num_fragments, num_atoms = fragment_indices.shape
        num_pockets, num_residues = pocket_sequence.shape[1:]

        atom_values, valid_atoms = self._gather_grouped(
            atom_nodes,
            fragment_indices,
            fragment_atom_mask,
            batch["drug_3d"]["batch"],
        )
        fragment_mask = fragment_mask & valid_atoms.any(dim=-1)
        atom_repr = self.atom_proj(atom_values)
        fragment_mean = (
            atom_repr * valid_atoms.unsqueeze(-1).to(atom_repr.dtype)
        ).sum(dim=2) / valid_atoms.sum(dim=2, keepdim=True).clamp_min(1).to(
            atom_repr.dtype
        )
        fragment_repr = self.fragment_projection(fragment_mean)

        token_values = local["protein_1d_token_values"]
        token_offsets = local["protein_1d_token_offsets"]
        token_lengths = local["protein_1d_token_lengths"]
        token_valid = (
            pocket_residue_mask
            & (pocket_sequence >= 0)
            & (pocket_sequence < token_lengths[:, None, None])
        )
        token_indices = pocket_sequence + token_offsets[:, None, None]
        token_indices = token_indices.clamp(0, max(token_values.shape[0] - 1, 0))
        token_repr = token_values[token_indices]
        protein_3d_repr, graph_valid = self._gather_grouped(
            protein_nodes,
            pocket_graph,
            pocket_residue_mask,
            batch["protein_3d"]["batch"],
        )
        residue_mask = token_valid & graph_valid
        pocket_mask = pocket_mask & residue_mask.any(dim=-1)
        residue_repr = self.residue_norm(
            self.residue_fusion(
                torch.cat(
                    [self.res1_proj(token_repr), self.res3_proj(protein_3d_repr)],
                    dim=-1,
                )
            )
        )
        pocket_mean = (
            residue_repr * residue_mask.unsqueeze(-1).to(residue_repr.dtype)
        ).sum(dim=2) / residue_mask.sum(dim=2, keepdim=True).clamp_min(1).to(
            residue_repr.dtype
        )
        pocket_repr = self.pocket_projection(pocket_mean)

        fragment_grid = fragment_repr[:, :, None, :].expand(
            -1, -1, num_pockets, -1
        )
        pocket_grid = pocket_repr[:, None, :, :].expand(
            -1, num_fragments, -1, -1
        )
        fp_pair_input = self.pair_input(fragment_grid, pocket_grid)
        fp_features = self.fp_feature_mlp(fp_pair_input)
        fp_scores = self.fp_score_mlp(fp_pair_input).squeeze(-1)
        region_mask = fragment_mask[:, :, None] & pocket_mask[:, None, :]

        selected_ar_regions = self._select_ar_regions(fp_scores, region_mask)
        region_indices = selected_ar_regions.nonzero(as_tuple=False)
        ar_indices = []
        ar_features = []
        ar_scores = []
        for start in range(0, region_indices.shape[0], self.ar_region_chunk_size):
            index_chunk = region_indices[start : start + self.ar_region_chunk_size]
            sample_index, fragment_index, pocket_index = index_chunk.unbind(dim=1)
            atom_chunk = atom_repr[sample_index, fragment_index]
            atom_mask_chunk = valid_atoms[sample_index, fragment_index]
            residue_chunk = residue_repr[sample_index, pocket_index]
            residue_mask_chunk = residue_mask[sample_index, pocket_index]
            pair_mask = atom_mask_chunk[:, :, None] & residue_mask_chunk[:, None, :]
            ar_input = self.pair_input(
                atom_chunk[:, :, None, :].expand(-1, -1, num_residues, -1),
                residue_chunk[:, None, :, :].expand(-1, num_atoms, -1, -1),
            )
            flat_input = ar_input.flatten(0, 2)
            flat_features = self.ar_feature_mlp(flat_input).view(
                len(index_chunk), num_atoms * num_residues, -1
            )
            flat_scores = self.ar_score_mlp(flat_input).view(
                len(index_chunk), num_atoms * num_residues
            )
            flat_pair_mask = pair_mask.flatten(1)
            pair_weights = _masked_softmax(flat_scores, flat_pair_mask, dim=1)
            region_feature = (pair_weights.unsqueeze(-1) * flat_features).sum(dim=1)
            region_score = self.ar_pair_score(region_feature).squeeze(-1)
            flat_region_index = (
                (sample_index * num_fragments + fragment_index) * num_pockets
                + pocket_index
            )
            ar_indices.append(flat_region_index)
            ar_features.append(region_feature)
            ar_scores.append(region_score)
            del ar_input, flat_input, flat_features, flat_scores

        flat_region_count = batch_size * num_fragments * num_pockets
        ar_feature_dtype = ar_features[0].dtype if ar_features else fp_features.dtype
        ar_score_dtype = ar_scores[0].dtype if ar_scores else fp_scores.dtype
        ar_stack_flat = torch.zeros(
            (flat_region_count, self.fragment_projection.out_features),
            device=fp_features.device,
            dtype=ar_feature_dtype,
        )
        ar_pair_scores_flat = torch.full(
            (flat_region_count,),
            torch.finfo(ar_score_dtype).min,
            device=fp_scores.device,
            dtype=ar_score_dtype,
        )
        if ar_indices:
            all_indices = torch.cat(ar_indices)
            ar_stack_flat = ar_stack_flat.index_copy(
                0, all_indices, torch.cat(ar_features)
            )
            ar_pair_scores_flat = ar_pair_scores_flat.index_copy(
                0, all_indices, torch.cat(ar_scores)
            )
        ar_stack = ar_stack_flat.view(
            batch_size, num_fragments, num_pockets, -1
        )
        ar_pair_scores = ar_pair_scores_flat.view(
            batch_size, num_fragments, num_pockets
        )
        ar_weights = _masked_softmax(
            ar_pair_scores.flatten(1), selected_ar_regions.flatten(1), dim=1
        )
        ar_repr = (
            ar_weights.unsqueeze(-1) * ar_stack.flatten(1, 2)
        ).sum(dim=1)

        if self.cross_scale_mode == "bottom_up":
            fp_features = self.fp_norm(
                fp_features
                + torch.sigmoid(self.ar_to_fp_gate) * self.ar_to_fp(ar_stack)
            )
        fp_weights = _masked_softmax(
            fp_scores.flatten(1), region_mask.flatten(1), dim=1
        )
        fp_repr = (
            fp_weights.unsqueeze(-1) * fp_features.flatten(1, 2)
        ).sum(dim=1)
        local_valid_mask = region_mask.flatten(1).any(dim=1)

        fragment_count = fragment_mask.sum(dim=1)
        pocket_count = pocket_mask.sum(dim=1)
        stats = {
            "fragments_per_sample": fragment_count.to(atom_nodes.dtype).mean(),
            "pockets_per_sample": pocket_count.to(atom_nodes.dtype).mean(),
            "fp_pairs_per_sample": region_mask.sum(dim=(1, 2)).to(atom_nodes.dtype).mean(),
            "ar_pairs_per_sample": selected_ar_regions.sum(dim=(1, 2)).to(atom_nodes.dtype).mean(),
            "local_missing_rate": (~local_valid_mask).to(atom_nodes.dtype).mean(),
            "sample_count": torch.tensor(batch_size, device=atom_nodes.device),
            "local_missing_count": (~local_valid_mask).sum(),
            "fp_pairs_total": region_mask.sum(),
            "ar_pairs_total": selected_ar_regions.sum(),
        }
        return fp_repr, ar_repr, local_valid_mask, stats

    def _local_one_reference(self, sample_index, atom_nodes, protein_nodes, batch):
        """Original loop implementation retained only for numerical alignment tests."""
        device = atom_nodes.device
        zero = atom_nodes.new_zeros(self.fragment_projection.out_features)
        fragments = (
            batch["drug_fragment_atom_indices"][sample_index]
            if batch["drug_fragment_valid"][sample_index]
            else []
        )
        pockets = (
            batch["protein_pockets"][sample_index]
            if batch["protein_pocket_valid"][sample_index]
            else []
        )
        fragments = sorted(
            fragments,
            key=lambda value: (
                -len(value),
                int(value.min()) if len(value) else 10**9,
            ),
        )[: self.max_fragments]
        pockets = sorted(pockets, key=lambda value: -value["score"])[
            : self.max_pockets
        ]
        atom_batch = batch["drug_3d"]["batch"]
        protein_batch = batch["protein_3d"]["batch"]
        atoms = atom_nodes[atom_batch == sample_index]
        protein_3d = protein_nodes[protein_batch == sample_index]
        protein_1d = batch["protein_1d_tokens"][sample_index]
        fp_features, fp_scores, ar_features, ar_pair_scores = [], [], [], []
        for fragment in fragments:
            fragment = fragment[: self.max_atoms_per_fragment].to(device)
            fragment = fragment[fragment < len(atoms)]
            if not len(fragment):
                continue
            atom_repr = self.atom_proj(atoms[fragment])
            fragment_repr = self.fragment_projection(atom_repr.mean(0))
            for pocket in pockets:
                sequence_index = pocket["sequence_indices"][
                    : self.max_residues_per_pocket
                ].to(device)
                graph_index = pocket["protein_graph_node_indices"][
                    : self.max_residues_per_pocket
                ].to(device)
                keep = (sequence_index < len(protein_1d)) & (
                    graph_index < len(protein_3d)
                )
                sequence_index, graph_index = sequence_index[keep], graph_index[keep]
                if not len(sequence_index):
                    continue
                residue_repr = self.residue_norm(
                    self.residue_fusion(
                        torch.cat(
                            [
                                self.res1_proj(protein_1d[sequence_index]),
                                self.res3_proj(protein_3d[graph_index]),
                            ],
                            dim=-1,
                        )
                    )
                )
                pocket_repr = self.pocket_projection(residue_repr.mean(0))
                fp_input = self.pair_input(fragment_repr, pocket_repr)
                fp_feature = self.fp_feature_mlp(fp_input)
                fp_features.append(fp_feature)
                fp_scores.append(self.fp_score_mlp(fp_input).squeeze())
                ar_input = self.pair_input(
                    atom_repr[:, None, :].expand(-1, len(residue_repr), -1),
                    residue_repr[None, :, :].expand(len(atom_repr), -1, -1),
                ).reshape(-1, 4 * atom_repr.shape[-1])
                ar_feature = self.ar_feature_mlp(ar_input)
                ar_score = self.ar_score_mlp(ar_input).squeeze(-1)
                ar_pair = (torch.softmax(ar_score, 0)[:, None] * ar_feature).sum(0)
                ar_features.append(ar_pair)
                ar_pair_scores.append(self.ar_pair_score(ar_pair).squeeze())
        if not fp_features:
            return zero, zero, True
        fp_stack = torch.stack(fp_features)
        ar_stack = torch.stack(ar_features)
        ar_repr = (
            torch.softmax(torch.stack(ar_pair_scores), 0)[:, None] * ar_stack
        ).sum(0)
        if self.cross_scale_mode == "bottom_up":
            fp_stack = self.fp_norm(
                fp_stack
                + torch.sigmoid(self.ar_to_fp_gate) * self.ar_to_fp(ar_stack)
            )
        fp_repr = (
            torch.softmax(torch.stack(fp_scores), 0)[:, None] * fp_stack
        ).sum(0)
        return fp_repr, ar_repr, False

    def _local_reference(self, atom_nodes, protein_nodes, batch):
        values = [
            self._local_one_reference(index, atom_nodes, protein_nodes, batch)
            for index in range(len(batch["drug_id"]))
        ]
        return (
            torch.stack([value[0] for value in values]),
            torch.stack([value[1] for value in values]),
            torch.tensor(
                [not value[2] for value in values],
                dtype=torch.bool,
                device=atom_nodes.device,
            ),
        )

    def forward(self, batch):
        need_nodes = bool({"fp", "ar"} & self.active_scales)
        base, global_repr, atom_nodes, protein_nodes = self._encode_global_and_nodes(
            batch, need_nodes=need_nodes
        )
        batch_size = base.shape[0]
        zero_delta = base.new_zeros(base.shape)
        local_dim = self.fragment_projection.out_features
        fp_repr = base.new_zeros((batch_size, local_dim))
        ar_repr = torch.zeros_like(fp_repr)
        local_valid_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=base.device
        )
        zero_stat = base.new_zeros(())
        stats = {
            "fragments_per_sample": zero_stat,
            "pockets_per_sample": zero_stat,
            "fp_pairs_per_sample": zero_stat,
            "ar_pairs_per_sample": zero_stat,
            "local_missing_rate": zero_stat,
            "sample_count": torch.tensor(batch_size, device=base.device),
            "local_missing_count": torch.zeros((), dtype=torch.long, device=base.device),
            "fp_pairs_total": torch.zeros((), dtype=torch.long, device=base.device),
            "ar_pairs_total": torch.zeros((), dtype=torch.long, device=base.device),
        }
        if need_nodes:
            fp_repr, ar_repr, local_valid_mask, stats = self._local_batched(
                atom_nodes, protein_nodes, batch
            )

        if self.cross_scale_mode == "bottom_up":
            global_repr = self.global_norm(
                global_repr
                + torch.sigmoid(self.fp_to_global_gate) * self.fp_to_global(fp_repr)
            )
        fp_valid_mask = (
            local_valid_mask
            if self.active_scales == {"global", "fp"}
            else torch.zeros_like(local_valid_mask)
        )
        ar_valid_mask = (
            local_valid_mask
            if self.active_scales == {"global", "ar"}
            else torch.zeros_like(local_valid_mask)
        )
        multiscale_valid_mask = (
            local_valid_mask
            if self.active_scales == {"global", "fp", "ar"}
            else torch.zeros_like(local_valid_mask)
        )
        fp_delta = (
            self.fp_head(torch.cat([global_repr, fp_repr], dim=-1))
            if self.active_scales == {"global", "fp"}
            else zero_delta
        )
        ar_delta = (
            self.ar_head(torch.cat([global_repr, ar_repr], dim=-1))
            if self.active_scales == {"global", "ar"}
            else zero_delta
        )
        multiscale_delta = (
            self.multiscale_head(torch.cat([global_repr, fp_repr, ar_repr], dim=-1))
            if self.active_scales == {"global", "fp", "ar"}
            else zero_delta
        )
        fp_delta = fp_delta * fp_valid_mask.to(fp_delta.dtype).unsqueeze(-1)
        ar_delta = ar_delta * ar_valid_mask.to(ar_delta.dtype).unsqueeze(-1)
        multiscale_delta = multiscale_delta * multiscale_valid_mask.to(
            multiscale_delta.dtype
        ).unsqueeze(-1)
        prediction = (
            base
            + self.fp_alpha * fp_delta
            + self.ar_alpha * ar_delta
            + self.multiscale_alpha * multiscale_delta
        )
        self.last_stats = stats
        return {
            "pred": prediction,
            "base_pred": base,
            "fp_delta": fp_delta,
            "ar_delta": ar_delta,
            "multiscale_delta": multiscale_delta,
            "fp_alpha": self.fp_alpha,
            "ar_alpha": self.ar_alpha,
            "multiscale_alpha": self.multiscale_alpha,
            "active_scales": sorted(self.active_scales),
            "fp_valid_mask": fp_valid_mask,
            "ar_valid_mask": ar_valid_mask,
            "multiscale_valid_mask": multiscale_valid_mask,
            "stats": stats,
        }
