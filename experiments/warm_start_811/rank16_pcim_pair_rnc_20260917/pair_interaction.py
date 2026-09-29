"""Coverage PCIM with bidirectional interaction and optional pair-RNC head."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class CrossInteraction(nn.Module):
    """Update local atom and/or residue states from the opposite entity."""
    def __init__(self, width=32, heads=2, dropout=0.1, max_scale=0.25):
        super().__init__()
        assert width % heads == 0
        self.width, self.heads, self.max_scale = width, heads, max_scale
        self.atom_norm = nn.LayerNorm(width)
        self.residue_norm = nn.LayerNorm(width)
        # Separate directional projections keep the one-way controls interpretable.
        self.atom_q = nn.Linear(width, width)
        self.residue_k = nn.Linear(width, width)
        self.residue_v = nn.Linear(width, width)
        self.atom_out = nn.Linear(width, width)
        self.residue_q = nn.Linear(width, width)
        self.atom_k = nn.Linear(width, width)
        self.atom_v = nn.Linear(width, width)
        self.residue_out = nn.Linear(width, width)
        self.atom_gamma = nn.Parameter(torch.zeros(()))
        self.residue_gamma = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.width // self.heads).transpose(1, 2)

    @staticmethod
    def _masked_mean(x, mask):
        return (x * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    def forward(self, atoms, residues, atom_mask, residue_mask, mode, shuffle_opposite=False,
                diagnostics=False):
        assert mode in ('atom_conditioned', 'residue_conditioned', 'bidirectional')
        atom_source = self.atom_norm(atoms)
        residue_source = self.residue_norm(residues)
        atom_keys, atom_key_mask = atom_source, atom_mask
        residue_keys, residue_key_mask = residue_source, residue_mask
        if shuffle_opposite:
            atom_keys, atom_key_mask = atom_keys.roll(1, 0), atom_key_mask.roll(1, 0)
            residue_keys, residue_key_mask = residue_keys.roll(1, 0), residue_key_mask.roll(1, 0)
        info = {}
        atom_update = torch.zeros_like(atoms)
        residue_update = torch.zeros_like(residues)

        if mode in ('atom_conditioned', 'bidirectional'):
            q = self._heads(self.atom_q(atom_source))
            key = self._heads(self.residue_k(residue_keys))
            value = self._heads(self.residue_v(residue_keys))
            score = q @ key.transpose(-1, -2) / math.sqrt(self.width // self.heads)
            score = score.masked_fill(~residue_key_mask[:, None, None, :], -torch.inf)
            attention = score.softmax(-1)
            context = (self.dropout(attention) @ value).transpose(1, 2).reshape_as(atoms)
            raw = self.atom_out(context)
            scale = self.max_scale * self.atom_gamma.tanh()
            atom_update = scale * raw
            atoms = atoms + self.dropout(atom_update)
            if diagnostics:
                entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1).mean(1)
                info['cross_atom_attention_entropy'] = self._masked_mean(entropy, atom_mask)
                numerator = (atom_update.square().sum(-1) * atom_mask).sum(-1)
                denominator = atom_mask.sum(-1).clamp_min(1)
                base = (atoms.detach().square().sum(-1) * atom_mask).sum(-1) / denominator
                info['cross_atom_update_rms'] = (numerator / denominator).sqrt()
                info['cross_atom_update_ratio'] = info['cross_atom_update_rms'] / base.sqrt().clamp_min(1e-12)
                info['cross_atom_gamma'] = scale.detach().expand(len(atoms))

        if mode in ('residue_conditioned', 'bidirectional'):
            q = self._heads(self.residue_q(residue_source))
            key = self._heads(self.atom_k(atom_keys))
            value = self._heads(self.atom_v(atom_keys))
            score = q @ key.transpose(-1, -2) / math.sqrt(self.width // self.heads)
            score = score.masked_fill(~atom_key_mask[:, None, None, :], -torch.inf)
            attention = score.softmax(-1)
            context = (self.dropout(attention) @ value).transpose(1, 2).reshape_as(residues)
            raw = self.residue_out(context)
            scale = self.max_scale * self.residue_gamma.tanh()
            residue_update = scale * raw
            residues = residues + self.dropout(residue_update)
            if diagnostics:
                entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1).mean(1)
                info['cross_residue_attention_entropy'] = self._masked_mean(entropy, residue_mask)
                numerator = (residue_update.square().sum(-1) * residue_mask).sum(-1)
                denominator = residue_mask.sum(-1).clamp_min(1)
                base = (residues.detach().square().sum(-1) * residue_mask).sum(-1) / denominator
                info['cross_residue_update_rms'] = (numerator / denominator).sqrt()
                info['cross_residue_update_ratio'] = info['cross_residue_update_rms'] / base.sqrt().clamp_min(1e-12)
                info['cross_residue_gamma'] = scale.detach().expand(len(residues))
        atoms = atoms.masked_fill(~atom_mask[..., None], 0.0)
        residues = residues.masked_fill(~residue_mask[..., None], 0.0)
        return atoms, residues, info


class PairGraph(nn.Module):
    def __init__(self, width=32, heads=2, dropout=0.1):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.relation_bias = nn.Parameter(torch.zeros(2, heads))
        self.proj = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(),
                                nn.Dropout(dropout), nn.Linear(2 * width, width))

    def forward(self, tokens, valid, atom_index, residue_index):
        b, k, h = tokens.shape
        q, key, value = self.qkv(self.norm1(tokens)).reshape(
            b, k, 3, self.heads, h // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        logits = q @ key.transpose(-1, -2) / math.sqrt(h // self.heads)
        same_a = atom_index[:, :, None] == atom_index[:, None, :]
        same_p = residue_index[:, :, None] == residue_index[:, None, :]
        logits = logits + same_a[:, None] * self.relation_bias[0][None, :, None, None]
        logits = logits + same_p[:, None] * self.relation_bias[1][None, :, None, None]
        logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
        attention = logits.softmax(-1)
        message = (self.drop(attention) @ value).transpose(1, 2).reshape(b, k, h)
        tokens = tokens + self.drop(self.proj(message))
        tokens = tokens + self.drop(self.ff(self.norm2(tokens)))
        return tokens.masked_fill(~valid[..., None], 0.0)


class PairResidual(nn.Module):
    def __init__(self, variant='baseline', width=32, top_k=64, max_delta=0.5,
                 selection='coverage', coverage_budget=32, cross_max_scale=0.25,
                 enable_contrast=False, contrast_dim=16):
        super().__init__()
        assert variant in ('baseline', 'atom_conditioned', 'residue_conditioned', 'bidirectional')
        assert selection == 'coverage'
        self.variant, self.top_k, self.max_delta = variant, top_k, max_delta
        self.selection, self.coverage_budget = selection, coverage_budget
        self.aa_embedding = nn.Embedding(21, 8)
        self.atom_projection = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width))
        self.protein_projection = nn.Sequential(nn.Linear(136, width), nn.LayerNorm(width))
        self.condition = nn.Sequential(nn.Linear(256, 16), nn.SiLU(), nn.Linear(16, 16))
        self.token_mlp = nn.Sequential(nn.Linear(3 * width + 1, width), nn.SiLU(), nn.Dropout(0.1))
        self.condition_gate = nn.Linear(16, width)
        self.graph = PairGraph(width)
        self.pool = nn.Sequential(nn.Linear(width + 16, width), nn.Tanh(), nn.Linear(width, 1))
        self.readout = nn.Sequential(nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, 1))
        self.output = self.readout[-1]
        self.cross = None
        if variant != 'baseline':
            # Preserve the RNG stream of every unchanged baseline parameter.
            cpu_rng = torch.get_rng_state()
            self.cross = CrossInteraction(width, heads=2, dropout=0.1, max_scale=cross_max_scale)
            torch.set_rng_state(cpu_rng)
        self.contrast_projection = None
        if enable_contrast:
            self.contrast_projection = nn.Sequential(
                nn.Linear(width, width), nn.SiLU(),
                nn.LayerNorm(width, elementwise_affine=False),
                nn.Linear(width, contrast_dim),
                nn.LayerNorm(contrast_dim, elementwise_affine=False),
            )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, b, diagnostics=False, cross_override=None, return_contrast=False):
        assert cross_override in (None, 'off', 'shuffle_opposite')
        assert not (diagnostics and return_contrast)
        global_pair = torch.cat([b['drug_global'], b['protein_global']], -1)
        atoms = self.atom_projection(b['atoms'])
        residues = self.protein_projection(torch.cat([b['residues'], self.aa_embedding(b['aa'])], -1))
        cross_info = {}
        if self.cross is not None and cross_override != 'off':
            atoms, residues, cross_info = self.cross(
                atoms, residues, b['atom_mask'], b['residue_mask'], self.variant,
                shuffle_opposite=cross_override == 'shuffle_opposite', diagnostics=diagnostics)
        c = self.condition(global_pair)
        valid_pairs = b['atom_mask'][:, :, None] & b['residue_mask'][:, None, :]
        scores = (atoms @ residues.transpose(-1, -2)) / math.sqrt(atoms.size(-1))
        scores = scores.masked_fill(~valid_pairs, -torch.inf)
        count = min(self.top_k, scores.size(1) * scores.size(2))
        selected, flat = select_pairs(scores, count, self.coverage_budget)
        valid = torch.isfinite(selected)
        ai, pi = flat // residues.size(1), flat % residues.size(1)
        av = atoms.gather(1, ai[..., None].expand(-1, -1, atoms.size(-1)))
        pv = residues.gather(1, pi[..., None].expand(-1, -1, residues.size(-1)))
        selected = selected.masked_fill(~valid, 0.0)
        tokens = self.token_mlp(torch.cat([av, pv, av * pv, selected[..., None]], -1))
        tokens = tokens * (1.0 + self.condition_gate(c).sigmoid()[:, None])
        tokens = tokens.masked_fill(~valid[..., None], 0.0)
        tokens = self.graph(tokens, valid, ai, pi)
        logits = self.pool(torch.cat([tokens, c[:, None].expand(-1, count, -1)], -1)).squeeze(-1)
        weights = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        z = (weights[..., None] * tokens).sum(1)
        delta = self.max_delta * torch.tanh(self.readout(z).squeeze(-1))
        if return_contrast:
            if self.contrast_projection is None:
                raise RuntimeError('Contrast representation requested for a model without an RNC head')
            return delta, F.normalize(self.contrast_projection(z), p=2, dim=-1)
        if not diagnostics:
            return delta
        info = {'atom_index': ai, 'residue_index': pi, 'valid': valid, 'weights': weights,
                'score': selected, 'entropy': -(weights * weights.clamp_min(1e-12).log()).sum(-1),
                **cross_info}
        return delta, info


def select_pairs(scores, count, coverage_budget=32):
    """Reserve one best-residue pair for up to 32 atoms, then fill globally."""
    flat_scores = scores.flatten(1)
    per_atom, best_residue = scores.max(-1)
    reserve_count = min(coverage_budget, scores.size(1), count)
    reserve_values, reserve_atoms = per_atom.topk(reserve_count, dim=-1)
    reserve_residues = best_residue.gather(1, reserve_atoms)
    reserve_flat = reserve_atoms * scores.size(2) + reserve_residues
    reserved = torch.zeros_like(flat_scores, dtype=torch.bool)
    reserved.scatter_(1, reserve_flat, torch.isfinite(reserve_values))
    priority = flat_scores.masked_fill(reserved, torch.inf)
    _, flat = priority.topk(count, dim=-1)
    selected = flat_scores.gather(1, flat)
    selected, order = selected.sort(dim=-1, descending=True)
    return selected, flat.gather(1, order)
