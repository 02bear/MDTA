"""Small conditional atom-residue pair graph on frozen Rank16 features."""
import math
import torch
from torch import nn


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
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.SiLU(),
                                nn.Dropout(dropout), nn.Linear(2*width, width))

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
        attn = logits.softmax(-1)
        message = (self.drop(attn) @ value).transpose(1, 2).reshape(b, k, h)
        tokens = tokens + self.drop(self.proj(message))
        tokens = tokens + self.drop(self.ff(self.norm2(tokens)))
        return tokens.masked_fill(~valid[..., None], 0.0)


class PairResidual(nn.Module):
    def __init__(self, variant='pair_graph', width=32, top_k=64, max_delta=0.5, selection='coverage', coverage_budget=32):
        super().__init__()
        assert variant in ('pair_graph', 'pair_pool', 'global_mlp')
        self.variant, self.top_k, self.max_delta = variant, top_k, max_delta
        assert selection in ('global', 'coverage')
        self.selection, self.coverage_budget = selection, coverage_budget
        if variant == 'global_mlp':
            self.global_net = nn.Sequential(nn.Linear(256, 96), nn.SiLU(), nn.Dropout(0.1),
                                            nn.Linear(96, 24), nn.SiLU(), nn.Linear(24, 1))
            self.output = self.global_net[-1]
        else:
            self.aa_embedding = nn.Embedding(21, 8)
            self.atom_projection = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width))
            self.protein_projection = nn.Sequential(nn.Linear(136, width), nn.LayerNorm(width))
            self.condition = nn.Sequential(nn.Linear(256, 16), nn.SiLU(), nn.Linear(16, 16))
            self.token_mlp = nn.Sequential(nn.Linear(3*width+1, width), nn.SiLU(), nn.Dropout(0.1))
            self.condition_gate = nn.Linear(16, width)
            self.graph = PairGraph(width) if variant == 'pair_graph' else None
            self.pool = nn.Sequential(nn.Linear(width+16, width), nn.Tanh(), nn.Linear(width, 1))
            self.readout = nn.Sequential(nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, 1))
            self.output = self.readout[-1]
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, b, diagnostics=False):
        global_pair = torch.cat([b['drug_global'], b['protein_global']], -1)
        if self.variant == 'global_mlp':
            delta = self.max_delta * torch.tanh(self.global_net(global_pair).squeeze(-1))
            return (delta, {}) if diagnostics else delta
        a = self.atom_projection(b['atoms'])
        p = self.protein_projection(torch.cat([b['residues'], self.aa_embedding(b['aa'])], -1))
        c = self.condition(global_pair)
        valid_pairs = b['atom_mask'][:, :, None] & b['residue_mask'][:, None, :]
        scores = (a @ p.transpose(-1, -2)) / math.sqrt(a.size(-1))
        scores = scores.masked_fill(~valid_pairs, -torch.inf)
        count = min(self.top_k, scores.size(1)*scores.size(2))
        selected, flat = select_pairs(scores, count, self.selection, self.coverage_budget)
        valid = torch.isfinite(selected)
        ai, pi = flat // p.size(1), flat % p.size(1)
        av = a.gather(1, ai[..., None].expand(-1, -1, a.size(-1)))
        pv = p.gather(1, pi[..., None].expand(-1, -1, p.size(-1)))
        selected = selected.masked_fill(~valid, 0.0)
        tokens = self.token_mlp(torch.cat([av, pv, av*pv, selected[..., None]], -1))
        tokens = tokens * (1.0 + self.condition_gate(c).sigmoid()[:, None])
        tokens = tokens.masked_fill(~valid[..., None], 0.0)
        if self.graph is not None:
            tokens = self.graph(tokens, valid, ai, pi)
        logits = self.pool(torch.cat([tokens, c[:, None].expand(-1, count, -1)], -1)).squeeze(-1)
        weights = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        z = (weights[..., None]*tokens).sum(1)
        delta = self.max_delta*torch.tanh(self.readout(z).squeeze(-1))
        if not diagnostics:
            return delta
        info = {'atom_index': ai, 'residue_index': pi, 'valid': valid, 'weights': weights,
                'score': selected, 'entropy': -(weights*weights.clamp_min(1e-12).log()).sum(-1)}
        return delta, info


def select_pairs(scores, count, selection, coverage_budget=32):
    """Keep original score ordering; reserve at most one pair per covered atom."""
    flat_scores = scores.flatten(1)
    if selection == 'global':
        return flat_scores.topk(count, dim=-1)
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
