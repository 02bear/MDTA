"""Coverage-selected atom-residue graph with optional PCIM conditional attention."""
import math

import torch
from torch import nn


class PairGraph(nn.Module):
    """Message passing over selected atom-residue pairs.

    The conditional branch follows PCIM's query-independent condition-to-key term:
      e_uv = (q_u @ k_v + b(c) @ k_v) / sqrt(d_h) + relation_bias.
    Because b(c) @ k_v changes with key v, it cannot cancel inside row softmax.
    """
    def __init__(self, width=32, heads=2, dropout=0.1, condition_dim=16, conditioned=False):
        super().__init__()
        assert width % heads == 0
        self.width, self.heads = width, heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.relation_bias = nn.Parameter(torch.zeros(2, heads))
        self.proj = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(),
                                nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.condition_bias = None
        if conditioned:
            # Do not advance the RNG seen by the unchanged downstream modules.
            # This makes every shared parameter bitwise identical to baseline at init.
            cpu_rng = torch.get_rng_state()
            self.condition_bias = nn.Linear(condition_dim, width)
            torch.set_rng_state(cpu_rng)
            nn.init.zeros_(self.condition_bias.weight)
            nn.init.zeros_(self.condition_bias.bias)

    def forward(self, tokens, valid, atom_index, residue_index, condition=None, diagnostics=False):
        b, k, h = tokens.shape
        head_width = h // self.heads
        q, key, value = self.qkv(self.norm1(tokens)).reshape(
            b, k, 3, self.heads, head_width).permute(2, 0, 3, 1, 4).unbind(0)
        qk_scores = q @ key.transpose(-1, -2) / math.sqrt(head_width)
        same_a = atom_index[:, :, None] == atom_index[:, None, :]
        same_p = residue_index[:, :, None] == residue_index[:, None, :]
        base_logits = qk_scores + same_a[:, None] * self.relation_bias[0][None, :, None, None]
        base_logits = base_logits + same_p[:, None] * self.relation_bias[1][None, :, None, None]
        condition_scores = torch.zeros_like(qk_scores)
        if condition is not None:
            assert self.condition_bias is not None
            bias = self.condition_bias(condition).reshape(b, self.heads, 1, head_width)
            condition_scores = bias @ key.transpose(-1, -2) / math.sqrt(head_width)
        key_mask = valid[:, None, None, :]
        base_logits = base_logits.masked_fill(~key_mask, -torch.inf)
        logits = (base_logits + condition_scores).masked_fill(~key_mask, -torch.inf)
        attention = logits.softmax(-1)
        message = (self.drop(attention) @ value).transpose(1, 2).reshape(b, k, h)
        tokens = tokens + self.drop(self.proj(message))
        tokens = tokens + self.drop(self.ff(self.norm2(tokens)))
        tokens = tokens.masked_fill(~valid[..., None], 0.0)
        if not diagnostics:
            return tokens
        pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
        denominator = (pair_valid.sum(dim=(1, 2, 3)) * self.heads).clamp_min(1)
        qk_rms = ((qk_scores.square() * pair_valid).sum(dim=(1, 2, 3)) / denominator).sqrt()
        condition_rms = ((condition_scores.square() * pair_valid).sum(dim=(1, 2, 3)) / denominator).sqrt()
        base_attention = base_logits.softmax(-1)
        attention_change = (((attention - base_attention).square() * pair_valid).sum(dim=(1, 2, 3)) /
                            denominator).sqrt()
        graph_info = {
            'qk_score_rms': qk_rms,
            'condition_score_rms': condition_rms,
            'condition_to_qk_rms': condition_rms / qk_rms.clamp_min(1e-12),
            'attention_condition_change_rms': attention_change,
        }
        return tokens, graph_info


class PairResidual(nn.Module):
    def __init__(self, variant='pair_graph', width=32, top_k=64, max_delta=0.5,
                 selection='coverage', coverage_budget=32, attention_mode='none',
                 shared_global_pair=None):
        super().__init__()
        assert variant == 'pair_graph'
        assert selection == 'coverage'
        assert attention_mode in ('none', 'dynamic', 'shared')
        self.variant, self.top_k, self.max_delta = variant, top_k, max_delta
        self.selection, self.coverage_budget = selection, coverage_budget
        self.attention_mode = attention_mode
        self.aa_embedding = nn.Embedding(21, 8)
        self.atom_projection = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width))
        self.protein_projection = nn.Sequential(nn.Linear(136, width), nn.LayerNorm(width))
        self.condition = nn.Sequential(nn.Linear(256, 16), nn.SiLU(), nn.Linear(16, 16))
        self.token_mlp = nn.Sequential(nn.Linear(3 * width + 1, width), nn.SiLU(), nn.Dropout(0.1))
        self.condition_gate = nn.Linear(16, width)
        self.graph = PairGraph(width, conditioned=attention_mode != 'none')
        self.pool = nn.Sequential(nn.Linear(width + 16, width), nn.Tanh(), nn.Linear(width, 1))
        self.readout = nn.Sequential(nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, 1))
        self.output = self.readout[-1]
        if attention_mode == 'shared':
            assert shared_global_pair is not None and tuple(shared_global_pair.shape) == (256,)
            self.register_buffer('shared_global_pair', shared_global_pair.detach().clone().float())
        else:
            assert shared_global_pair is None
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, b, diagnostics=False, attention_override=None):
        assert attention_override in (None, 'off', 'shuffle')
        global_pair = torch.cat([b['drug_global'], b['protein_global']], -1)
        a = self.atom_projection(b['atoms'])
        p = self.protein_projection(torch.cat([b['residues'], self.aa_embedding(b['aa'])], -1))
        c = self.condition(global_pair)
        valid_pairs = b['atom_mask'][:, :, None] & b['residue_mask'][:, None, :]
        scores = (a @ p.transpose(-1, -2)) / math.sqrt(a.size(-1))
        scores = scores.masked_fill(~valid_pairs, -torch.inf)
        count = min(self.top_k, scores.size(1) * scores.size(2))
        selected, flat = select_pairs(scores, count, self.coverage_budget)
        valid = torch.isfinite(selected)
        ai, pi = flat // p.size(1), flat % p.size(1)
        av = a.gather(1, ai[..., None].expand(-1, -1, a.size(-1)))
        pv = p.gather(1, pi[..., None].expand(-1, -1, p.size(-1)))
        selected = selected.masked_fill(~valid, 0.0)
        tokens = self.token_mlp(torch.cat([av, pv, av * pv, selected[..., None]], -1))
        tokens = tokens * (1.0 + self.condition_gate(c).sigmoid()[:, None])
        tokens = tokens.masked_fill(~valid[..., None], 0.0)

        graph_condition = None
        if self.attention_mode == 'dynamic':
            graph_condition = c
        elif self.attention_mode == 'shared':
            graph_condition = self.condition(self.shared_global_pair[None]).expand(len(c), -1)
        if attention_override == 'off':
            graph_condition = None
        elif attention_override == 'shuffle' and graph_condition is not None:
            graph_condition = graph_condition.roll(1, 0)
        graph_result = self.graph(tokens, valid, ai, pi, graph_condition, diagnostics)
        if diagnostics:
            tokens, graph_info = graph_result
        else:
            tokens = graph_result
            graph_info = {}

        logits = self.pool(torch.cat([tokens, c[:, None].expand(-1, count, -1)], -1)).squeeze(-1)
        weights = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        z = (weights[..., None] * tokens).sum(1)
        delta = self.max_delta * torch.tanh(self.readout(z).squeeze(-1))
        if not diagnostics:
            return delta
        info = {'atom_index': ai, 'residue_index': pi, 'valid': valid, 'weights': weights,
                'score': selected, 'entropy': -(weights * weights.clamp_min(1e-12).log()).sum(-1),
                **graph_info}
        if graph_condition is not None:
            info['attention_condition'] = graph_condition
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
