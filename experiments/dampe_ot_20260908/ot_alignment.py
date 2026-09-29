"""Train-entity feature-dimension OT. No affinity labels or heldout fitting.

T is a probability coupling; A = T / b[None,:] is the target-normalized
mapping used for the main experiment. Raw E @ T is a separate control.
"""
import hashlib
import json
import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn


def ids_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def compute_cost_matrix(source, target):
    x, y = as_numpy(source), as_numpy(target)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or min(x.shape+y.shape) == 0:
        raise ValueError('Expected nonempty paired [N, D] source and target matrices')
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Nonfinite embedding')
    # Direct differences avoid cancellation for nearly identical/constant columns.
    c = np.empty((x.shape[1], y.shape[1]), dtype=np.float64)
    for i in range(x.shape[1]):
        c[i] = np.sqrt(np.mean((x[:, i, None]-y)**2, axis=0))
    return c


def sinkhorn_transport(cost, epsilon=1e-3, max_iter=50000, tolerance=1e-7,
                       check_every=25):
    c = as_numpy(cost)
    if c.ndim != 2 or min(c.shape) == 0 or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError('Cost must be a finite nonnegative matrix')
    if epsilon <= 0 or max_iter < 1 or tolerance <= 0:
        raise ValueError('Invalid Sinkhorn configuration')
    m, n = c.shape
    a, b = np.full(m, 1/m), np.full(n, 1/n)
    la, lb = np.log(a), np.log(b)
    # Row/column additive potentials do not change the balanced OT optimum.
    # Keep the *raw* RMSE matrix for objectives/provenance; no multiplicative scaling.
    reduced = c-c.min(axis=1, keepdims=True)
    reduced = reduced-reduced.min(axis=0, keepdims=True)
    lk = -reduced/epsilon
    lu, lv = np.zeros(m), np.zeros(n)
    converged = False
    previous_cost = None
    for iteration in range(1, max_iter+1):
        lu = la-logsumexp(lk+lv[None, :], axis=1)
        lv = lb-logsumexp(lk+lu[:, None], axis=0)
        if iteration % check_every == 0 or iteration == max_iter:
            t = np.exp(lk+lu[:, None]+lv[None, :])
            row_error = float(np.max(np.abs(t.sum(1)-a)))
            col_error = float(np.max(np.abs(t.sum(0)-b)))
            rel_error = max(row_error*m, col_error*n)
            value = float(np.sum(t*c))
            change = None if previous_cost is None else abs(value-previous_cost)
            previous_cost = value
            if rel_error <= tolerance:
                converged = True
                break
    meta = dict(epsilon=epsilon, iterations=iteration, max_iter=max_iter,
                tolerance_relative=tolerance, converged=converged,
                row_marginal_max_abs_error=row_error,
                column_marginal_max_abs_error=col_error,
                marginal_max_relative_error=rel_error,
                transport_cost=value, transport_cost_change=change,
                solver='float64_log_sinkhorn', cost_scaling='none',
                additive_cost_reduction=True)
    if not converged:
        raise RuntimeError('Sinkhorn failed marginal tolerance: '+json.dumps(meta))
    return t, meta


def diagnostics(t, source, target):
    x, y = as_numpy(source), as_numpy(target)
    a = t*t.shape[1]
    z = x@a
    def entropy(prob, axis):
        return -(prob*np.log(np.maximum(prob, np.finfo(float).tiny))).sum(axis=axis)
    sv = np.linalg.svd(a, compute_uv=False)
    p = sv/max(sv.sum(), np.finfo(float).tiny)
    def stats(v):
        return dict(mean_l2=float(np.linalg.norm(v, axis=1).mean()),
                    mean_feature_std=float(v.std(axis=0).mean()),
                    max_abs=float(np.abs(v).max()))
    result = dict(T_min=float(t.min()), T_max=float(t.max()), T_mean=float(t.mean()),
                  floating_nonzero_ratio=float(np.mean(t > 0)),
                  normalized_column_entropy_mean=float(entropy(a, 0).mean()),
                  normalized_row_entropy_mean=float(entropy(t*t.shape[0], 1).mean()),
                  singular_values=sv.tolist(), effective_rank=float(np.exp(entropy(p, 0))),
                  source=stats(x), target=stats(y), mapped_normalized=stats(z),
                  mapped_raw=stats(x@t))
    if x.shape[1] == y.shape[1]:
        def cosine(v, w):
            return float(np.mean(np.sum(v*w, 1)/np.maximum(np.linalg.norm(v, axis=1)*np.linalg.norm(w, axis=1), 1e-30)))
        result.update(train_cosine_before=cosine(x, y), train_cosine_after=cosine(z, y))
    return result


def fit_alignment(source, target, source_ids, target_ids, allowed_train_ids,
                  epsilon=1e-3, shuffle_seed=None):
    ids = list(map(str, source_ids))
    if ids != list(map(str, target_ids)):
        raise ValueError('Entity order mismatch between modalities')
    if len(ids) != len(set(ids)) or set(ids) != set(map(str, allowed_train_ids)):
        raise ValueError('OT requires exactly the unique allowed training entities')
    x, y = as_numpy(source), as_numpy(target)
    if len(ids) != len(x) or len(ids) != len(y):
        raise ValueError('ID/embedding length mismatch')
    permutation = np.arange(len(ids))
    if shuffle_seed is not None:
        permutation = np.random.default_rng(shuffle_seed).permutation(len(ids))
    c = compute_cost_matrix(x, y[permutation])
    t, meta = sinkhorn_transport(c, epsilon)
    meta.update(source_dim=x.shape[1], target_dim=y.shape[1],
                num_unique_train_entities=len(ids), train_entity_ids=ids,
                train_entity_ids_hash=ids_hash(ids), cost_function='dimension_rmse',
                cost_min=float(c.min()), cost_max=float(c.max()), cost_mean=float(c.mean()),
                shuffle_seed=shuffle_seed, target_permutation=permutation.tolist(),
                diagnostics=diagnostics(t, x, y))
    return torch.from_numpy(t), meta


def apply_ot_alignment(source, mapping):
    return source @ mapping.to(device=source.device, dtype=source.dtype)


class OTAlignment(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.register_buffer('mapping', torch.as_tensor(mapping, dtype=torch.float32).clone())

    def forward(self, source):
        return apply_ot_alignment(source, self.mapping)
