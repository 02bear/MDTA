"""Train-entity feature-dimension OT for the periodic end-to-end runs.

The requested map is A = T / b[None, :] = 128 * T.  The solver keeps the
requested raw RMSE cost and final epsilon, while using larger epsilons only as
a numerical warm start for the final float64 log-domain Sinkhorn solve.
"""
import hashlib
import json

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn


class SinkhornConvergenceError(RuntimeError):
    """Finite Sinkhorn iterate that did not meet the requested tolerance."""
    def __init__(self, metadata, transport):
        self.metadata = metadata
        self.transport = transport
        super().__init__('Sinkhorn failed marginal tolerance: '+json.dumps(metadata))


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
    c = np.empty((x.shape[1], y.shape[1]), dtype=np.float64)
    for i in range(x.shape[1]):
        c[i] = np.sqrt(np.mean((x[:, i, None]-y)**2, axis=0))
    return c


def _marginal_errors(log_kernel, log_u, log_v, a, b):
    transport = np.exp(log_kernel+log_u[:, None]+log_v[None, :])
    row_error = float(np.max(np.abs(transport.sum(1)-a)))
    col_error = float(np.max(np.abs(transport.sum(0)-b)))
    relative_error = max(row_error*len(a), col_error*len(b))
    return transport, row_error, col_error, relative_error


def _primal_marginal_polish(transport, a, b, tolerance, max_iter=50000,
                            check_every=100):
    """Finish diagonal matrix balancing after the stable log-domain solve.

    Row/column rescaling preserves ``diag(u) K diag(v)`` and therefore the
    entropic OT solution.  It is used only when the log solve is already close;
    at that point direct float64 sums are safe and much faster than repeating
    tens of thousands of logsumexp calls.
    """
    t = np.asarray(transport, dtype=np.float64).copy()
    if not np.isfinite(t).all() or (t < 0).any():
        raise RuntimeError('Cannot polish a nonfinite transport plan')
    row_error = col_error = relative_error = float('inf')
    for iteration in range(1, max_iter+1):
        row_sum = t.sum(axis=1)
        if (row_sum <= 0).any():
            raise RuntimeError('Zero row encountered during marginal polishing')
        t *= (a/row_sum)[:, None]
        col_sum = t.sum(axis=0)
        if (col_sum <= 0).any():
            raise RuntimeError('Zero column encountered during marginal polishing')
        t *= (b/col_sum)[None, :]
        if iteration % check_every == 0 or iteration == max_iter:
            row_error = float(np.max(np.abs(t.sum(1)-a)))
            col_error = float(np.max(np.abs(t.sum(0)-b)))
            relative_error = max(row_error*len(a), col_error*len(b))
            if relative_error <= tolerance:
                return t, iteration, row_error, col_error, relative_error, True
    return t, iteration, row_error, col_error, relative_error, False


def sinkhorn_transport(cost, epsilon=1e-3, max_iter=50000, tolerance=1e-5,
                       check_every=25):
    """Solve balanced entropic OT with epsilon continuation.

    Continuation changes only how the dual potentials are initialized.  The
    returned coupling is solved and checked at exactly ``epsilon``.  This
    avoids false failures when a cold start at 1e-3 progresses extremely
    slowly after its transport cost has already stabilized.
    """
    c = as_numpy(cost)
    if c.ndim != 2 or min(c.shape) == 0 or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError('Cost must be a finite nonnegative matrix')
    if epsilon <= 0 or max_iter < 1 or tolerance <= 0:
        raise ValueError('Invalid Sinkhorn configuration')
    m, n = c.shape
    a, b = np.full(m, 1/m), np.full(n, 1/n)
    log_a, log_b = np.log(a), np.log(b)
    # Additive row/column reductions preserve the balanced OT optimum.
    reduced = c-c.min(axis=1, keepdims=True)
    reduced = reduced-reduced.min(axis=0, keepdims=True)

    # Geometric epsilon continuation, ending at the requested epsilon.  Store
    # dimensional dual potentials (f, g) so warm starts convert correctly when
    # epsilon changes.
    start_epsilon = min(max(epsilon, float(reduced.max())/8), epsilon*64)
    if start_epsilon > epsilon*(1+1e-12):
        count = int(np.ceil(np.log2(start_epsilon/epsilon)))+1
        schedule = np.geomspace(start_epsilon, epsilon, count).tolist()
        schedule[-1] = float(epsilon)
    else:
        schedule = [float(epsilon)]
    f = np.zeros(m, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    stage_records = []
    total_iterations = 0
    transport = None
    row_error = col_error = relative_error = float('inf')
    value = change = None
    final_converged = False
    polish_iterations = 0

    for stage_index, stage_epsilon in enumerate(schedule):
        log_kernel = -reduced/stage_epsilon
        log_u, log_v = f/stage_epsilon, g/stage_epsilon
        is_final = stage_index == len(schedule)-1
        stage_limit = max_iter if is_final else min(5000, max_iter)
        stage_tolerance = tolerance if is_final else 1e-6
        previous_cost = None
        stage_converged = False
        for iteration in range(1, stage_limit+1):
            log_u = log_a-logsumexp(log_kernel+log_v[None, :], axis=1)
            log_v = log_b-logsumexp(log_kernel+log_u[:, None], axis=0)
            # Fix the free additive gauge to keep warm-start potentials bounded.
            gauge = float(log_u.mean())
            log_u -= gauge
            log_v += gauge
            if iteration % check_every == 0 or iteration == stage_limit:
                transport, row_error, col_error, relative_error = _marginal_errors(
                    log_kernel, log_u, log_v, a, b)
                value = float(np.sum(transport*c))
                change = None if previous_cost is None else abs(value-previous_cost)
                previous_cost = value
                if relative_error <= stage_tolerance:
                    stage_converged = True
                    break
        total_iterations += iteration
        f, g = stage_epsilon*log_u, stage_epsilon*log_v
        stage_records.append(dict(epsilon=float(stage_epsilon), iterations=iteration,
                                  converged=stage_converged,
                                  marginal_max_relative_error=relative_error))
        if is_final:
            final_converged = stage_converged

    # A cold 1e-3 problem can be within 1e-4 relative marginal error while log
    # iterations make only tiny progress.  Finish the mathematically equivalent
    # diagonal balancing in primal float64 once it is numerically safe.
    if not final_converged and relative_error <= 1e-4:
        (transport, polish_iterations, row_error, col_error, relative_error,
         final_converged) = _primal_marginal_polish(
            transport, a, b, tolerance)
        value = float(np.sum(transport*c))

    meta = dict(epsilon=epsilon, iterations=total_iterations,
                final_stage_iterations=stage_records[-1]['iterations'],
                max_iter=max_iter, tolerance_relative=tolerance,
                converged=final_converged,
                row_marginal_max_abs_error=row_error,
                column_marginal_max_abs_error=col_error,
                marginal_max_relative_error=relative_error,
                transport_cost=value, transport_cost_change=change,
                primal_marginal_polish_iterations=polish_iterations,
                solver='float64_log_sinkhorn_epsilon_continuation_with_diagonal_polish',
                epsilon_schedule=stage_records, cost_scaling='none',
                additive_cost_reduction=True)
    if not final_converged:
        raise SinkhornConvergenceError(meta, transport)
    return transport, meta


def diagnostics(t, source, target):
    x, y = as_numpy(source), as_numpy(target)
    a = t*t.shape[1]
    z = x@a
    def entropy(prob, axis):
        return -(prob*np.log(np.maximum(prob, np.finfo(float).tiny))).sum(axis=axis)
    singular_values = np.linalg.svd(a, compute_uv=False)
    p = singular_values/max(singular_values.sum(), np.finfo(float).tiny)
    def stats(v):
        return dict(mean_l2=float(np.linalg.norm(v, axis=1).mean()),
                    mean_feature_std=float(v.std(axis=0).mean()),
                    max_abs=float(np.abs(v).max()))
    result = dict(T_min=float(t.min()), T_max=float(t.max()), T_mean=float(t.mean()),
                  floating_nonzero_ratio=float(np.mean(t > 0)),
                  normalized_column_entropy_mean=float(entropy(a, 0).mean()),
                  normalized_row_entropy_mean=float(entropy(t*t.shape[0], 1).mean()),
                  singular_values=singular_values.tolist(), effective_rank=float(np.exp(entropy(p, 0))),
                  source=stats(x), target=stats(y), mapped_normalized=stats(z),
                  mapped_raw=stats(x@t))
    if x.shape[1] == y.shape[1]:
        def cosine(v, w):
            denominator = np.maximum(np.linalg.norm(v, axis=1)*np.linalg.norm(w, axis=1), 1e-30)
            return float(np.mean(np.sum(v*w, 1)/denominator))
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
    try:
        t, meta = sinkhorn_transport(c, epsilon)
    except SinkhornConvergenceError as error:
        error.metadata.update(source_dim=x.shape[1], target_dim=y.shape[1],
                              num_unique_train_entities=len(ids), train_entity_ids=ids,
                              train_entity_ids_hash=ids_hash(ids), cost_function='dimension_rmse',
                              cost_min=float(c.min()), cost_max=float(c.max()),
                              cost_mean=float(c.mean()), cost_std=float(c.std()),
                              shuffle_seed=shuffle_seed,
                              target_permutation=permutation.tolist())
        raise
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
