"""Pair-level Rank-N-Contrast loss and stratified auxiliary-batch sampling."""
import numpy as np
import torch


def _pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return (x * y).sum() / denominator.clamp_min(1e-12)


def rank_n_contrast_loss(features, labels, temperature=2.0, mode='standard',
                         high_threshold=7.0, high_width=0.5, high_strength=2.0):
    """Order pair embeddings by their absolute distance in continuous affinity.

    For anchor i and reference j, the denominator contains every non-anchor k
    whose label is at least as far from i as j is. This is the RnC objective
    used by the reference implementation. ``high_weighted`` changes only the
    contribution of each anchor; it does not collapse all high-affinity labels
    into one class.
    """
    if mode not in ('standard', 'high_weighted'):
        raise ValueError(mode)
    features = features.float()
    labels = labels.float().reshape(-1)
    if features.ndim != 2 or features.size(0) != labels.numel():
        raise ValueError((features.shape, labels.shape))
    n = labels.numel()
    if n < 3:
        raise ValueError('Rank-N-Contrast requires at least three samples')
    if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
        raise ValueError('Non-finite features or labels')

    # The projection head already L2-normalizes, but normalize here as an
    # invariant so callers cannot accidentally change the distance scale.
    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    feature_distance = torch.cdist(features, features, p=2)
    logits = -feature_distance / float(temperature)
    label_distance = (labels[:, None] - labels[None, :]).abs()
    eye = torch.eye(n, dtype=torch.bool, device=features.device)
    denominator_mask = ((~eye)[:, None, :]
                        & (label_distance[:, None, :] >= label_distance[:, :, None]))
    denominator_logits = logits[:, None, :].expand(n, n, n).masked_fill(
        ~denominator_mask, -torch.inf)
    pair_loss = -logits + torch.logsumexp(denominator_logits, dim=-1)
    per_anchor = pair_loss.masked_fill(eye, 0.0).sum(1) / (n - 1)

    if mode == 'high_weighted':
        anchor_weight = 1.0 + float(high_strength) * torch.sigmoid(
            (labels - float(high_threshold)) / float(high_width))
        loss = (per_anchor * anchor_weight).sum() / anchor_weight.sum()
    else:
        anchor_weight = torch.ones_like(labels)
        loss = per_anchor.mean()

    # Diagnostics never participate in optimization.
    with torch.no_grad():
        j_closer = label_distance[:, :, None] < label_distance[:, None, :]
        valid = j_closer & (~eye)[:, :, None] & (~eye)[:, None, :]
        reference_distance = feature_distance[:, :, None].expand(n, n, n)
        candidate_distance = feature_distance[:, None, :].expand(n, n, n)
        order_accuracy = (reference_distance[valid]
                          < candidate_distance[valid]).float().mean()
        upper = torch.triu(torch.ones_like(eye), diagonal=1).bool()
        distance_alignment = _pearson(feature_distance[upper], label_distance[upper])
        diagnostics = {
            'order_accuracy': float(order_accuracy),
            'distance_alignment': float(distance_alignment),
            'feature_std': float(features.std(0, unbiased=False).mean()),
            'anchor_weight_mean': float(anchor_weight.mean()),
            'anchor_weight_max': float(anchor_weight.max()),
        }
    return loss, diagnostics


class StratifiedPairSampler:
    """Sample distinct training pairs with explicit high/mid-affinity coverage."""
    def __init__(self, labels, batch_size=32, high_count=8, mid_count=8,
                 high_threshold=7.0, mid_threshold=5.0):
        self.labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        if not np.isfinite(self.labels).all():
            raise ValueError('Sampler labels must be finite')
        if len(self.labels) < batch_size:
            raise ValueError('Not enough distinct training pairs')
        self.batch_size = int(batch_size)
        self.high_count = int(high_count)
        self.mid_count = int(mid_count)
        self.high = np.flatnonzero(self.labels >= high_threshold)
        self.mid = np.flatnonzero((self.labels > mid_threshold)
                                  & (self.labels < high_threshold))
        self.all = np.arange(len(self.labels))

    @staticmethod
    def _draw(pool, count, selected):
        available = np.setdiff1d(pool, np.asarray(selected, dtype=np.int64),
                                 assume_unique=False)
        take = min(int(count), len(available))
        if not take:
            return []
        return np.random.choice(available, size=take, replace=False).tolist()

    def sample(self):
        selected = self._draw(self.high, self.high_count, [])
        selected += self._draw(self.mid, self.mid_count, selected)
        selected += self._draw(self.all, self.batch_size - len(selected), selected)
        if len(selected) != self.batch_size or len(set(selected)) != self.batch_size:
            raise RuntimeError('Failed to construct a distinct RNC batch')
        np.random.shuffle(selected)
        return np.asarray(selected, dtype=np.int64)

    def audit(self):
        return {'pairs': int(len(self.labels)), 'high_pool': int(len(self.high)),
                'mid_pool': int(len(self.mid)), 'batch_size': self.batch_size,
                'high_quota': self.high_count, 'mid_quota': self.mid_count}
