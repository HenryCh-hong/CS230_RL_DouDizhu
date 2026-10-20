"""Predictor metrics and the two baselines every predictor is compared with.

A belief is (..., 15, 5): for each card kind, P(the hand holds 0..4 copies). Both baselines use only
what the acting player knows, its count u of each unseen kind and the public hand sizes:

- legal-uniform: every feasible count 0..u equally likely, impossible counts exactly 0
  (for u = 2: [1/3, 1/3, 1/3, 0, 0]); it knows the feasible range and nothing else;
- hypergeometric card counting: the hand is a uniformly random subset of the unseen cards of the
  public size, P(k) = C(u, k) C(N - u, n - k) / C(N, n): the exact posterior if the cards had been
  dealt at random and play revealed nothing beyond what was played.

CE is the mean over samples and the 15 kinds of -log P(true count), in nats, as in DouZero+: lower
is better, and kinds with no unseen copy contribute 0 to every model alike. MAE is the mean
absolute error of the expected count. For two-hand predictors, the conservation violation is the
mean over kinds of |E[next] + E[other] - u| (in cards), and reflect_tv the mean over kinds of the
total-variation distance between the learned p_other and reflect(p_next).
"""
import math

import numpy as np
import torch

from douzero.env import hand_belief as hb

PHASES = ('early', 'mid', 'late')
HANDS = ('next', 'other')
_LOG_FACTORIAL = np.array([math.lgamma(n + 1) for n in range(64)])


def _log_comb(n, k):
    valid = (k >= 0) & (k <= n)
    n_, k_ = np.where(valid, n, 0), np.where(valid, k, 0)
    return np.where(valid, _LOG_FACTORIAL[n_] - _LOG_FACTORIAL[k_] - _LOG_FACTORIAL[n_ - k_], -np.inf)


def uniform_probs(union):
    """(..., 15) unseen counts -> (..., 15, 5) legal-uniform beliefs."""
    mask = hb.legal_count_mask(np.asarray(union)).astype(np.float64)
    return mask / mask.sum(axis=-1, keepdims=True)


def hypergeometric_probs(union, size):
    """(B, 15) unseen counts and (B,) hand sizes -> (B, 15, 5) card-counting beliefs (vectorized)."""
    union = np.asarray(union, dtype=np.int64)[..., None]
    size = np.asarray(size, dtype=np.int64)[:, None, None]
    total = union.sum(axis=1, keepdims=True)
    k = np.arange(hb.NUM_COUNTS)
    log_p = _log_comb(union, k) + _log_comb(total - union, size - k) - _log_comb(total, size)
    with np.errstate(invalid='ignore'):  # -inf - x where a count is impossible
        return np.where(np.isfinite(log_p), np.exp(log_p), 0.0)


def nll(probs, labels):
    """(B, 15, 5) beliefs and (B, 15) true counts -> (B, 15) -log P(true count)."""
    p_true = np.take_along_axis(np.asarray(probs), np.asarray(labels, dtype=np.int64)[..., None], axis=-1)[..., 0]
    if (p_true <= 0).any():
        raise ValueError('a belief gives a true count zero probability')
    return -np.log(p_true)


def expected(probs):
    return np.asarray(probs) @ np.arange(hb.NUM_COUNTS)


def summarize(sample_nll, probs, labels, phase):
    """CE overall and by phase, and the expected-count MAE, of one belief over a set of samples."""
    per_sample = sample_nll.mean(axis=-1)
    by_phase = {name: float(per_sample[phase == i].mean()) if (phase == i).any() else float('nan')
                for i, name in enumerate(PHASES)}
    return dict(ce=float(per_sample.mean()), ce_by_phase=by_phase,
                mae=float(np.abs(expected(probs) - labels).mean()))


def baseline_report(arrays):
    """{'uniform' | 'hypergeometric': {'next' | 'other': summary}} over one seat's samples of one split."""
    union, sizes, labels, phase = arrays['union'], arrays['sizes'], arrays['labels'], arrays['phase']
    report = {}
    for name in ('uniform', 'hypergeometric'):
        report[name] = {}
        for h, hand in enumerate(HANDS):
            probs = uniform_probs(union) if name == 'uniform' else hypergeometric_probs(union, sizes[:, h])
            report[name][hand] = summarize(nll(probs, labels[:, h]), probs, labels[:, h], phase)
    return report


def predict(predictor, arrays, batch_size=1024):
    """The predictor's (B, K, 15, 5) log-probabilities over a whole split, without gradients."""
    out = []
    with torch.no_grad():
        for start in range(0, len(arrays['labels']), batch_size):
            z = torch.from_numpy(arrays['z'][start:start + batch_size]).float()
            x = torch.from_numpy(arrays['x_no_action'][start:start + batch_size]).float()
            out.append(torch.log_softmax(predictor(z, x)['logits'], dim=-1))
    return torch.cat(out).double().numpy()


def sample_ce(predictor, arrays):
    """Per-sample CE (mean over kinds, then over the predicted hands) of the model and of both baselines."""
    log_probs = predict(predictor, arrays)
    labels, union, sizes = arrays['labels'].astype(np.int64), arrays['union'], arrays['sizes']
    out = dict(model=[], uniform=[], hypergeometric=[])
    for h in range(predictor.num_targets):
        out['model'].append(-np.take_along_axis(log_probs[:, h], labels[:, h, :, None], axis=-1)[..., 0].mean(axis=-1))
        out['uniform'].append(nll(uniform_probs(union), labels[:, h]).mean(axis=-1))
        out['hypergeometric'].append(nll(hypergeometric_probs(union, sizes[:, h]), labels[:, h]).mean(axis=-1))
    return {name: np.mean(values, axis=0) for name, values in out.items()}


def clustered_ci(values, games, resamples=2000, seed=0):
    """95% bootstrap interval of the mean of values, resampling whole games: decisions of one game are correlated."""
    ids, index = np.unique(games, return_inverse=True)
    sums = np.bincount(index, weights=values, minlength=len(ids))
    counts = np.bincount(index, minlength=len(ids)).astype(np.float64)
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(resamples, len(ids)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def model_report(predictor, arrays):
    """The predictor's summary for each hand it predicts, plus the two-hand consistency measures."""
    log_probs = predict(predictor, arrays)
    labels, phase = arrays['labels'], arrays['phase']
    report = {}
    for h in range(predictor.num_targets):
        target = labels[:, h]
        sample_nll = -np.take_along_axis(log_probs[:, h], target.astype(np.int64)[..., None], axis=-1)[..., 0]
        report[HANDS[h]] = summarize(sample_nll, np.exp(log_probs[:, h]), target, phase)
    if predictor.num_targets == 2:
        p_next, p_other = np.exp(log_probs[:, 0]), np.exp(log_probs[:, 1])
        union = arrays['union']
        report['conservation'] = float(np.abs(expected(p_next) + expected(p_other) - union).mean())
        report['reflect_tv'] = float(0.5 * np.abs(p_other - hb.reflect_belief(p_next, union)).sum(axis=-1).mean())
    return report
