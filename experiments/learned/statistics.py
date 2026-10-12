"""Paired uncertainty with the independently trained seed as experimental unit."""
import numpy as np


def paired_interval(left, right, *, confidence=.95, bootstrap_samples=20000, seed=917):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or len(left) < 3:
        raise ValueError('Require at least three aligned independent seed summaries')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Non-finite measurements cannot be analyzed')
    differences = left - right
    indices = np.random.default_rng(seed).integers(0, len(left), (bootstrap_samples, len(left)))
    means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, [(1-confidence)/2, 1-(1-confidence)/2])
    return {'n_independent_seeds': len(left), 'paired_mean': float(differences.mean()),
            'lower': float(lower), 'upper': float(upper), 'confidence': confidence,
            'experimental_unit': 'independently initialized and trained seed',
            'method': 'paired bootstrap of seed means', 'bootstrap_samples': bootstrap_samples}


def classification(interval, *, direction, minimum_effect, equivalence_margin=None):
    if direction == 'positive' and interval['lower'] > minimum_effect:
        return 'supports_local_prediction'
    if direction == 'negative' and interval['upper'] < -minimum_effect:
        return 'supports_local_prediction'
    if equivalence_margin is not None and interval['lower'] >= -equivalence_margin and interval['upper'] <= equivalence_margin:
        return 'equivalent_within_preregistered_margin'
    if (direction == 'positive' and interval['upper'] < 0) or (direction == 'negative' and interval['lower'] > 0):
        return 'opposite_to_prediction'
    return 'inconclusive'
