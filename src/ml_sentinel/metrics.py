"""
Evaluation Metrics
==================

Helpers for scoring a monitoring signal against the real accuracy drop.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def compute_correlation(
    scores: np.ndarray, degradation: np.ndarray, method: str = "pearson"
) -> Tuple[float, float]:
    """
    Correlation between a monitoring signal and the actual accuracy drop.

    Parameters
    ----------
    scores : np.ndarray
        One signal value per deployment (drift score, estimated drop, ...)
    degradation : np.ndarray
        Actual accuracy drop per deployment
    method : str
        'pearson' or 'spearman'

    Returns
    -------
    (r, p) : correlation coefficient and its p-value
    """
    if method == "pearson":
        r, p = stats.pearsonr(scores, degradation)
    elif method == "spearman":
        r, p = stats.spearmanr(scores, degradation)
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(r), float(p)


def compute_roc_auc(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    AUROC of a signal for separating degraded (label 1) from healthy (0)
    deployments, plus the ROC curve. AUROC is the probability that a random
    degraded deployment scores higher than a random healthy one.

    Returns
    -------
    (auc, fpr, tpr, thresholds)
    """
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(auc), fpr, tpr, thresholds


def compute_tpr_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Tuple[float, float]:
    """
    True and false positive rate of the alarm ``scores > threshold``.

    Parameters
    ----------
    scores : np.ndarray
        Signal value per deployment
    labels : np.ndarray
        1 = degraded, 0 = healthy
    threshold : float
        Alarm threshold

    Returns
    -------
    (tpr, fpr)
    """
    alarm = scores > threshold
    pos, neg = labels == 1, labels == 0
    tpr = np.sum(alarm & pos) / np.sum(pos) if np.sum(pos) > 0 else 0.0
    fpr = np.sum(alarm & neg) / np.sum(neg) if np.sum(neg) > 0 else 0.0
    return float(tpr), float(fpr)


def bootstrap_ci(
    scores: np.ndarray,
    degradation: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    method: str = "pearson",
    random_state: Optional[int] = 0,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval for a correlation.

    Resamples (score, degradation) pairs with replacement.

    Returns
    -------
    (r, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(random_state)
    n = len(scores)
    rs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        r, _ = compute_correlation(scores[idx], degradation[idx], method)
        if not np.isnan(r):
            rs.append(r)
    r_point, _ = compute_correlation(scores, degradation, method)
    alpha = 1 - ci
    lo, hi = np.percentile(rs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(r_point), float(lo), float(hi)
