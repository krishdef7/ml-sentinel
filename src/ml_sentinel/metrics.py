"""
Evaluation Metrics
==================

Statistical metrics for evaluating drift detection vs sentinel monitoring.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from typing import Tuple, Optional, List
import warnings


def compute_correlation(
    scores: np.ndarray, degradation: np.ndarray, method: str = "pearson"
) -> Tuple[float, float]:
    """
    Compute correlation between scores and degradation.

    Used to compare how well each monitoring signal tracks the accuracy drop.

    Parameters
    ----------
    scores : np.ndarray
        Detection scores (drift or sentinel)
    degradation : np.ndarray
        Actual accuracy degradation
    method : str
        'pearson' or 'spearman'

    Returns
    -------
    r : float
        Correlation coefficient
    p : float
        p-value

    Example
    -------
    >>> r, p = compute_correlation(scores, acc_drops, method='spearman')
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
    Compute ROC-AUC and curve.

    Parameters
    ----------
    scores : np.ndarray
        Detection scores
    labels : np.ndarray
        Binary labels (1 = degraded)

    Returns
    -------
    auc : float
        Area under ROC curve
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    thresholds : np.ndarray
        Decision thresholds
    """
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(auc), fpr, tpr, thresholds


def compute_tpr_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float = 0.15
) -> Tuple[float, float]:
    """
    Compute TPR and FPR at a fixed threshold.


    Parameters
    ----------
    scores : np.ndarray
        Detection scores
    labels : np.ndarray
        Binary labels (1 = degraded)
    threshold : float
        Decision threshold

    Returns
    -------
    tpr : float
        True Positive Rate (sensitivity)
    fpr : float
        False Positive Rate (1 - specificity)

    Example
    -------
    >>> tpr, fpr = compute_tpr_at_threshold(drift_scores, is_degraded, 0.15)
    """
    predictions = scores > threshold
    positives = labels == 1
    negatives = labels == 0

    tp = np.sum(predictions & positives)
    fp = np.sum(predictions & negatives)
    fn = np.sum(~predictions & positives)
    tn = np.sum(~predictions & negatives)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return float(tpr), float(fpr)


def bootstrap_ci(
    scores: np.ndarray,
    degradation: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    method: str = "pearson",
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for correlation.

    Parameters
    ----------
    scores : np.ndarray
        Detection scores
    degradation : np.ndarray
        Actual accuracy degradation
    n_bootstrap : int
        Number of bootstrap iterations
    ci : float
        Confidence level (0.95 = 95%)
    method : str
        Correlation method ('pearson' or 'spearman')

    Returns
    -------
    r : float
        Point estimate of correlation
    ci_lower : float
        Lower bound of CI
    ci_upper : float
        Upper bound of CI

    Example
    -------
    >>> r, lo, hi = bootstrap_ci(sentinel_scores, acc_drops)
    >>> print(f"r = {r:.2f} [{lo:.2f}, {hi:.2f}]")
    >>> # r = 0.90 [0.85, 0.94]
    """
    n = len(scores)
    bootstrap_rs = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        r, _ = compute_correlation(scores[idx], degradation[idx], method)
        if not np.isnan(r):
            bootstrap_rs.append(r)

    bootstrap_rs = np.array(bootstrap_rs)
    r_point, _ = compute_correlation(scores, degradation, method)

    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_rs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_rs, 100 * (1 - alpha / 2))

    return float(r_point), float(ci_lower), float(ci_upper)


def compute_threshold_gap(
    drift_scores: np.ndarray, is_degraded: np.ndarray, standard_threshold: float = 0.15
) -> dict:
    """
    Compute the gap between degraded scenario drift and detection threshold.

    Reports how far degraded scenarios' drift scores sit from the threshold.

    Parameters
    ----------
    drift_scores : np.ndarray
        Drift scores for each scenario
    is_degraded : np.ndarray
        Binary: is this scenario degraded?
    standard_threshold : float
        Industry-standard threshold

    Returns
    -------
    dict
        Gap analysis results:
        - max_drift: Maximum drift among degraded scenarios
        - mean_drift: Mean drift among degraded scenarios
        - gap_ratio: threshold / max_drift (how many × below threshold)

    Example
    -------
    >>> gap = compute_threshold_gap(drift_scores, is_degraded)
    >>> print(f"Gap ratio: {gap['gap_ratio']:.1f}×")
    >>> # CivilComments: 33× below threshold!
    """
    degraded_drift = drift_scores[is_degraded.astype(bool)]

    if len(degraded_drift) == 0:
        return {
            "n_degraded": 0,
            "max_drift": np.nan,
            "mean_drift": np.nan,
            "gap_ratio": np.nan,
        }

    max_drift = float(np.max(degraded_drift))
    mean_drift = float(np.mean(degraded_drift))
    gap_ratio = standard_threshold / max_drift if max_drift > 0 else np.inf

    return {
        "n_degraded": int(np.sum(is_degraded)),
        "max_drift": max_drift,
        "mean_drift": mean_drift,
        "min_drift": float(np.min(degraded_drift)),
        "standard_threshold": standard_threshold,
        "gap_ratio": float(gap_ratio),
    }


def compute_all_metrics(
    drift_scores: np.ndarray,
    sentinel_scores: np.ndarray,
    accuracy_drops: np.ndarray,
    is_degraded: np.ndarray,
    threshold: float = 0.15,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Compute all evaluation metrics for a benchmark.


    Parameters
    ----------
    drift_scores : np.ndarray
        KS drift scores
    sentinel_scores : np.ndarray
        Sentinel confidence scores
    accuracy_drops : np.ndarray
        Actual accuracy degradation
    is_degraded : np.ndarray
        Binary: significant degradation?
    threshold : float
        Drift detection threshold
    n_bootstrap : int
        Bootstrap iterations for CI

    Returns
    -------
    dict
        Comprehensive metrics including:
        - Correlations with CI
        - ROC-AUC comparison
        - TPR/FPR at threshold
        - Threshold gap analysis
    """
    # Correlation analysis
    r_drift, p_drift = compute_correlation(drift_scores, accuracy_drops)
    r_sent, p_sent = compute_correlation(sentinel_scores, accuracy_drops)

    # Bootstrap CIs
    r_drift_ci = bootstrap_ci(drift_scores, accuracy_drops, n_bootstrap)
    r_sent_ci = bootstrap_ci(sentinel_scores, accuracy_drops, n_bootstrap)

    # ROC-AUC
    auc_drift, _, _, _ = compute_roc_auc(drift_scores, is_degraded)
    auc_sent, _, _, _ = compute_roc_auc(sentinel_scores, is_degraded)

    # TPR at threshold
    tpr_drift, fpr_drift = compute_tpr_at_threshold(drift_scores, is_degraded, threshold)

    # Threshold gap
    gap = compute_threshold_gap(drift_scores, is_degraded, threshold)

    return {
        "correlation": {
            "drift": {
                "r": r_drift,
                "p": p_drift,
                "ci_lower": r_drift_ci[1],
                "ci_upper": r_drift_ci[2],
            },
            "sentinel": {
                "r": r_sent,
                "p": p_sent,
                "ci_lower": r_sent_ci[1],
                "ci_upper": r_sent_ci[2],
            },
        },
        "roc_auc": {
            "drift": auc_drift,
            "sentinel": auc_sent,
        },
        "at_threshold": {
            "threshold": threshold,
            "drift_tpr": tpr_drift,
            "drift_fpr": fpr_drift,
        },
        "threshold_gap": gap,
    }
