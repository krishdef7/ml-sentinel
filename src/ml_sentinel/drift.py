"""
Drift Detection Methods
=======================

Per-feature input drift scores (KS, PSI, Wasserstein, chi-square) and a
multivariate one (MMD).

These measure how far P(X) moved. They say nothing directly about whether
accuracy dropped. See experiments/folktables_benchmark.py for a measured
comparison against label-free accuracy estimates.

Every per-feature score has two forms: ``*_scores`` returns one value per
feature, and ``*_drift`` returns the maximum, which is what a "max per-feature"
alarm thresholds.
"""

from typing import Literal, Optional

import numpy as np
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance


def _as_2d(X: np.ndarray) -> np.ndarray:
    """Treat a 1-D array as a single feature column."""
    X = np.asarray(X)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def ks_scores(X_ref: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Two-sample Kolmogorov-Smirnov statistic for each feature.

    The statistic is the largest gap between the two empirical CDFs (0 to 1).
    It is used as an effect size: unlike the p-value, it does not go to zero
    just because the samples are large.
    """
    X_ref, X_test = _as_2d(X_ref), _as_2d(X_test)
    return np.array([ks_2samp(X_ref[:, i], X_test[:, i])[0] for i in range(X_ref.shape[1])])


def ks_drift(X_ref: np.ndarray, X_test: np.ndarray) -> float:
    """Maximum per-feature KS statistic."""
    return float(np.max(ks_scores(X_ref, X_test)))


def psi_scores(X_ref: np.ndarray, X_test: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Population Stability Index for each feature.

    Bins are the deciles of the reference sample. For a feature with few
    distinct values several deciles coincide; the repeated edges give empty
    bins, which add (almost) nothing. Each bin count gets +1 smoothing so
    empty bins do not produce log(0).

    PSI = sum over bins of (p_test - p_ref) * ln(p_test / p_ref).
    Common rule of thumb (credit scoring): < 0.1 stable, 0.1-0.25 moderate,
    > 0.25 significant shift.

    Note: categorical features coded as integers are binned as if the codes
    were ordered. That is the usual practice for a quick PSI, but it is crude
    for nominal codes such as occupation or place of birth.
    """
    X_ref, X_test = _as_2d(X_ref), _as_2d(X_test)
    scores = []
    for i in range(X_ref.shape[1]):
        edges = np.percentile(X_ref[:, i], np.linspace(0, 100, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts, _ = np.histogram(X_ref[:, i], bins=edges)
        test_counts, _ = np.histogram(X_test[:, i], bins=edges)
        p_ref = (ref_counts + 1) / (len(X_ref) + n_bins)
        p_test = (test_counts + 1) / (len(X_test) + n_bins)
        scores.append(np.sum((p_test - p_ref) * np.log(p_test / p_ref)))
    return np.array(scores)


def psi_drift(X_ref: np.ndarray, X_test: np.ndarray, n_bins: int = 10) -> float:
    """Maximum per-feature PSI."""
    return float(np.max(psi_scores(X_ref, X_test, n_bins)))


def wasserstein_scores(X_ref: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """1-D Wasserstein (earth mover's) distance for each feature, in feature units."""
    X_ref, X_test = _as_2d(X_ref), _as_2d(X_test)
    return np.array(
        [wasserstein_distance(X_ref[:, i], X_test[:, i]) for i in range(X_ref.shape[1])]
    )


def wasserstein_drift(X_ref: np.ndarray, X_test: np.ndarray) -> float:
    """
    Maximum per-feature Wasserstein distance.

    Not scale-free: standardize features first if they have different units.
    """
    return float(np.max(wasserstein_scores(X_ref, X_test)))


def chi2_scores(X_ref: np.ndarray, X_test: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Chi-square statistic per feature (on pooled-decile bins), divided by its
    degrees of freedom.
    """
    X_ref, X_test = _as_2d(X_ref), _as_2d(X_test)
    scores = []
    for i in range(X_ref.shape[1]):
        combined = np.concatenate([X_ref[:, i], X_test[:, i]])
        edges = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts, _ = np.histogram(X_ref[:, i], bins=edges)
        test_counts, _ = np.histogram(X_test[:, i], bins=edges)
        table = np.array([ref_counts, test_counts])
        table = table[:, table.sum(axis=0) > 0]  # drop empty bins
        dof = table.shape[1] - 1
        if dof < 1:
            scores.append(0.0)
            continue
        stat = chi2_contingency(table)[0]
        scores.append(stat / dof)
    return np.array(scores)


def chi2_drift(X_ref: np.ndarray, X_test: np.ndarray, n_bins: int = 10) -> float:
    """Maximum per-feature chi-square statistic (per degree of freedom)."""
    return float(np.max(chi2_scores(X_ref, X_test, n_bins)))


def mmd_drift(
    X_ref: np.ndarray,
    X_test: np.ndarray,
    kernel: Literal["rbf", "linear"] = "rbf",
    gamma: Optional[float] = None,
    n_samples: int = 1000,
    random_state: Optional[int] = 0,
) -> float:
    """
    Unbiased squared Maximum Mean Discrepancy between the two samples.

    Unlike the per-feature scores, MMD compares the joint distribution of all
    features. Each sample is subsampled to ``n_samples`` rows (the kernel
    matrices are quadratic in size); ``random_state`` makes that reproducible.
    Features should be standardized first; ``gamma`` defaults to 1 / n_features.
    """
    X_ref, X_test = _as_2d(X_ref).astype(float), _as_2d(X_test).astype(float)
    rng = np.random.default_rng(random_state)
    if len(X_ref) > n_samples:
        X_ref = X_ref[rng.choice(len(X_ref), n_samples, replace=False)]
    if len(X_test) > n_samples:
        X_test = X_test[rng.choice(len(X_test), n_samples, replace=False)]
    if gamma is None:
        gamma = 1.0 / X_ref.shape[1]

    def k(A, B):
        if kernel == "linear":
            return A @ B.T
        sq = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * sq)

    n, m = len(X_ref), len(X_test)
    K_xx, K_yy, K_xy = k(X_ref, X_ref), k(X_test, X_test), k(X_ref, X_test)
    mmd2 = (
        (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
        + (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
        - 2 * K_xy.sum() / (n * m)
    )
    return float(max(0.0, mmd2))


class DriftDetector:
    """
    Threshold one of the drift scores.

    Parameters
    ----------
    method : str
        'ks', 'psi', 'wasserstein', 'chi2' or 'mmd'
    threshold : float, optional
        Alert when the score is above this. Defaults: KS 0.10 and PSI 0.25
        (the values used in the benchmark); the others are placeholders that
        depend on feature scaling and should be set for your data.

    Example
    -------
    >>> detector = DriftDetector(method="psi")
    >>> detector.score(X_train, X_prod)
    >>> detector.alert(X_train, X_prod)
    """

    METHODS = {
        "ks": ks_drift,
        "psi": psi_drift,
        "wasserstein": wasserstein_drift,
        "chi2": chi2_drift,
        "mmd": mmd_drift,
    }

    DEFAULT_THRESHOLDS = {
        "ks": 0.10,
        "psi": 0.25,
        "wasserstein": 0.1,
        "chi2": 3.0,
        "mmd": 0.01,
    }

    def __init__(self, method: str = "ks", threshold: Optional[float] = None, **kwargs):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHODS)}")
        self.method = method
        self.threshold = self.DEFAULT_THRESHOLDS[method] if threshold is None else threshold
        self.kwargs = kwargs
        self._drift_fn = self.METHODS[method]

    def score(self, X_ref: np.ndarray, X_test: np.ndarray) -> float:
        """Drift score (higher = more drift)."""
        return self._drift_fn(X_ref, X_test, **self.kwargs)

    def alert(self, X_ref: np.ndarray, X_test: np.ndarray) -> bool:
        """True if the drift score is above the threshold."""
        return bool(self.score(X_ref, X_test) > self.threshold)

    def __repr__(self):
        return f"DriftDetector(method='{self.method}', threshold={self.threshold})"
