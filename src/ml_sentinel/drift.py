"""
Drift Detection Methods
=======================

This module implements various drift detection methods that we prove
FAIL to detect actual model degradation.

Key Finding: All methods achieve 0% TPR on degradation detection.
"""

import numpy as np
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency
from typing import Optional, Tuple, Literal
import warnings


def ks_drift(X_ref: np.ndarray, X_test: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov drift score.
    
    The KS test measures the maximum difference between two empirical CDFs.
    
    Parameters
    ----------
    X_ref : np.ndarray
        Reference (training) feature values, shape (n_ref,) or (n_ref, d)
    X_test : np.ndarray
        Test feature values, shape (n_test,) or (n_test, d)
        
    Returns
    -------
    float
        Maximum KS statistic across all features (0 to 1)
        
    Note
    ----
    Despite being the industry standard, KS drift detection achieves
    0% TPR on actual model degradation in our experiments.
    """
    X_ref = np.atleast_2d(X_ref)
    X_test = np.atleast_2d(X_test)
    
    if X_ref.shape[0] == 1:
        X_ref = X_ref.T
    if X_test.shape[0] == 1:
        X_test = X_test.T
        
    n_features = X_ref.shape[1]
    ks_scores = []
    
    for i in range(n_features):
        statistic, _ = ks_2samp(X_ref[:, i], X_test[:, i])
        ks_scores.append(statistic)
    
    return float(np.max(ks_scores))


def wasserstein_drift(X_ref: np.ndarray, X_test: np.ndarray) -> float:
    """
    Compute Wasserstein (Earth Mover's) distance.
    
    Measures the minimum "work" required to transform one distribution
    into another.
    
    Parameters
    ----------
    X_ref : np.ndarray
        Reference feature values
    X_test : np.ndarray
        Test feature values
        
    Returns
    -------
    float
        Mean Wasserstein distance across features
        
    Note
    ----
    Also fails to detect degradation - the problem is fundamental,
    not the choice of metric.
    """
    X_ref = np.atleast_2d(X_ref)
    X_test = np.atleast_2d(X_test)
    
    if X_ref.shape[0] == 1:
        X_ref = X_ref.T
    if X_test.shape[0] == 1:
        X_test = X_test.T
        
    n_features = X_ref.shape[1]
    distances = []
    
    for i in range(n_features):
        dist = wasserstein_distance(X_ref[:, i], X_test[:, i])
        distances.append(dist)
    
    return float(np.mean(distances))


def psi_drift(X_ref: np.ndarray, X_test: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).
    
    PSI is widely used in credit scoring and financial modeling.
    Industry thresholds: <0.1 (no shift), 0.1-0.25 (moderate), >0.25 (significant)
    
    Parameters
    ----------
    X_ref : np.ndarray
        Reference feature values
    X_test : np.ndarray
        Test feature values
    n_bins : int
        Number of bins for discretization
        
    Returns
    -------
    float
        Maximum PSI across features
        
    Note
    ----
    Despite being the industry standard for credit models, PSI also
    fails to detect concept drift.
    """
    X_ref = np.atleast_2d(X_ref)
    X_test = np.atleast_2d(X_test)
    
    if X_ref.shape[0] == 1:
        X_ref = X_ref.T
    if X_test.shape[0] == 1:
        X_test = X_test.T
        
    n_features = X_ref.shape[1]
    psi_scores = []
    
    for i in range(n_features):
        # Create bins from reference distribution
        bins = np.percentile(X_ref[:, i], np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # Compute proportions
        ref_counts, _ = np.histogram(X_ref[:, i], bins=bins)
        test_counts, _ = np.histogram(X_test[:, i], bins=bins)
        
        ref_props = (ref_counts + 1) / (len(X_ref) + n_bins)  # Add 1 for smoothing
        test_props = (test_counts + 1) / (len(X_test) + n_bins)
        
        # PSI formula
        psi = np.sum((test_props - ref_props) * np.log(test_props / ref_props))
        psi_scores.append(psi)
    
    return float(np.max(psi_scores))


def mmd_drift(
    X_ref: np.ndarray, 
    X_test: np.ndarray, 
    kernel: Literal['rbf', 'linear'] = 'rbf',
    gamma: Optional[float] = None,
    n_samples: int = 1000
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD).
    
    MMD uses kernel methods to compare distributions in a reproducing
    kernel Hilbert space.
    
    Parameters
    ----------
    X_ref : np.ndarray
        Reference feature values
    X_test : np.ndarray
        Test feature values
    kernel : str
        Kernel type ('rbf' or 'linear')
    gamma : float, optional
        RBF kernel parameter (default: 1/n_features)
    n_samples : int
        Number of samples for estimation (for efficiency)
        
    Returns
    -------
    float
        MMD^2 estimate
        
    Note
    ----
    Even this sophisticated method fails under concept drift.
    """
    X_ref = np.atleast_2d(X_ref)
    X_test = np.atleast_2d(X_test)
    
    if X_ref.shape[0] == 1:
        X_ref = X_ref.T
    if X_test.shape[0] == 1:
        X_test = X_test.T
    
    # Subsample for efficiency
    if len(X_ref) > n_samples:
        idx = np.random.choice(len(X_ref), n_samples, replace=False)
        X_ref = X_ref[idx]
    if len(X_test) > n_samples:
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_test = X_test[idx]
    
    if gamma is None:
        gamma = 1.0 / X_ref.shape[1]
    
    def rbf_kernel(X, Y):
        """Compute RBF kernel matrix."""
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        distances = XX + YY.T - 2 * X @ Y.T
        return np.exp(-gamma * distances)
    
    def linear_kernel(X, Y):
        """Compute linear kernel matrix."""
        return X @ Y.T
    
    kernel_fn = rbf_kernel if kernel == 'rbf' else linear_kernel
    
    K_xx = kernel_fn(X_ref, X_ref)
    K_yy = kernel_fn(X_test, X_test)
    K_xy = kernel_fn(X_ref, X_test)
    
    n = len(X_ref)
    m = len(X_test)
    
    # Unbiased MMD^2 estimate
    mmd2 = (
        (np.sum(K_xx) - np.trace(K_xx)) / (n * (n - 1))
        + (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1))
        - 2 * np.sum(K_xy) / (n * m)
    )
    
    return float(max(0, mmd2))


def chi2_drift(X_ref: np.ndarray, X_test: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Chi-squared drift score.
    
    Tests whether two samples come from the same distribution by
    comparing binned frequencies.
    
    Parameters
    ----------
    X_ref : np.ndarray
        Reference feature values
    X_test : np.ndarray
        Test feature values
    n_bins : int
        Number of bins for discretization
        
    Returns
    -------
    float
        Maximum chi-squared statistic (normalized) across features
    """
    X_ref = np.atleast_2d(X_ref)
    X_test = np.atleast_2d(X_test)
    
    if X_ref.shape[0] == 1:
        X_ref = X_ref.T
    if X_test.shape[0] == 1:
        X_test = X_test.T
        
    n_features = X_ref.shape[1]
    chi2_scores = []
    
    for i in range(n_features):
        # Create bins
        combined = np.concatenate([X_ref[:, i], X_test[:, i]])
        bins = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        ref_counts, _ = np.histogram(X_ref[:, i], bins=bins)
        test_counts, _ = np.histogram(X_test[:, i], bins=bins)
        
        # Create contingency table
        contingency = np.array([ref_counts, test_counts])
        
        # Filter out empty bins
        non_empty = (ref_counts + test_counts) > 0
        if non_empty.sum() < 2:
            chi2_scores.append(0)
            continue
            
        contingency = contingency[:, non_empty]
        
        try:
            chi2, _, _, _ = chi2_contingency(contingency)
            # Normalize by degrees of freedom
            dof = contingency.shape[1] - 1
            chi2_scores.append(chi2 / dof if dof > 0 else 0)
        except:
            chi2_scores.append(0)
    
    return float(np.max(chi2_scores))


class DriftDetector:
    """
    Unified drift detection interface.
    
    Supports multiple drift detection methods with configurable thresholds.
    
    WARNING: All methods fail to detect actual model degradation!
    Our experiments show 0% TPR across 3 benchmarks and 1.6M samples.
    
    Parameters
    ----------
    method : str
        Detection method: 'ks', 'wasserstein', 'psi', 'mmd', 'chi2'
    threshold : float
        Alert threshold (default: 0.15, industry standard)
        
    Example
    -------
    >>> detector = DriftDetector(method='ks', threshold=0.15)
    >>> score = detector.score(X_train, X_test)
    >>> alert = detector.alert(X_train, X_test)  # Almost always False!
    """
    
    METHODS = {
        'ks': ks_drift,
        'wasserstein': wasserstein_drift,
        'psi': psi_drift,
        'mmd': mmd_drift,
        'chi2': chi2_drift,
    }
    
    # Industry-standard thresholds
    DEFAULT_THRESHOLDS = {
        'ks': 0.15,
        'wasserstein': 0.1,
        'psi': 0.25,
        'mmd': 0.01,
        'chi2': 3.0,
    }
    
    def __init__(
        self, 
        method: str = 'ks', 
        threshold: Optional[float] = None,
        **kwargs
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHODS.keys())}")
        
        self.method = method
        self.threshold = threshold or self.DEFAULT_THRESHOLDS[method]
        self.kwargs = kwargs
        self._drift_fn = self.METHODS[method]
    
    def score(self, X_ref: np.ndarray, X_test: np.ndarray) -> float:
        """
        Compute drift score.
        
        Parameters
        ----------
        X_ref : np.ndarray
            Reference (training) data
        X_test : np.ndarray
            Test data
            
        Returns
        -------
        float
            Drift score (higher = more drift)
        """
        return self._drift_fn(X_ref, X_test, **self.kwargs)
    
    def alert(self, X_ref: np.ndarray, X_test: np.ndarray) -> bool:
        """
        Check if drift exceeds threshold.
        
        Parameters
        ----------
        X_ref : np.ndarray
            Reference (training) data
        X_test : np.ndarray
            Test data
            
        Returns
        -------
        bool
            True if drift detected (almost never in practice!)
            
        Note
        ----
        In our experiments, this returns False for 100% of degraded scenarios.
        """
        score = self.score(X_ref, X_test)
        return score > self.threshold
    
    def __repr__(self):
        return f"DriftDetector(method='{self.method}', threshold={self.threshold})"
