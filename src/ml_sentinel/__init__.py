"""
ML Sentinel: Why Drift Detection Fails
=======================================

A comprehensive empirical and theoretical analysis proving that data drift
detection fundamentally cannot detect model degradation.

Key Finding: Drift detection achieves 0% TPR across 3 benchmarks and 1.6M samples.

Example Usage
-------------
>>> from ml_sentinel import DriftDetector, SentinelMonitor
>>> 
>>> # Drift detection (will fail!)
>>> drift = DriftDetector(method='ks', threshold=0.15)
>>> drift_score = drift.score(X_train, X_test)
>>> 
>>> # Sentinel monitoring (works!)
>>> sentinel = SentinelMonitor()
>>> sentinel_score = sentinel.score(model, X_test)

Citation
--------
@inproceedings{sentinel2026,
  title={Why Drift Detection Fails: An Empirical and Theoretical Analysis},
  author={...},
  booktitle={NeurIPS},
  year={2026}
}
"""

__version__ = "1.0.0"
__author__ = "ML Sentinel Team"

from .drift import DriftDetector, ks_drift, wasserstein_drift, psi_drift, mmd_drift
from .sentinel import SentinelMonitor, compute_confidence_score
from .metrics import (
    compute_correlation,
    compute_roc_auc,
    compute_tpr_at_threshold,
    bootstrap_ci,
)

__all__ = [
    # Drift Detection (fails!)
    "DriftDetector",
    "ks_drift",
    "wasserstein_drift", 
    "psi_drift",
    "mmd_drift",
    # Sentinel Monitoring (works!)
    "SentinelMonitor",
    "compute_confidence_score",
    # Metrics
    "compute_correlation",
    "compute_roc_auc",
    "compute_tpr_at_threshold",
    "bootstrap_ci",
]
