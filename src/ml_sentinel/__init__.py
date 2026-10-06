"""
ML Sentinel: label-free monitoring of model accuracy under distribution shift.

Input-drift scores (KS, PSI, ...) measure how much P(X) moved. Confidence-based
estimators (average confidence, ATC) estimate how much accuracy moved. The
benchmark in experiments/folktables_benchmark.py compares the two on 147
cross-state deployments of ACS (folktables) models.

>>> from ml_sentinel import ks_drift, SentinelMonitor, atc_threshold, atc_accuracy
>>> t = atc_threshold(model.predict_proba(X_val), y_val)
>>> est_acc = atc_accuracy(model.predict_proba(X_prod), t)
"""

__version__ = "1.0.0"
__author__ = "Krish Garg"

from .drift import DriftDetector, ks_drift, wasserstein_drift, psi_drift, mmd_drift
from .sentinel import SentinelMonitor, atc_accuracy, atc_threshold, compute_confidence_score
from .metrics import (
    compute_correlation,
    compute_roc_auc,
    compute_tpr_at_threshold,
    bootstrap_ci,
)

__all__ = [
    # Input drift
    "DriftDetector",
    "ks_drift",
    "wasserstein_drift",
    "psi_drift",
    "mmd_drift",
    # Label-free accuracy estimation
    "SentinelMonitor",
    "compute_confidence_score",
    "atc_threshold",
    "atc_accuracy",
    # Metrics
    "compute_correlation",
    "compute_roc_auc",
    "compute_tpr_at_threshold",
    "bootstrap_ci",
]
