"""
Tests for ML Sentinel metrics module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_sentinel import (
    DriftDetector,
    SentinelMonitor,
    compute_correlation,
    compute_roc_auc,
    compute_tpr_at_threshold,
    bootstrap_ci,
    ks_drift,
    compute_confidence_score,
)


class TestDriftDetection:
    """Tests for drift detection methods."""

    def test_ks_drift_identical(self):
        """Identical distributions should have low drift."""
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        drift = ks_drift(X, X)
        assert drift < 0.1

    def test_ks_drift_different(self):
        """Different distributions should have high drift."""
        np.random.seed(42)
        X_ref = np.random.randn(1000, 5)
        X_test = np.random.randn(1000, 5) + 2  # Shifted
        drift = ks_drift(X_ref, X_test)
        assert drift > 0.5

    def test_drift_detector_class(self):
        """Test DriftDetector class interface."""
        detector = DriftDetector(method="ks", threshold=0.15)

        np.random.seed(42)
        X_ref = np.random.randn(100, 3)
        X_test = np.random.randn(100, 3)

        score = detector.score(X_ref, X_test)
        assert 0 <= score <= 1

        alert = detector.alert(X_ref, X_test)
        assert isinstance(alert, bool)

    def test_drift_methods(self):
        """Test all drift methods work."""
        np.random.seed(42)
        X_ref = np.random.randn(100, 3)
        X_test = np.random.randn(100, 3)

        for method in ["ks", "wasserstein", "psi", "mmd", "chi2"]:
            detector = DriftDetector(method=method)
            score = detector.score(X_ref, X_test)
            assert isinstance(score, float)


class TestSentinelMonitoring:
    """Tests for sentinel monitoring."""

    def test_confidence_score_probabilities(self):
        """Test confidence computation from probabilities."""
        # High confidence predictions
        probs_high = np.array(
            [
                [0.1, 0.9],
                [0.05, 0.95],
                [0.2, 0.8],
            ]
        )
        score_high = compute_confidence_score(probs_high)

        # Low confidence predictions
        probs_low = np.array(
            [
                [0.5, 0.5],
                [0.45, 0.55],
                [0.48, 0.52],
            ]
        )
        score_low = compute_confidence_score(probs_low)

        assert score_high > score_low

    def test_confidence_methods(self):
        """Test different confidence computation methods."""
        np.random.seed(42)
        probs = np.random.dirichlet([1, 1, 1], size=100)

        for method in ["max_prob", "entropy", "margin"]:
            score = compute_confidence_score(probs, method=method)
            assert 0 <= score <= 1

    def test_sentinel_monitor_class(self):
        """Test SentinelMonitor class interface."""
        monitor = SentinelMonitor(method="max_prob", threshold=0.05)

        # Test score_from_predictions
        probs = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
        score = monitor.score_from_predictions(probs)
        assert 0 <= score <= 1


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_compute_correlation(self):
        """Test correlation computation."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.1  # Highly correlated

        r, p = compute_correlation(x, y)
        assert r > 0.9
        assert p < 0.01

    def test_compute_roc_auc(self):
        """Test ROC-AUC computation."""
        np.random.seed(42)

        # Good classifier
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.15])
        labels = np.array([1, 1, 1, 0, 0, 0])

        auc, fpr, tpr, thresholds = compute_roc_auc(scores, labels)
        assert auc > 0.9
        assert len(fpr) == len(tpr)

    def test_tpr_at_threshold(self):
        """Test TPR/FPR at threshold computation."""
        scores = np.array([0.2, 0.18, 0.12, 0.05, 0.03])
        labels = np.array([1, 1, 1, 0, 0])  # First 3 are degraded

        # At threshold 0.15, should detect 2/3 (scores 0.2, 0.18)
        tpr, fpr = compute_tpr_at_threshold(scores, labels, threshold=0.15)
        assert tpr == pytest.approx(2 / 3)
        assert fpr == 0.0

    def test_bootstrap_ci(self):
        """Test bootstrap confidence intervals."""
        np.random.seed(42)
        x = np.random.randn(30)
        y = x * 0.9 + np.random.randn(30) * 0.1

        r, ci_low, ci_high = bootstrap_ci(x, y, n_bootstrap=100)

        assert ci_low < r < ci_high
        assert ci_low > 0  # Should be significantly positive


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_analysis_pipeline(self):
        """Test the complete analysis workflow."""
        np.random.seed(42)

        # Simulate concept drift scenario
        n_scenarios = 20

        # Under concept drift: drift stays low, accuracy varies
        drift_scores = np.random.uniform(0.02, 0.08, n_scenarios)
        accuracy_drops = np.random.uniform(-0.02, 0.12, n_scenarios)
        sentinel_scores = accuracy_drops + np.random.normal(0, 0.01, n_scenarios)
        is_degraded = accuracy_drops > 0.03

        # Drift should NOT correlate well with degradation
        r_drift, _ = compute_correlation(drift_scores, accuracy_drops)

        # Sentinel SHOULD correlate well
        r_sent, _ = compute_correlation(sentinel_scores, accuracy_drops)

        assert r_sent > r_drift

        # TPR at standard threshold should be low for drift
        tpr, _ = compute_tpr_at_threshold(drift_scores, is_degraded, 0.15)
        assert tpr < 0.1  # Less than 10% detected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestATC:
    """Tests for Average Thresholded Confidence."""

    def test_atc_recovers_in_distribution_accuracy(self):
        from ml_sentinel import atc_accuracy, atc_threshold

        rng = np.random.default_rng(0)
        p1 = rng.uniform(0, 1, 5000)
        probs = np.column_stack([1 - p1, p1])
        labels = (rng.uniform(0, 1, 5000) < p1).astype(int)  # calibrated by construction
        t = atc_threshold(probs, labels)
        true_acc = np.mean(probs.argmax(1) == labels)
        assert abs(atc_accuracy(probs, t) - true_acc) < 0.01

    def test_atc_drops_when_confidence_drops(self):
        from ml_sentinel import atc_accuracy, atc_threshold

        rng = np.random.default_rng(1)
        p1 = rng.uniform(0, 1, 5000)
        probs = np.column_stack([1 - p1, p1])
        labels = (rng.uniform(0, 1, 5000) < p1).astype(int)
        t = atc_threshold(probs, labels)
        shifted = 0.5 + 0.5 * (probs - 0.5)  # same argmax, less confident
        assert atc_accuracy(shifted, t) < atc_accuracy(probs, t)
