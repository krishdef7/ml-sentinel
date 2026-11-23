"""
Sentinel Monitoring
===================

Label-free accuracy monitoring from model confidence.

For a calibrated model, E[max P(Y|X)] = accuracy. Under shift the model can
become over- or under-confident, so two estimators are provided:
- SentinelMonitor: drop in mean max-probability (average confidence).
- atc_threshold / atc_accuracy: Average Thresholded Confidence
  (Garg et al., "Leveraging Unlabeled Data to Predict Out-of-Distribution
  Performance", ICLR 2022).
"""

from typing import Any, Callable, Optional

import numpy as np
from scipy.special import softmax


def compute_confidence_score(predictions: np.ndarray, method: str = "max_prob") -> float:
    """
    Compute model confidence score from predictions.

    This is the core of Sentinel monitoring - tracking model confidence
    as a proxy for accuracy without requiring ground truth labels.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions. Can be:
        - Probabilities: shape (n_samples, n_classes)
        - Logits: shape (n_samples, n_classes)
        - Binary probabilities: shape (n_samples,)
    method : str
        Confidence metric:
        - 'max_prob': Maximum predicted probability (default)
        - 'entropy': Negative prediction entropy
        - 'margin': Difference between top 2 predictions

    Returns
    -------
    float
        Mean confidence score (higher = more confident = better expected accuracy)

    Theory
    ------
    For a well-calibrated classifier:
        E[max_y P(Y=y|X)] = Accuracy(f)

    This provides a theoretical foundation for using confidence as
    an accuracy proxy.
    """
    predictions = np.atleast_2d(predictions)

    # Handle single column (binary classification)
    if predictions.shape[1] == 1 or predictions.ndim == 1:
        predictions = predictions.flatten()
        predictions = np.column_stack([1 - predictions, predictions])

    # Ensure probabilities (apply softmax if logits)
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = softmax(predictions, axis=1)

    if method == "max_prob":
        # Expected accuracy for calibrated model
        confidence = np.max(predictions, axis=1)
    elif method == "entropy":
        # Lower entropy = higher confidence
        eps = 1e-10
        entropy = -np.sum(predictions * np.log(predictions + eps), axis=1)
        max_entropy = np.log(predictions.shape[1])
        confidence = 1 - (entropy / max_entropy)
    elif method == "margin":
        # Difference between top 2 probabilities
        sorted_probs = np.sort(predictions, axis=1)
        confidence = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(np.mean(confidence))


class SentinelMonitor:
    """
    Sentinel monitoring for ML model degradation detection.

    Tracks the drop in mean confidence relative to an in-distribution baseline.

    Parameters
    ----------
    method : str
        Confidence computation method ('max_prob', 'entropy', 'margin')
    baseline_score : float, optional
        Reference confidence score (computed from validation set)
    threshold : float
        Alert threshold for confidence drop (default: 0.05)

    Example
    -------
    >>> sentinel = SentinelMonitor()
    >>>
    >>> # Compute baseline on validation set
    >>> baseline = sentinel.compute_baseline(model, X_val)
    >>>
    >>> # Monitor production data
    >>> score = sentinel.score(model, X_test)
    >>> drop = baseline - score
    >>>
    >>> # Check for degradation
    >>> if sentinel.alert(model, X_test):
    ...     print("Model degradation detected!")
    """

    def __init__(
        self,
        method: str = "max_prob",
        baseline_score: Optional[float] = None,
        threshold: float = 0.05,
    ):
        self.method = method
        self.baseline_score = baseline_score
        self.threshold = threshold

    def compute_baseline(
        self, model: Any, X: np.ndarray, predict_fn: Optional[Callable] = None
    ) -> float:
        """
        Compute baseline confidence from reference data.

        Parameters
        ----------
        model : Any
            Trained model with predict_proba or predict method
        X : np.ndarray
            Reference data (validation or training holdout)
        predict_fn : Callable, optional
            Custom prediction function

        Returns
        -------
        float
            Baseline confidence score
        """
        predictions = self._get_predictions(model, X, predict_fn)
        self.baseline_score = compute_confidence_score(predictions, self.method)
        return self.baseline_score

    def score(self, model: Any, X: np.ndarray, predict_fn: Optional[Callable] = None) -> float:
        """
        Compute confidence score for given data.

        Parameters
        ----------
        model : Any
            Trained model
        X : np.ndarray
            Data to evaluate
        predict_fn : Callable, optional
            Custom prediction function

        Returns
        -------
        float
            Confidence score (higher = better expected accuracy)
        """
        predictions = self._get_predictions(model, X, predict_fn)
        return compute_confidence_score(predictions, self.method)

    def score_from_predictions(self, predictions: np.ndarray) -> float:
        """
        Compute confidence score directly from predictions.

        Useful when predictions are already computed.

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions (probabilities or logits)

        Returns
        -------
        float
            Confidence score
        """
        return compute_confidence_score(predictions, self.method)

    def alert(self, model: Any, X: np.ndarray, predict_fn: Optional[Callable] = None) -> bool:
        """
        Check if model degradation is detected.

        Parameters
        ----------
        model : Any
            Trained model
        X : np.ndarray
            Data to evaluate
        predict_fn : Callable, optional
            Custom prediction function

        Returns
        -------
        bool
            True if confidence drop exceeds threshold

        Raises
        ------
        ValueError
            If baseline_score hasn't been computed
        """
        if self.baseline_score is None:
            raise ValueError(
                "Baseline score not set. Call compute_baseline() first "
                "or provide baseline_score in constructor."
            )

        current_score = self.score(model, X, predict_fn)
        drop = self.baseline_score - current_score
        return drop > self.threshold

    def degradation_estimate(
        self, model: Any, X: np.ndarray, predict_fn: Optional[Callable] = None
    ) -> float:
        """
        Estimate accuracy degradation without ground truth.

        For well-calibrated models, the confidence drop closely
        approximates the actual accuracy drop.

        Parameters
        ----------
        model : Any
            Trained model
        X : np.ndarray
            Data to evaluate
        predict_fn : Callable, optional
            Custom prediction function

        Returns
        -------
        float
            Estimated accuracy drop (positive = degradation)
        """
        if self.baseline_score is None:
            raise ValueError("Baseline score not set.")

        current_score = self.score(model, X, predict_fn)
        return self.baseline_score - current_score

    def _get_predictions(
        self, model: Any, X: np.ndarray, predict_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """Get model predictions."""
        if predict_fn is not None:
            return predict_fn(X)

        # Try common prediction methods
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        elif hasattr(model, "predict"):
            return model.predict(X)
        else:
            raise ValueError(
                "Model must have predict_proba or predict method, "
                "or provide a custom predict_fn."
            )

    def __repr__(self):
        return (
            f"SentinelMonitor(method='{self.method}', "
            f"baseline={self.baseline_score}, threshold={self.threshold})"
        )


def atc_threshold(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Fit the ATC confidence threshold on labeled in-distribution data.

    Returns t such that the fraction of samples with confidence <= t equals
    the error rate on this data. Equivalently, the fraction above t equals the
    accuracy, so ``atc_accuracy(predictions, t)`` reproduces the source
    accuracy by construction. On new data, ATC reports the fraction of rows
    above the same t.

    This uses max-probability as the confidence score; Garg et al. also use
    negative entropy.
    """
    predictions = np.atleast_2d(predictions)
    if predictions.shape[1] == 1:
        predictions = np.column_stack([1 - predictions, predictions])
    confidences = np.max(predictions, axis=1)
    accuracy = np.mean(np.argmax(predictions, axis=1) == labels)
    return float(np.quantile(confidences, 1.0 - accuracy))


def atc_accuracy(predictions: np.ndarray, threshold: float) -> float:
    """Estimate accuracy on unlabeled data: fraction with confidence above the ATC threshold."""
    predictions = np.atleast_2d(predictions)
    if predictions.shape[1] == 1:
        predictions = np.column_stack([1 - predictions, predictions])
    return float(np.mean(np.max(predictions, axis=1) > threshold))


def compute_sentinel_scores(predictions_list: list, method: str = "max_prob") -> np.ndarray:
    """
    Compute sentinel scores for multiple prediction sets.

    Useful for batch analysis across multiple scenarios.

    Parameters
    ----------
    predictions_list : list
        List of prediction arrays
    method : str
        Confidence method

    Returns
    -------
    np.ndarray
        Array of confidence scores
    """
    return np.array([compute_confidence_score(p, method) for p in predictions_list])


def calibration_error(predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Lower ECE means confidence scores better predict accuracy,
    making Sentinel monitoring more effective.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (probabilities)
    labels : np.ndarray
        True labels
    n_bins : int
        Number of confidence bins

    Returns
    -------
    float
        Expected Calibration Error (0 to 1)
    """
    predictions = np.atleast_2d(predictions)
    if predictions.shape[1] == 1:
        predictions = np.column_stack([1 - predictions, predictions])

    confidences = np.max(predictions, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracies = (predicted_labels == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += in_bin.sum() * np.abs(avg_accuracy - avg_confidence)

    return ece / len(labels)
