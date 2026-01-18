"""
Confidence aggregation and calibration utilities.

Provides methods for combining confidence scores and calibrating
agent outputs based on historical performance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from ..models import Annotation

logger = structlog.get_logger()


@dataclass
class CalibrationData:
    """Historical calibration data for an agent."""

    agent_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    confidence_sum: float = 0.0
    accuracy: float = 0.0
    calibration_factor: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ConfidenceAggregator:
    """
    Aggregates and calibrates confidence scores from multiple agents.

    Supports:
    - Weighted averaging
    - Min/max/mean aggregation
    - Historical calibration
    - Uncertainty estimation
    """

    def __init__(self):
        """Initialize confidence aggregator."""
        self._calibration_data: dict[str, CalibrationData] = {}

    def aggregate(
        self,
        scores: list[float],
        weights: list[float] | None = None,
        method: str = "weighted_mean",
    ) -> float:
        """
        Aggregate multiple confidence scores.

        Args:
            scores: List of confidence scores
            weights: Optional weights for each score
            method: Aggregation method (weighted_mean, mean, min, max, harmonic)

        Returns:
            Aggregated confidence score
        """
        if not scores:
            return 0.0

        if weights is None:
            weights = [1.0 / len(scores)] * len(scores)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        if method == "weighted_mean":
            return sum(s * w for s, w in zip(scores, weights))

        elif method == "mean":
            return sum(scores) / len(scores)

        elif method == "min":
            return min(scores)

        elif method == "max":
            return max(scores)

        elif method == "harmonic":
            # Harmonic mean - penalizes low values more
            if any(s == 0 for s in scores):
                return 0.0
            return len(scores) / sum(1.0 / s for s in scores)

        elif method == "geometric":
            # Geometric mean
            import math

            if any(s <= 0 for s in scores):
                return 0.0
            return math.exp(sum(math.log(s) for s in scores) / len(scores))

        else:
            logger.warning(f"unknown_aggregation_method: {method}, using weighted_mean")
            return sum(s * w for s, w in zip(scores, weights))

    def calibrate(
        self,
        raw_score: float,
        agent_name: str,
    ) -> float:
        """
        Calibrate a raw confidence score based on historical accuracy.

        Args:
            raw_score: Raw confidence from agent
            agent_name: Name of the agent

        Returns:
            Calibrated confidence score
        """
        calibration = self._calibration_data.get(agent_name)

        if calibration is None or calibration.total_predictions < 10:
            # Not enough data for calibration
            return raw_score

        # Apply calibration factor
        calibrated = raw_score * calibration.calibration_factor

        # Clamp to valid range
        return max(0.0, min(1.0, calibrated))

    def update_calibration(
        self,
        agent_name: str,
        predicted_confidence: float,
        was_correct: bool,
    ) -> None:
        """
        Update calibration data based on feedback.

        Args:
            agent_name: Name of the agent
            predicted_confidence: The confidence score that was predicted
            was_correct: Whether the prediction was correct
        """
        if agent_name not in self._calibration_data:
            self._calibration_data[agent_name] = CalibrationData(agent_name=agent_name)

        data = self._calibration_data[agent_name]
        data.total_predictions += 1
        data.confidence_sum += predicted_confidence

        if was_correct:
            data.correct_predictions += 1

        # Update accuracy
        data.accuracy = data.correct_predictions / data.total_predictions

        # Calculate calibration factor
        # If agent is overconfident, factor < 1
        # If agent is underconfident, factor > 1
        avg_confidence = data.confidence_sum / data.total_predictions
        if avg_confidence > 0:
            data.calibration_factor = data.accuracy / avg_confidence
        else:
            data.calibration_factor = 1.0

        data.last_updated = datetime.utcnow()

        logger.debug(
            "calibration_updated",
            agent=agent_name,
            accuracy=round(data.accuracy, 4),
            calibration_factor=round(data.calibration_factor, 4),
        )

    def detect_low_confidence(
        self,
        annotations: list[Annotation],
        threshold: float = 0.5,
    ) -> bool:
        """
        Detect if any agent has critically low confidence.

        Args:
            annotations: List of annotations
            threshold: Confidence threshold for low detection

        Returns:
            True if low confidence detected
        """
        for annotation in annotations:
            if annotation.confidence < threshold:
                logger.info(
                    "low_confidence_detected",
                    agent=annotation.agent_name,
                    confidence=round(annotation.confidence, 4),
                    threshold=threshold,
                )
                return True
        return False

    def detect_high_variance(
        self,
        annotations: list[Annotation],
        variance_threshold: float = 0.1,
    ) -> bool:
        """
        Detect if agents have highly varying confidence scores.

        High variance often indicates ambiguous or difficult tasks.

        Args:
            annotations: List of annotations
            variance_threshold: Variance threshold

        Returns:
            True if high variance detected
        """
        if len(annotations) < 2:
            return False

        scores = [a.confidence for a in annotations]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)

        if variance > variance_threshold:
            logger.info(
                "high_variance_detected",
                variance=round(variance, 4),
                threshold=variance_threshold,
                scores=[round(s, 4) for s in scores],
            )
            return True
        return False

    def estimate_uncertainty(
        self,
        annotations: list[Annotation],
    ) -> dict[str, float]:
        """
        Estimate uncertainty in the consensus.

        Args:
            annotations: List of annotations

        Returns:
            Dict with uncertainty metrics
        """
        if not annotations:
            return {
                "variance": 0.0,
                "range": 0.0,
                "entropy": 0.0,
                "agreement": 0.0,
            }

        scores = [a.confidence for a in annotations]
        mean = sum(scores) / len(scores)

        # Variance
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)

        # Range
        score_range = max(scores) - min(scores)

        # Simple entropy approximation
        import math

        entropy = 0.0
        for s in scores:
            if 0 < s < 1:
                entropy -= s * math.log(s) + (1 - s) * math.log(1 - s)
        entropy /= len(scores)

        # Agreement score (inverse of variance)
        agreement = max(0.0, 1.0 - variance)

        return {
            "variance": round(variance, 4),
            "range": round(score_range, 4),
            "entropy": round(entropy, 4),
            "agreement": round(agreement, 4),
        }

    def get_calibration_stats(self) -> dict[str, Any]:
        """Get calibration statistics for all agents."""
        return {
            agent: {
                "total_predictions": data.total_predictions,
                "accuracy": round(data.accuracy, 4),
                "calibration_factor": round(data.calibration_factor, 4),
                "last_updated": data.last_updated.isoformat(),
            }
            for agent, data in self._calibration_data.items()
        }


# Singleton instance
_confidence_aggregator: ConfidenceAggregator | None = None


def get_confidence_aggregator() -> ConfidenceAggregator:
    """Get the singleton confidence aggregator instance."""
    global _confidence_aggregator
    if _confidence_aggregator is None:
        _confidence_aggregator = ConfidenceAggregator()
    return _confidence_aggregator
