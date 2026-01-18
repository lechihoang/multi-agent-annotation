"""
Threshold management for routing decisions.

Defines and manages thresholds for auto-approval, human review, and escalation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from ..config.settings import settings
from ..models import Annotation, ConsensusResult, DecisionType

logger = structlog.get_logger()


class RoutingAction(str, Enum):
    """Actions that can be taken based on threshold evaluation."""

    AUTO_APPROVE = "auto_approve"
    HUMAN_REVIEW = "human_review"
    EXPERT_ESCALATION = "expert_escalation"
    REPROCESS = "reprocess"


@dataclass
class ThresholdConfig:
    """Configuration for a threshold level."""

    name: str
    min_score: float
    max_score: float
    action: RoutingAction
    description: str


class ThresholdManager:
    """
    Manages threshold rules for routing decisions.

    Default Thresholds:
    - Auto-approve: score >= 0.85
    - Human review: 0.60 <= score < 0.85
    - Expert escalation: score < 0.60
    """

    def __init__(
        self,
        auto_approve: float | None = None,
        human_review: float | None = None,
    ):
        """
        Initialize threshold manager.

        Args:
            auto_approve: Threshold for auto-approval (default: 0.85)
            human_review: Threshold for human review (default: 0.60)
        """
        self.auto_approve_threshold = auto_approve or settings.auto_approve_threshold
        self.human_review_threshold = human_review or settings.human_review_threshold

        # Validate thresholds
        if self.human_review_threshold >= self.auto_approve_threshold:
            raise ValueError(
                f"human_review_threshold ({self.human_review_threshold}) must be "
                f"less than auto_approve_threshold ({self.auto_approve_threshold})"
            )

        # Define threshold configurations
        self._thresholds: list[ThresholdConfig] = [
            ThresholdConfig(
                name="auto_approve",
                min_score=self.auto_approve_threshold,
                max_score=1.0,
                action=RoutingAction.AUTO_APPROVE,
                description="High confidence - automatically approved",
            ),
            ThresholdConfig(
                name="human_review",
                min_score=self.human_review_threshold,
                max_score=self.auto_approve_threshold,
                action=RoutingAction.HUMAN_REVIEW,
                description="Medium confidence - requires human review",
            ),
            ThresholdConfig(
                name="expert_escalation",
                min_score=0.0,
                max_score=self.human_review_threshold,
                action=RoutingAction.EXPERT_ESCALATION,
                description="Low confidence - requires expert attention",
            ),
        ]

    def apply_threshold(self, score: float) -> DecisionType:
        """
        Apply threshold rules to get decision.

        Args:
            score: Consensus score (0.0 to 1.0)

        Returns:
            Decision type (APPROVE, REVIEW, or ESCALATE)
        """
        if score >= self.auto_approve_threshold:
            return DecisionType.APPROVE
        elif score >= self.human_review_threshold:
            return DecisionType.REVIEW
        else:
            return DecisionType.ESCALATE

    def get_action(self, result: ConsensusResult) -> RoutingAction:
        """
        Get routing action based on consensus result.

        Args:
            result: ConsensusResult from voting

        Returns:
            RoutingAction to take
        """
        score = result.score

        for threshold in self._thresholds:
            if threshold.min_score <= score < threshold.max_score or (
                threshold.max_score == 1.0 and score == 1.0
            ):
                logger.info(
                    "threshold_matched",
                    score=round(score, 4),
                    threshold=threshold.name,
                    action=threshold.action.value,
                )
                return threshold.action

        # Default to escalation
        return RoutingAction.EXPERT_ESCALATION

    def get_threshold_info(self, score: float) -> dict[str, Any]:
        """
        Get detailed information about which threshold a score falls into.

        Args:
            score: Consensus score

        Returns:
            Dict with threshold details
        """
        for threshold in self._thresholds:
            if threshold.min_score <= score < threshold.max_score or (
                threshold.max_score == 1.0 and score == 1.0
            ):
                return {
                    "threshold_name": threshold.name,
                    "action": threshold.action.value,
                    "description": threshold.description,
                    "score_range": {
                        "min": threshold.min_score,
                        "max": threshold.max_score,
                    },
                    "distance_to_next": (
                        round(threshold.max_score - score, 4)
                        if threshold.max_score < 1.0
                        else None
                    ),
                }

        return {"threshold_name": "unknown", "action": "escalate"}

    def should_flag_for_review(
        self,
        consensus: ConsensusResult,
        annotations: list[Annotation],
        conflict_threshold: float = 0.3,
    ) -> tuple[bool, list[str]]:
        """
        Determine if result should be flagged for review.

        May flag even high-confidence results if agents disagree significantly.

        Args:
            consensus: ConsensusResult
            annotations: List of annotations
            conflict_threshold: Threshold for detecting conflicts

        Returns:
            Tuple of (should_flag, reasons)
        """
        reasons: list[str] = []

        # Check score-based threshold
        if consensus.decision in [DecisionType.REVIEW, DecisionType.ESCALATE]:
            reasons.append(f"Score below auto-approve threshold ({consensus.score:.2f})")

        # Check for agent conflicts
        if len(annotations) >= 2:
            confidences = [a.confidence for a in annotations]
            max_diff = max(confidences) - min(confidences)

            if max_diff >= conflict_threshold:
                reasons.append(
                    f"High agent disagreement (confidence diff: {max_diff:.2f})"
                )

        # Check for any very low confidence
        low_confidence_agents = [
            a.agent_name for a in annotations if a.confidence < 0.4
        ]
        if low_confidence_agents:
            reasons.append(
                f"Low confidence from: {', '.join(low_confidence_agents)}"
            )

        should_flag = len(reasons) > 0

        if should_flag:
            logger.info(
                "flagged_for_review",
                score=round(consensus.score, 4),
                reasons=reasons,
            )

        return should_flag, reasons

    def adjust_thresholds(
        self,
        auto_approve: float | None = None,
        human_review: float | None = None,
    ) -> None:
        """
        Dynamically adjust thresholds.

        Args:
            auto_approve: New auto-approve threshold
            human_review: New human review threshold
        """
        if auto_approve is not None:
            self.auto_approve_threshold = auto_approve

        if human_review is not None:
            self.human_review_threshold = human_review

        # Validate
        if self.human_review_threshold >= self.auto_approve_threshold:
            raise ValueError("Invalid threshold configuration")

        # Rebuild threshold configs
        self._thresholds = [
            ThresholdConfig(
                name="auto_approve",
                min_score=self.auto_approve_threshold,
                max_score=1.0,
                action=RoutingAction.AUTO_APPROVE,
                description="High confidence - automatically approved",
            ),
            ThresholdConfig(
                name="human_review",
                min_score=self.human_review_threshold,
                max_score=self.auto_approve_threshold,
                action=RoutingAction.HUMAN_REVIEW,
                description="Medium confidence - requires human review",
            ),
            ThresholdConfig(
                name="expert_escalation",
                min_score=0.0,
                max_score=self.human_review_threshold,
                action=RoutingAction.EXPERT_ESCALATION,
                description="Low confidence - requires expert attention",
            ),
        ]

        logger.info(
            "thresholds_adjusted",
            auto_approve=self.auto_approve_threshold,
            human_review=self.human_review_threshold,
        )

    def get_config(self) -> dict[str, Any]:
        """Get current threshold configuration."""
        return {
            "auto_approve": self.auto_approve_threshold,
            "human_review": self.human_review_threshold,
            "escalate": 0.0,
            "thresholds": [
                {
                    "name": t.name,
                    "min_score": t.min_score,
                    "max_score": t.max_score,
                    "action": t.action.value,
                }
                for t in self._thresholds
            ],
        }


# Singleton instance
_threshold_manager: ThresholdManager | None = None


def get_threshold_manager() -> ThresholdManager:
    """Get the singleton threshold manager instance."""
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = ThresholdManager()
    return _threshold_manager
