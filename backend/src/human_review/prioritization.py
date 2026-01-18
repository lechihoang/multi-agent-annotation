"""
Review Item Prioritization.

Calculates priority scores for review queue items.
"""

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from ..models import AnnotationResult, DecisionType, ReviewItem, Task

logger = structlog.get_logger()


@dataclass
class PriorityFactors:
    """Factors contributing to priority calculation."""

    consensus_score: float
    decision_type: str
    age_hours: float
    conflict_severity: float
    business_importance: float
    final_priority: int


class PriorityScorer:
    """
    Calculates priority scores for review items.

    Priority factors:
    1. Consensus score (lower = higher priority)
    2. Decision type (escalate > review > spot_check)
    3. Age (older = higher priority)
    4. Conflict severity (more conflicts = higher priority)
    5. Business importance (configurable per task type)
    """

    # Priority weights
    WEIGHTS = {
        "consensus": 0.35,
        "decision": 0.25,
        "age": 0.15,
        "conflicts": 0.15,
        "importance": 0.10,
    }

    # Decision type base scores (lower = higher priority)
    DECISION_SCORES = {
        DecisionType.ESCALATE: 100,
        DecisionType.REVIEW: 300,
        DecisionType.APPROVE: 500,  # For spot checks
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        max_age_hours: float = 24.0,
    ):
        """
        Initialize priority scorer.

        Args:
            weights: Custom priority weights
            max_age_hours: Maximum age for age-based priority
        """
        self.weights = weights or self.WEIGHTS
        self.max_age_hours = max_age_hours

    def calculate_priority(
        self,
        item: ReviewItem,
    ) -> int:
        """
        Calculate priority score for a review item.

        Lower score = higher priority (processed first).

        Args:
            item: ReviewItem to score

        Returns:
            Priority score (0-1000)
        """
        factors = self._calculate_factors(item)

        # Combine factors into final score
        consensus_component = (1 - factors.consensus_score) * 400
        decision_component = self.DECISION_SCORES.get(
            DecisionType(item.annotation_result.consensus.decision)
            if item.annotation_result.consensus
            else DecisionType.REVIEW,
            300,
        )
        age_component = min(factors.age_hours / self.max_age_hours, 1.0) * 100
        conflict_component = factors.conflict_severity * 100
        importance_component = (1 - factors.business_importance) * 100

        weighted_score = (
            self.weights["consensus"] * consensus_component
            + self.weights["decision"] * (1000 - decision_component)
            + self.weights["age"] * (100 - age_component)
            + self.weights["conflicts"] * conflict_component
            + self.weights["importance"] * importance_component
        )

        # Invert and scale to 0-1000 (lower = higher priority)
        final_priority = max(0, min(1000, int(1000 - weighted_score)))

        logger.debug(
            "priority_calculated",
            item_id=str(item.id),
            priority=final_priority,
            consensus_score=round(factors.consensus_score, 4),
            age_hours=round(factors.age_hours, 2),
        )

        return final_priority

    def _calculate_factors(self, item: ReviewItem) -> PriorityFactors:
        """Calculate individual priority factors."""
        # Consensus score
        consensus_score = item.consensus_score

        # Decision type
        decision = (
            item.annotation_result.consensus.decision
            if item.annotation_result.consensus
            else DecisionType.REVIEW
        )

        # Age in hours
        age = datetime.utcnow() - item.created_at
        age_hours = age.total_seconds() / 3600

        # Conflict severity (from audit trail)
        conflict_severity = 0.0
        if item.annotation_result.consensus:
            audit = item.annotation_result.consensus.audit_trail
            conflicts = audit.get("conflicts", [])
            if conflicts:
                conflict_severity = min(len(conflicts) * 0.2, 1.0)

        # Business importance (placeholder - could be extended)
        business_importance = self._calculate_business_importance(item.task)

        return PriorityFactors(
            consensus_score=consensus_score,
            decision_type=decision.value,
            age_hours=age_hours,
            conflict_severity=conflict_severity,
            business_importance=business_importance,
            final_priority=0,  # Will be set by calculate_priority
        )

    def _calculate_business_importance(self, task: Task) -> float:
        """
        Calculate business importance score.

        This is a placeholder that can be extended based on:
        - Task metadata
        - Customer tier
        - Domain/category
        - etc.

        Args:
            task: Task to evaluate

        Returns:
            Importance score (0.0 to 1.0)
        """
        # Check for importance hints in metadata
        metadata = task.metadata or {}

        if metadata.get("high_priority"):
            return 1.0
        if metadata.get("vip_customer"):
            return 0.9
        if metadata.get("urgent"):
            return 0.8

        return 0.5  # Default importance

    def should_spot_check(
        self,
        item: ReviewItem,
        spot_check_rate: float = 0.05,
    ) -> bool:
        """
        Determine if an auto-approved item should be spot-checked.

        Args:
            item: ReviewItem (auto-approved)
            spot_check_rate: Percentage to spot check

        Returns:
            True if should be spot-checked
        """
        # Only spot check approved items
        if item.annotation_result.consensus:
            if item.annotation_result.consensus.decision != DecisionType.APPROVE:
                return False

        # Random sampling
        if random.random() < spot_check_rate:
            logger.info(
                "spot_check_selected",
                item_id=str(item.id),
                rate=spot_check_rate,
            )
            return True

        return False

    def reprioritize(
        self,
        items: list[ReviewItem],
    ) -> list[ReviewItem]:
        """
        Recalculate priorities and sort items.

        Args:
            items: List of items to reprioritize

        Returns:
            Sorted list (highest priority first)
        """
        for item in items:
            item.priority = self.calculate_priority(item)

        sorted_items = sorted(items, key=lambda x: x.priority)

        logger.info(
            "items_reprioritized",
            count=len(items),
            top_priority=sorted_items[0].priority if sorted_items else None,
        )

        return sorted_items


# Singleton instance
_priority_scorer: PriorityScorer | None = None


def get_priority_scorer() -> PriorityScorer:
    """Get the singleton priority scorer instance."""
    global _priority_scorer
    if _priority_scorer is None:
        _priority_scorer = PriorityScorer()
    return _priority_scorer
