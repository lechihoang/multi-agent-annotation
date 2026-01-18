"""
Human Review Queue Management.

Priority-based queue for items requiring human review.
"""

from collections import deque
from datetime import datetime
from typing import Any, Deque
from uuid import UUID

import structlog

from ..config.settings import settings
from ..models import (
    AnnotationResult,
    DecisionType,
    ReviewItem,
    ReviewStatus,
    Task,
)

logger = structlog.get_logger()


class ReviewQueue:
    """
    Priority-based queue for human review.

    Queue structure:
    - Escalated: Low confidence items (score < 0.60) - highest priority
    - Medium: Medium confidence items (0.60-0.85) - normal priority
    - Spot Check: Random sampling for quality assurance - lowest priority

    Items are processed in priority order: escalated > medium > spot_check
    """

    def __init__(
        self,
        max_size: int | None = None,
        spot_check_rate: float = 0.05,
    ):
        """
        Initialize review queue.

        Args:
            max_size: Maximum queue size (default from settings)
            spot_check_rate: Rate of auto-approved items to spot check
        """
        self.max_size = max_size or settings.max_queue_size
        self.spot_check_rate = spot_check_rate

        # Priority queues
        self.escalated: Deque[ReviewItem] = deque()
        self.medium: Deque[ReviewItem] = deque()
        self.spot_check: Deque[ReviewItem] = deque()

        # Index for quick lookup
        self._index: dict[UUID, ReviewItem] = {}

        # Statistics
        self._stats = {
            "total_added": 0,
            "total_processed": 0,
            "escalated_count": 0,
            "medium_count": 0,
            "spot_check_count": 0,
        }

    @property
    def size(self) -> int:
        """Total number of items in queue."""
        return len(self.escalated) + len(self.medium) + len(self.spot_check)

    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return self.size >= self.max_size

    def add(
        self,
        task: Task,
        annotation_result: AnnotationResult,
        priority: int | None = None,
    ) -> ReviewItem | None:
        """
        Add item to appropriate queue based on consensus score.

        Args:
            task: Original task
            annotation_result: Annotation result from judge
            priority: Override priority (optional)

        Returns:
            ReviewItem if added, None if queue full
        """
        if self.is_full:
            logger.warning("review_queue_full", size=self.size, max=self.max_size)
            return None

        # Determine priority based on consensus
        consensus = annotation_result.consensus
        score = consensus.score if consensus else 0.0
        decision = consensus.decision if consensus else DecisionType.ESCALATE

        # Calculate priority score
        if priority is None:
            priority = self._calculate_priority(score, decision, task)

        item = ReviewItem(
            task=task,
            annotation_result=annotation_result,
            consensus_score=score,
            priority=priority,
            status=ReviewStatus.PENDING,
        )

        # Route to appropriate queue
        if decision == DecisionType.ESCALATE or score < 0.60:
            self.escalated.append(item)
            self._stats["escalated_count"] += 1
        elif decision == DecisionType.REVIEW:
            self.medium.append(item)
            self._stats["medium_count"] += 1
        else:
            # Spot check for auto-approved items
            self.spot_check.append(item)
            self._stats["spot_check_count"] += 1

        self._index[item.id] = item
        self._stats["total_added"] += 1

        logger.info(
            "item_added_to_review_queue",
            item_id=str(item.id),
            score=round(score, 4),
            decision=decision.value if decision else "unknown",
            queue_size=self.size,
        )

        return item

    def _calculate_priority(
        self,
        score: float,
        decision: DecisionType,
        task: Task,
    ) -> int:
        """
        Calculate priority score for queue ordering.

        Lower score = higher priority.

        Args:
            score: Consensus score
            decision: Routing decision
            task: Original task

        Returns:
            Priority score (0-1000)
        """
        base_priority = 500

        # Adjust by decision type
        if decision == DecisionType.ESCALATE:
            base_priority = 100  # Highest priority
        elif decision == DecisionType.REVIEW:
            base_priority = 300

        # Adjust by confidence (lower confidence = higher priority)
        confidence_adjustment = int((1 - score) * 200)

        # Adjust by age (older = higher priority)
        age_seconds = (datetime.utcnow() - task.created_at).total_seconds()
        age_adjustment = min(int(age_seconds / 3600), 100)  # Max 100 points for age

        return base_priority - confidence_adjustment - age_adjustment

    def get_next(self) -> ReviewItem | None:
        """
        Get next item for review (highest priority first).

        Returns:
            Next ReviewItem or None if queue empty
        """
        item = None

        # Check queues in priority order
        if self.escalated:
            item = self.escalated.popleft()
        elif self.medium:
            item = self.medium.popleft()
        elif self.spot_check:
            item = self.spot_check.popleft()

        if item:
            item.status = ReviewStatus.IN_PROGRESS
            self._stats["total_processed"] += 1

            logger.info(
                "item_retrieved_for_review",
                item_id=str(item.id),
                priority=item.priority,
            )

        return item

    def get_batch(self, size: int | None = None) -> list[ReviewItem]:
        """
        Get batch of items for review session.

        Args:
            size: Batch size (default from settings)

        Returns:
            List of ReviewItems
        """
        size = size or settings.review_batch_size
        items = []

        for _ in range(size):
            item = self.get_next()
            if item:
                items.append(item)
            else:
                break

        return items

    def get_by_id(self, item_id: UUID) -> ReviewItem | None:
        """Get item by ID."""
        return self._index.get(item_id)

    def return_to_queue(self, item: ReviewItem) -> None:
        """
        Return item to queue (e.g., if reviewer can't complete).

        Args:
            item: Item to return
        """
        item.status = ReviewStatus.PENDING
        item.assigned_to = None

        # Re-add based on original consensus
        score = item.consensus_score
        if score < 0.60:
            self.escalated.appendleft(item)
        elif score < 0.85:
            self.medium.appendleft(item)
        else:
            self.spot_check.appendleft(item)

        logger.info("item_returned_to_queue", item_id=str(item.id))

    def remove(self, item_id: UUID) -> bool:
        """
        Remove item from queue.

        Args:
            item_id: ID of item to remove

        Returns:
            True if removed, False if not found
        """
        if item_id not in self._index:
            return False

        item = self._index.pop(item_id)

        # Remove from appropriate queue
        for queue in [self.escalated, self.medium, self.spot_check]:
            try:
                queue.remove(item)
                break
            except ValueError:
                continue

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "total_size": self.size,
            "max_size": self.max_size,
            "escalated_pending": len(self.escalated),
            "medium_pending": len(self.medium),
            "spot_check_pending": len(self.spot_check),
            **self._stats,
        }

    def get_items_by_status(self, status: ReviewStatus) -> list[ReviewItem]:
        """Get all items with a specific status."""
        return [
            item for item in self._index.values()
            if item.status == status
        ]

    def clear(self) -> int:
        """
        Clear all items from queue.

        Returns:
            Number of items cleared
        """
        count = self.size
        self.escalated.clear()
        self.medium.clear()
        self.spot_check.clear()
        self._index.clear()

        logger.info("review_queue_cleared", items_cleared=count)
        return count


# Singleton instance
_review_queue: ReviewQueue | None = None


def get_review_queue() -> ReviewQueue:
    """Get the singleton review queue instance."""
    global _review_queue
    if _review_queue is None:
        _review_queue = ReviewQueue()
    return _review_queue
