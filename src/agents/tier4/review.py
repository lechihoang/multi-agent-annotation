

from dataclasses import dataclass, field
from typing import List, Deque, Dict, Any
from collections import deque
import heapq
import time


@dataclass(order=True)
class ReviewItem:
    priority: int
    task_id: str = field(compare=False)
    original_text: str = field(compare=False)
    annotation: Dict[str, Any] = field(compare=False)
    consensus_score: float = field(compare=False)
    created_at: float = field(compare=False)
    decision: str = field(compare=False)


class ReviewQueue:

    PRIORITY_ESCALATED = 1
    PRIORITY_REVIEW = 2
    PRIORITY_SPOT_CHECK = 3

    def __init__(self, spot_check_rate: float = 0.05):
        self.queue: List[ReviewItem] = []
        self.spot_check_rate = spot_check_rate
        self._approved_count = 0
        self._review_count = 0
        self._escalated_count = 0

    def add(
        self,
        task_id: str,
        original_text: str,
        annotation: Dict[str, Any],
        consensus_score: float,
    ):
        decision = annotation.get("decision", "review")
        priority = self._get_priority(decision, consensus_score)

        item = ReviewItem(
            priority=priority,
            task_id=task_id,
            original_text=original_text,
            annotation=annotation,
            consensus_score=consensus_score,
            created_at=time.time(),
            decision=decision,
        )

        heapq.heappush(self.queue, item)

        if decision == "escalate":
            self._escalated_count += 1
        elif decision == "review":
            self._review_count += 1

    def _get_priority(self, decision: str, score: float) -> int:
        if decision == "escalate" or score <= 0.60:
            return self.PRIORITY_ESCALATED
        elif decision == "review" or score <= 0.85:
            return self.PRIORITY_REVIEW
        return self.PRIORITY_SPOT_CHECK

    def get_next(self) -> ReviewItem | None:
        if not self.queue:
            return None
        return heapq.heappop(self.queue)

    def size(self) -> int:
        """Get queue size."""
        return len(self.queue)

    def get_stats(self) -> Dict[str, int]:
        return {
            "total": len(self.queue),
            "escalated": self._escalated_count,
            "review": self._review_count,
            "spot_check": self._approved_count,
        }


from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class ReviewDecision:
    task_id: str
    decision: str
    corrected_annotation: Dict[str, Any] | None
    reviewer_notes: str | None
    reviewed_at: str


class ReviewWorkflow:

    def __init__(self):
        self.history: list[ReviewDecision] = []

    def approve_auto(self, task_id: str, annotation: Dict[str, Any]) -> ReviewDecision:
        decision = ReviewDecision(
            task_id=task_id,
            decision="auto_approved",
            corrected_annotation=None,
            reviewer_notes=None,
            reviewed_at=datetime.now().isoformat(),
        )
        self.history.append(decision)
        return decision

    def submit_review(
        self,
        task_id: str,
        decision: str,
        corrected_annotation: Dict[str, Any] | None = None,
        reviewer_notes: str | None = None,
    ) -> ReviewDecision:
        review = ReviewDecision(
            task_id=task_id,
            decision=decision,
            corrected_annotation=corrected_annotation,
            reviewer_notes=reviewer_notes,
            reviewed_at=datetime.now().isoformat(),
        )
        self.history.append(review)
        return review

    def get_review_history(self, task_id: str) -> list[ReviewDecision]:
        return [r for r in self.history if r.task_id == task_id]

    def get_stats(self) -> Dict[str, int]:
        stats = {
            "auto_approved": 0,
            "human_approved": 0,
            "human_rejected": 0,
            "human_escalated": 0,
        }
        for decision in self.history:
            if decision.decision == "auto_approved":
                stats["auto_approved"] += 1
            elif decision.decision == "approved":
                stats["human_approved"] += 1
            elif decision.decision == "rejected":
                stats["human_rejected"] += 1
            elif decision.decision == "escalated":
                stats["human_escalated"] += 1
        return stats
