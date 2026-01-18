"""
Weighted voting algorithm for multi-agent consensus.

Implements the core consensus mechanism that aggregates agent annotations
using weighted voting to produce final decisions.
"""

from datetime import datetime
from typing import Any

import structlog

from ..config.settings import settings
from ..models import Annotation, ConsensusResult, DecisionType, Vote

logger = structlog.get_logger()


class ConsensusEngine:
    """
    Weighted voting consensus mechanism.

    Formula: Score = Σ(agent_confidence × agent_weight) / N

    Agent Weights (default):
    - Intent Agent: 0.35
    - Entity Agent: 0.35
    - FAQ Agent: 0.30
    """

    def __init__(
        self,
        agent_weights: dict[str, float] | None = None,
        auto_approve_threshold: float | None = None,
        human_review_threshold: float | None = None,
    ):
        """
        Initialize consensus engine.

        Args:
            agent_weights: Custom agent weights (uses defaults if None)
            auto_approve_threshold: Threshold for auto-approval
            human_review_threshold: Threshold for human review
        """
        self.agent_weights = agent_weights or settings.agent_weights
        self.auto_approve_threshold = (
            auto_approve_threshold or settings.auto_approve_threshold
        )
        self.human_review_threshold = (
            human_review_threshold or settings.human_review_threshold
        )

        # Validate weights sum to ~1.0
        weight_sum = sum(self.agent_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                "agent_weights_not_normalized",
                weight_sum=weight_sum,
                weights=self.agent_weights,
            )

    def calculate_consensus(
        self,
        annotations: list[Annotation],
    ) -> ConsensusResult:
        """
        Calculate weighted consensus score from agent annotations.

        Args:
            annotations: List of annotations from different agents

        Returns:
            ConsensusResult with score, decision, votes, and audit trail
        """
        if not annotations:
            logger.warning("empty_annotations_for_consensus")
            return ConsensusResult(
                score=0.0,
                decision=DecisionType.ESCALATE,
                votes=[],
                audit_trail={"error": "No annotations provided"},
            )

        # Calculate weighted scores
        votes: list[Vote] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for annotation in annotations:
            weight = self._get_agent_weight(annotation.agent_name)
            weighted_score = annotation.confidence * weight

            vote = Vote(
                agent_name=annotation.agent_name,
                confidence=annotation.confidence,
                weight=weight,
                weighted_score=weighted_score,
            )
            votes.append(vote)

            weighted_sum += weighted_score
            total_weight += weight

        # Calculate final score
        if total_weight > 0:
            score = weighted_sum / total_weight
        else:
            score = 0.0

        # Determine decision
        decision = self._determine_decision(score)

        # Generate audit trail
        audit_trail = self._generate_audit_trail(
            annotations=annotations,
            votes=votes,
            score=score,
            decision=decision,
        )

        logger.info(
            "consensus_calculated",
            score=round(score, 4),
            decision=decision.value,
            num_agents=len(annotations),
        )

        return ConsensusResult(
            score=score,
            decision=decision,
            votes=votes,
            audit_trail=audit_trail,
        )

    def _get_agent_weight(self, agent_name: str) -> float:
        """Get weight for an agent, defaulting to equal weight if unknown."""
        # Handle different naming conventions
        normalized_name = agent_name.lower().replace("-", "_")

        # Direct match
        if normalized_name in self.agent_weights:
            return self.agent_weights[normalized_name]

        # Try without '_agent' suffix
        base_name = normalized_name.replace("_agent", "")
        for key, weight in self.agent_weights.items():
            if base_name in key or key in base_name:
                return weight

        # Default to equal weight
        default_weight = 1.0 / max(len(self.agent_weights), 1)
        logger.warning(
            "unknown_agent_weight",
            agent_name=agent_name,
            using_default=default_weight,
        )
        return default_weight

    def _determine_decision(self, score: float) -> DecisionType:
        """Determine routing decision based on score."""
        if score >= self.auto_approve_threshold:
            return DecisionType.APPROVE
        elif score >= self.human_review_threshold:
            return DecisionType.REVIEW
        else:
            return DecisionType.ESCALATE

    def _generate_audit_trail(
        self,
        annotations: list[Annotation],
        votes: list[Vote],
        score: float,
        decision: DecisionType,
    ) -> dict[str, Any]:
        """Generate detailed audit trail for traceability."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "algorithm": "weighted_voting",
            "configuration": {
                "agent_weights": self.agent_weights,
                "auto_approve_threshold": self.auto_approve_threshold,
                "human_review_threshold": self.human_review_threshold,
            },
            "input": {
                "num_annotations": len(annotations),
                "agents": [a.agent_name for a in annotations],
            },
            "calculation": {
                "votes": [
                    {
                        "agent": v.agent_name,
                        "confidence": round(v.confidence, 4),
                        "weight": round(v.weight, 4),
                        "weighted_score": round(v.weighted_score, 4),
                    }
                    for v in votes
                ],
                "weighted_sum": round(sum(v.weighted_score for v in votes), 4),
                "total_weight": round(sum(v.weight for v in votes), 4),
            },
            "output": {
                "final_score": round(score, 4),
                "decision": decision.value,
            },
        }

    def detect_conflicts(
        self,
        annotations: list[Annotation],
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Detect significant disagreements between agents.

        Args:
            annotations: List of annotations
            threshold: Confidence difference threshold for conflict detection

        Returns:
            List of detected conflicts
        """
        conflicts = []

        if len(annotations) < 2:
            return conflicts

        for i, ann1 in enumerate(annotations):
            for ann2 in annotations[i + 1 :]:
                diff = abs(ann1.confidence - ann2.confidence)
                if diff >= threshold:
                    conflicts.append(
                        {
                            "agents": [ann1.agent_name, ann2.agent_name],
                            "confidence_diff": round(diff, 4),
                            "confidences": {
                                ann1.agent_name: round(ann1.confidence, 4),
                                ann2.agent_name: round(ann2.confidence, 4),
                            },
                        }
                    )

        if conflicts:
            logger.warning(
                "conflicts_detected",
                num_conflicts=len(conflicts),
                conflicts=conflicts,
            )

        return conflicts

    def recalculate_with_exclusion(
        self,
        annotations: list[Annotation],
        exclude_agents: list[str],
    ) -> ConsensusResult:
        """
        Recalculate consensus excluding specific agents.

        Useful when certain agents are known to be unreliable for a task.

        Args:
            annotations: All annotations
            exclude_agents: Agents to exclude from calculation

        Returns:
            New ConsensusResult
        """
        filtered = [
            a for a in annotations if a.agent_name not in exclude_agents
        ]
        return self.calculate_consensus(filtered)


# Singleton instance
_consensus_engine: ConsensusEngine | None = None


def get_consensus_engine() -> ConsensusEngine:
    """Get the singleton consensus engine instance."""
    global _consensus_engine
    if _consensus_engine is None:
        _consensus_engine = ConsensusEngine()
    return _consensus_engine
