"""
Judge Agent - Tier 3.

Aggregates multi-agent annotations and makes final routing decisions.
Responsible for consensus calculation and quality control.
"""

from datetime import datetime
from typing import Any

import structlog

from ..consensus import (
    ConsensusEngine,
    ConfidenceAggregator,
    ThresholdManager,
    get_consensus_engine,
    get_confidence_aggregator,
    get_threshold_manager,
)
from ..models import (
    Annotation,
    AnnotationResult,
    ConsensusResult,
    DecisionType,
    Task,
    TaskStatus,
)

logger = structlog.get_logger()


class JudgeAgent:
    """
    Judge Agent (Tier 3).

    Responsibilities:
    - Aggregate multi-agent votes using weighted voting
    - Calculate consensus scores
    - Make routing decisions (approve/review/escalate)
    - Generate audit trails for traceability
    - Flag low-confidence or conflicting results
    """

    name = "judge_agent"
    tier = 3

    def __init__(
        self,
        consensus_engine: ConsensusEngine | None = None,
        confidence_aggregator: ConfidenceAggregator | None = None,
        threshold_manager: ThresholdManager | None = None,
    ):
        """
        Initialize Judge Agent.

        Args:
            consensus_engine: Custom consensus engine
            confidence_aggregator: Custom confidence aggregator
            threshold_manager: Custom threshold manager
        """
        self.consensus_engine = consensus_engine or get_consensus_engine()
        self.confidence_aggregator = confidence_aggregator or get_confidence_aggregator()
        self.threshold_manager = threshold_manager or get_threshold_manager()

    def calculate_consensus(
        self,
        annotations: list[Annotation],
    ) -> ConsensusResult:
        """
        Calculate weighted consensus from multiple agent annotations.

        Args:
            annotations: List of annotations from different agents

        Returns:
            ConsensusResult with score, decision, and audit trail
        """
        if not annotations:
            logger.warning("judge_received_empty_annotations")
            return ConsensusResult(
                score=0.0,
                decision=DecisionType.ESCALATE,
                votes=[],
                audit_trail={"error": "No annotations to judge"},
            )

        # Calculate consensus using voting engine
        consensus = self.consensus_engine.calculate_consensus(annotations)

        # Check for conflicts
        conflicts = self.consensus_engine.detect_conflicts(annotations)
        if conflicts:
            consensus.audit_trail["conflicts"] = conflicts

        # Check uncertainty
        uncertainty = self.confidence_aggregator.estimate_uncertainty(annotations)
        consensus.audit_trail["uncertainty"] = uncertainty

        logger.info(
            "consensus_calculated",
            score=round(consensus.score, 4),
            decision=consensus.decision.value,
            num_annotations=len(annotations),
            has_conflicts=len(conflicts) > 0,
        )

        return consensus

    def flag_for_review(
        self,
        consensus: ConsensusResult,
        annotations: list[Annotation],
    ) -> tuple[bool, list[str]]:
        """
        Determine if result should be flagged for human review.

        Args:
            consensus: ConsensusResult
            annotations: Original annotations

        Returns:
            Tuple of (should_flag, reasons)
        """
        return self.threshold_manager.should_flag_for_review(
            consensus=consensus,
            annotations=annotations,
        )

    def judge(
        self,
        task: Task,
        annotations: list[Annotation],
    ) -> AnnotationResult:
        """
        Make final judgment on a task's annotations.

        This is the main entry point for the Judge Agent.

        Args:
            task: The original task
            annotations: Annotations from all agents

        Returns:
            AnnotationResult with final decision
        """
        # Calculate consensus
        consensus = self.calculate_consensus(annotations)

        # Determine if needs review
        needs_review, review_reasons = self.flag_for_review(consensus, annotations)

        # Update consensus audit trail
        consensus.audit_trail["review_flagged"] = needs_review
        consensus.audit_trail["review_reasons"] = review_reasons

        # Determine final status
        if consensus.decision == DecisionType.APPROVE:
            status = TaskStatus.COMPLETED
        elif consensus.decision == DecisionType.REVIEW:
            status = TaskStatus.IN_REVIEW
        else:
            status = TaskStatus.IN_REVIEW  # Escalation also goes to review

        # Extract specific annotation types
        intent_annotation = None
        entity_annotation = None
        faq_annotation = None

        for ann in annotations:
            if "intent" in ann.agent_name.lower():
                intent_annotation = ann.result
            elif "entity" in ann.agent_name.lower():
                entity_annotation = ann.result
            elif "faq" in ann.agent_name.lower():
                faq_annotation = ann.result

        result = AnnotationResult(
            task_id=task.id,
            intent=intent_annotation,
            entities=entity_annotation,
            faq=faq_annotation,
            consensus=consensus,
            annotations=annotations,
            status=status,
        )

        logger.info(
            "judgment_complete",
            task_id=str(task.id),
            decision=consensus.decision.value,
            status=status.value,
            needs_review=needs_review,
        )

        return result

    def generate_audit_trail(
        self,
        task: Task,
        annotations: list[Annotation],
        consensus: ConsensusResult,
    ) -> dict[str, Any]:
        """
        Generate detailed audit trail for compliance and debugging.

        Args:
            task: Original task
            annotations: All annotations
            consensus: Consensus result

        Returns:
            Comprehensive audit trail
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "judge_agent": self.name,
            "task": {
                "id": str(task.id),
                "text_preview": task.text[:100] + "..." if len(task.text) > 100 else task.text,
                "created_at": task.created_at.isoformat(),
            },
            "annotations": [
                {
                    "agent": ann.agent_name,
                    "confidence": round(ann.confidence, 4),
                    "weight": round(ann.weight, 4),
                    "latency_ms": round(ann.latency_ms, 2),
                }
                for ann in annotations
            ],
            "consensus": {
                "score": round(consensus.score, 4),
                "decision": consensus.decision.value,
                "votes": [
                    {
                        "agent": v.agent_name,
                        "weighted_score": round(v.weighted_score, 4),
                    }
                    for v in consensus.votes
                ],
            },
            "configuration": {
                "thresholds": self.threshold_manager.get_config(),
                "agent_weights": self.consensus_engine.agent_weights,
            },
        }

    def reconsider(
        self,
        annotations: list[Annotation],
        exclude_agents: list[str],
    ) -> ConsensusResult:
        """
        Recalculate consensus excluding specific agents.

        Useful when certain agents are known to be unreliable.

        Args:
            annotations: All annotations
            exclude_agents: Agents to exclude

        Returns:
            New ConsensusResult
        """
        return self.consensus_engine.recalculate_with_exclusion(
            annotations=annotations,
            exclude_agents=exclude_agents,
        )


# Singleton instance
_judge_agent: JudgeAgent | None = None


def get_judge_agent() -> JudgeAgent:
    """Get the singleton judge agent instance."""
    global _judge_agent
    if _judge_agent is None:
        _judge_agent = JudgeAgent()
    return _judge_agent
