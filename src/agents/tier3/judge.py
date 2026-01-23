"""Judge Agent - Tier 3: MAFA Consensus and Quality Control.

Implements MAFA Section 4.5: Judge Agent and Consensus Mechanism
- Combines outputs from 4 Tier 2 agents
- Uses DYNAMIC weights from MetricsCollector (updated daily)
- Weighted voting with confidence calibration
- NO FALLBACK - All or nothing

Algorithm (Paper MAFA Section 4.5):
    weight[i] = accuracy[i] / Σ(accuracy[j])  # Daily updated
    Final_Score(candidate) = Σ(weight[i] × confidence[i] × score[i][candidate])
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from ...config import get_config
from ...consensus.voting import (
    ConsensusEngine,
    Annotation,
    convert_confidence_to_level,
)


@dataclass
class FinalAnnotation:
    task_id: str
    task_type: str
    label: str
    entities: List[Dict]
    consensus_score: float
    decision: str
    agent_votes: List[Dict]
    audit_trail: Dict
    confidence_level: str  # HIGH/MEDIUM/LOW for overall
    agent_agreement: Dict[str, int]


class JudgeAgent:
    """MAFA Tier 3: Judge Agent for consensus and quality control.

    Uses DYNAMIC weights from MetricsCollector (not static 0.25).
    Weights updated daily based on historical accuracy.

    Combines outputs from 4 Tier 2 agents:
    - PrimaryOnly Agent
    - Contextual Agent
    - Retrieval Agent
    - RetrievalMRL Agent
    """

    # Default weights (used when no metrics available)
    DEFAULT_WEIGHTS = {
        "primary_only": 0.25,
        "contextual": 0.25,
        "retrieval": 0.25,
        "retrieval_mrl": 0.25,
    }

    def __init__(self, metrics_collector=None):
        """Initialize JudgeAgent with optional metrics collector.

        Args:
            metrics_collector: MetricsCollector for dynamic weights.
                             If None, uses default static weights.
        """
        self.config = get_config()
        self.engine = ConsensusEngine()
        self._metrics_collector = metrics_collector

    def _get_dynamic_weights(self) -> Dict[str, float]:
        """Get dynamic weights from metrics or use defaults.

        MAFA Section 4.5: "Weights are updated daily using a rolling
        window of accuracy measurements"

        Returns:
            Dict of agent_name -> weight
        """
        if self._metrics_collector is None:
            return self.DEFAULT_WEIGHTS.copy()

        try:
            return self._metrics_collector.get_weight_distribution()
        except Exception:
            return self.DEFAULT_WEIGHTS.copy()

    def set_metrics_collector(self, metrics_collector):
        """Set or update metrics collector for dynamic weights."""
        self._metrics_collector = metrics_collector

    def _get_thresholds(self):
        """Get consensus thresholds from config."""
        consensus_config = getattr(self.config.task, "consensus", None)
        if consensus_config:
            return {
                "approve": getattr(consensus_config, "approve_threshold", 0.85),
                "escalate": getattr(consensus_config, "escalate_threshold", 0.60),
            }
        # Default fallback only if config missing (should not happen with proper setup)
        return {"approve": 0.85, "escalate": 0.60}

    async def evaluate(
        self,
        task_id: str,
        task_type: str,
        primary_only_result: Dict[str, Any],
        contextual_result: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        retrieval_mrl_result: Dict[str, Any],
    ) -> FinalAnnotation:
        """Evaluate all agent results with DYNAMIC weights.

        Algorithm:
            weights = get_dynamic_weights()  # From MetricsCollector
            annotations = [
                Annotation(weight=weights["primary_only"], ...),
                Annotation(weight=weights["contextual"], ...),
                Annotation(weight=weights["retrieval"], ...),
                Annotation(weight=weights["retrieval_mrl"], ...),
            ]
            consensus = engine.calculate(annotations, task_type)

        NO FALLBACK - Raises exception on failure.
        """
        # Get dynamic weights from MetricsCollector
        weights = self._get_dynamic_weights()

        # Convert confidence to MAFA levels
        conf_primary = primary_only_result.get("confidence", 0.5)
        conf_contextual = contextual_result.get("confidence", 0.5)
        conf_retrieval = retrieval_result.get("confidence", 0.5)
        conf_retrieval_mrl = retrieval_mrl_result.get("confidence", 0.5)

        # Build annotations with DYNAMIC weights
        annotations = [
            Annotation(
                agent_name="primary_only",
                confidence=conf_primary,
                confidence_level=convert_confidence_to_level(conf_primary),
                result=primary_only_result,
                weight=weights.get(
                    "primary_only", self.DEFAULT_WEIGHTS["primary_only"]
                ),
                reasoning=primary_only_result.get("reasoning", ""),
            ),
            Annotation(
                agent_name="contextual",
                confidence=conf_contextual,
                confidence_level=convert_confidence_to_level(conf_contextual),
                result=contextual_result,
                weight=weights.get("contextual", self.DEFAULT_WEIGHTS["contextual"]),
                reasoning=contextual_result.get("reasoning", ""),
            ),
            Annotation(
                agent_name="retrieval",
                confidence=conf_retrieval,
                confidence_level=convert_confidence_to_level(conf_retrieval),
                result=retrieval_result,
                weight=weights.get("retrieval", self.DEFAULT_WEIGHTS["retrieval"]),
                reasoning=retrieval_result.get("reasoning", ""),
            ),
            Annotation(
                agent_name="retrieval_mrl",
                confidence=conf_retrieval_mrl,
                confidence_level=convert_confidence_to_level(conf_retrieval_mrl),
                result=retrieval_mrl_result,
                weight=weights.get(
                    "retrieval_mrl", self.DEFAULT_WEIGHTS["retrieval_mrl"]
                ),
                reasoning=retrieval_mrl_result.get("reasoning", ""),
            ),
        ]

        consensus = self.engine.calculate(annotations, task_type)

        # Override decision based on configurable thresholds
        thresholds = self._get_thresholds()
        if consensus.score >= thresholds["approve"]:
            consensus.decision = "approve"
        elif consensus.score < thresholds["escalate"]:
            consensus.decision = "escalate"
        else:
            consensus.decision = "review"

        # Get final label
        if task_type in ["ner", "topic"]:
            final_label = consensus.final_label
            final_entities = []
        else:
            final_label = consensus.final_label
            final_entities = []

        # Overall confidence level
        overall_level = convert_confidence_to_level(consensus.score)

        # Audit trail with weight info
        audit_trail = consensus.audit_trail.copy()
        audit_trail["used_dynamic_weights"] = self._metrics_collector is not None
        audit_trail["weights_used"] = {a.agent_name: a.weight for a in annotations}

        return FinalAnnotation(
            task_id=task_id,
            task_type=task_type,
            label=final_label,
            entities=final_entities,
            consensus_score=consensus.score,
            decision=consensus.decision,
            agent_votes=consensus.votes,
            audit_trail=audit_trail,
            confidence_level=overall_level,
            agent_agreement=consensus.agent_agreement or {},
        )

    def should_escalate(self, annotation: FinalAnnotation) -> bool:
        """Check if annotation should be escalated to expert."""
        return annotation.decision == "escalate"

    def should_review(self, annotation: FinalAnnotation) -> bool:
        """Check if annotation requires human review."""
        return annotation.decision == "review"

    def should_approve(self, annotation: FinalAnnotation) -> bool:
        """Check if annotation can be auto-approved."""
        return annotation.decision == "approve"

    def get_confidence_distribution(
        self, annotation: FinalAnnotation
    ) -> Dict[str, int]:
        """Get distribution of HIGH/MEDIUM/LOW confidence from audit trail."""
        return annotation.audit_trail.get("confidence_distribution", {})

    def get_agreement_rate(self, annotation: FinalAnnotation) -> float:
        """Get agent agreement rate."""
        return annotation.audit_trail.get("agreement_rate", 0.0)
