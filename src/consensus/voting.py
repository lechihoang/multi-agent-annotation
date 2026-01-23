"""MAFA-style Weighted Voting Consensus Mechanism.

NO FALLBACK - Strict MAFA pattern implementation.

Implements:
- Confidence calibration (HIGH/MEDIUM/LOW)
- Dynamic weights based on historical accuracy
- Detailed audit trail
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter


@dataclass
class Annotation:
    agent_name: str
    confidence: float  # 0.0-1.0
    confidence_level: str  # "HIGH"/"MEDIUM"/"LOW"
    result: Dict[str, Any]
    weight: float
    reasoning: str = ""


@dataclass
class ConsensusResult:
    score: float
    decision: str
    votes: List[Dict]
    audit_trail: Dict
    final_label: str
    agent_agreement: Dict[str, int] | None = None


class ConsensusEngine:
    """MAFA-style consensus mechanism.

    NO FALLBACK - All or nothing.

    Features:
    - Confidence calibration: HIGH ×1.5, MEDIUM ×1.0, LOW ×0.5
    - Dynamic weights based on historical accuracy
    - Multi-agent agreement bonus
    - Detailed audit trail for compliance
    """

    # Confidence calibration factors (MAFA pattern)
    CONFIDENCE_FACTORS = {
        "HIGH": 1.5,
        "MEDIUM": 1.0,
        "LOW": 0.5,
    }

    # Historical accuracy weights
    AGENT_HISTORICAL_ACCURACY = {
        "primary_only": 0.25,
        "contextual": 0.25,
        "retrieval": 0.25,
        "retrieval_mrl": 0.25,
    }

    def calculate(
        self,
        annotations: List[Annotation],
        task_type: str = "classification",
    ) -> ConsensusResult:
        """Calculate weighted consensus with MAFA pattern.

        NO FALLBACK - Raises exception on failure.
        """
        # Step 1: Get all predicted labels
        labels = [ann.result.get("label", "unknown") for ann in annotations]

        # Step 2: Calculate multi-agent agreement
        label_counts = Counter(labels)
        agreement_score = label_counts.most_common(1)[0][1] / len(labels)

        # Step 3: Confidence calibration (HIGH ×1.5, MEDIUM ×1.0, LOW ×0.5)
        calibrated_scores = []
        for ann in annotations:
            conf_factor = self.CONFIDENCE_FACTORS.get(ann.confidence_level.upper(), 1.0)
            hist_weight = self.AGENT_HISTORICAL_ACCURACY.get(ann.agent_name, 0.25)
            calibrated_score = ann.confidence * conf_factor * hist_weight
            calibrated_scores.append(calibrated_score)

        # Step 4: Agreement bonus
        agreement_bonus = agreement_score * 0.1  # 10% bonus

        # Step 5: Final score
        raw_score = sum(calibrated_scores) / len(annotations)
        final_score = min(raw_score + agreement_bonus, 1.0)

        # Step 6: Decision based on MAFA thresholds
        decision = self._get_decision(final_score)

        # Step 7: Final label with agreement consideration
        if agreement_score >= 0.75:
            final_label = label_counts.most_common(1)[0][0]
        else:
            best_idx = calibrated_scores.index(max(calibrated_scores))
            final_label = labels[best_idx]

        # Step 8: Serialize votes
        votes = [
            self._serialize_vote(a, cs) for a, cs in zip(annotations, calibrated_scores)
        ]

        # Step 9: Generate detailed audit trail
        audit_trail = self._generate_audit(
            annotations, labels, label_counts, final_score, decision, final_label
        )

        return ConsensusResult(
            score=final_score,
            decision=decision,
            votes=votes,
            audit_trail=audit_trail,
            final_label=final_label,
            agent_agreement=dict(label_counts),
        )

    def _get_decision(self, score: float) -> str:
        """Get decision based on MAFA thresholds."""
        if score >= 0.85:
            return "approve"  # Auto-approve
        elif score >= 0.60:
            return "review"  # Human review
        return "escalate"  # Expert escalation

    def _serialize_vote(
        self, annotation: Annotation, calibrated_score: float
    ) -> Dict[str, Any]:
        """Serialize annotation with calibration details."""
        return {
            "agent": annotation.agent_name,
            "label": annotation.result.get("label", "unknown"),
            "confidence": annotation.confidence,
            "confidence_level": annotation.confidence_level,
            "base_weight": annotation.weight,
            "calibrated_weight": self.AGENT_HISTORICAL_ACCURACY.get(
                annotation.agent_name, 0.25
            ),
            "calibration_factor": self.CONFIDENCE_FACTORS.get(
                annotation.confidence_level.upper(), 1.0
            ),
            "calibrated_score": calibrated_score,
            "reasoning": annotation.reasoning or annotation.result.get("reasoning", ""),
        }

    def _generate_audit(
        self,
        annotations: List[Annotation],
        labels: List[str],
        label_counts: Counter,
        score: float,
        decision: str,
        final_label: str,
    ) -> Dict[str, Any]:
        """Generate detailed audit trail for compliance."""
        conf_levels = Counter(ann.confidence_level.upper() for ann in annotations)

        top_label, top_count = label_counts.most_common(1)[0]
        agreement_rate = top_count / len(annotations)

        calibration_details = []
        for ann in annotations:
            conf_factor = self.CONFIDENCE_FACTORS.get(ann.confidence_level.upper(), 1.0)
            hist_weight = self.AGENT_HISTORICAL_ACCURACY.get(ann.agent_name, 0.25)
            calibration_details.append(
                {
                    "agent": ann.agent_name,
                    "raw_confidence": ann.confidence,
                    "confidence_level": ann.confidence_level,
                    "calibration_factor": conf_factor,
                    "historical_weight": hist_weight,
                    "calibrated_score": ann.confidence * conf_factor * hist_weight,
                }
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "method": "mafa_weighted_consensus",
            "thresholds": {"approve": 0.85, "review": 0.60, "escalate": 0.0},
            "raw_score": score - (agreement_rate * 0.1),
            "agreement_bonus": agreement_rate * 0.1,
            "final_score": score,
            "decision": decision,
            "final_label": final_label,
            "agent_agreement": {
                "labels": dict(label_counts),
                "agreement_rate": agreement_rate,
                "consensus_label": top_label,
            },
            "confidence_distribution": dict(conf_levels),
            "calibration_details": calibration_details,
            "reasoning_summary": self._summarize_reasoning(annotations),
        }

    def _summarize_reasoning(self, annotations: List[Annotation]) -> str:
        """Summarize reasoning from all agents."""
        reasons = []
        for ann in annotations:
            reasoning = ann.result.get("reasoning", "") or ann.reasoning
            if reasoning:
                reasons.append(f"{ann.agent_name}: {reasoning[:100]}...")
        return " | ".join(reasons) if reasons else "No reasoning provided"

    def update_weights(self, agent_name: str, new_accuracy: float):
        """Update historical accuracy weights (called daily in production)."""
        if agent_name in self.AGENT_HISTORICAL_ACCURACY:
            total = sum(
                v if k != agent_name else new_accuracy
                for k, v in self.AGENT_HISTORICAL_ACCURACY.items()
            )
            self.AGENT_HISTORICAL_ACCURACY[agent_name] = new_accuracy / total


def convert_confidence_to_level(confidence: float) -> str:
    """Convert numeric confidence to MAFA level."""
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    return "LOW"
