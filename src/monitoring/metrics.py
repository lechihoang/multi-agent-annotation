

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import json
import os


@dataclass
class AnnotationRecord:
    """Single annotation record for tracking."""

    task_id: str
    text: str
    true_label: Optional[str]
    agent_predictions: Dict[str, str]
    final_label: str
    consensus_score: float
    task_type: str
    timestamp: datetime


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""

    agent_name: str
    total_annotations: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.5
    task_type_metrics: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    confidence_distribution: Dict[str, int] = field(
        default_factory=dict
    )
    recent_scores: List[float] = field(default_factory=list)

    def update(self, prediction: str, true_label: str, confidence: float):
        """Update metrics with new observation."""
        self.total_annotations += 1
        if prediction == true_label:
            self.correct_predictions += 1
        self.accuracy = self.correct_predictions / self.total_annotations
        self.recent_scores.append(1.0 if prediction == true_label else 0.0)
        if len(self.recent_scores) > 100:
            self.recent_scores = self.recent_scores[-100:]


class MetricsCollector:

    def __init__(
        self,
        storage_path: str = "data/metrics.json",
        window_days: int = 30,
        min_samples: int = 10,
    ):
        """Initialize metrics collector.

        Args:
            storage_path: Path to persist metrics
            window_days: Lookback period for weight calculation
            min_samples: Minimum samples required for weight update
        """
        self.storage_path = storage_path
        self.window_days = window_days
        self.min_samples = min_samples

        self._records: List[AnnotationRecord] = []
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._lock = threading.Lock()

        self._agent_weights: Dict[str, float] = {
            "primary_only": 0.25,
            "contextual": 0.25,
            "retrieval": 0.25,
            "retrieval_mrl": 0.25,
        }

        for agent in self._agent_weights.keys():
            self._agent_metrics[agent] = AgentMetrics(agent_name=agent)

        self._load_from_disk()

    def record_annotation(
        self,
        task_id: str,
        text: str,
        true_label: Optional[str],
        agent_predictions: Dict[str, str],
        final_label: str,
        consensus_score: float,
        task_type: str = "classification",
        agent_confidences: Optional[Dict[str, float]] = None,
    ):
        with self._lock:
            record = AnnotationRecord(
                task_id=task_id,
                text=text,
                true_label=true_label,
                agent_predictions=agent_predictions,
                final_label=final_label,
                consensus_score=consensus_score,
                task_type=task_type,
                timestamp=datetime.now(),
            )
            self._records.append(record)

            if true_label is not None:
                self._update_agent_metrics(
                    agent_predictions, true_label, agent_confidences
                )

    def _update_agent_metrics(
        self,
        predictions: Dict[str, str],
        true_label: str,
        confidences: Optional[Dict[str, float]],
    ):
        for agent_name, pred in predictions.items():
            if agent_name in self._agent_metrics:
                confidence = confidences.get(agent_name, 0.5) if confidences else 0.5
                self._agent_metrics[agent_name].update(pred, true_label, confidence)

    def calculate_weights(self) -> Dict[str, float]:
        with self._lock:
            if len(self._records) < self.min_samples:
                return self._agent_weights.copy()

            agent_scores: Dict[str, float] = {}

            for agent_name, metrics in self._agent_metrics.items():
                if metrics.total_annotations < self.min_samples:
                    agent_scores[agent_name] = 0.5
                else:
                    recent = metrics.recent_scores[-50:]
                    if recent:
                        agent_scores[agent_name] = sum(recent) / len(recent)
                    else:
                        agent_scores[agent_name] = metrics.accuracy

            total_score = sum(agent_scores.values())
            if total_score > 0:
                new_weights = {
                    agent: score / total_score for agent, score in agent_scores.items()
                }
            else:
                new_weights = self._agent_weights.copy()

            return new_weights

    def update_weights(self) -> Dict[str, float]:
        new_weights = self.calculate_weights()

        with self._lock:
            self._agent_weights = new_weights
            self._save_to_disk()

        return new_weights

    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        if agent_name not in self._agent_metrics:
            return {"error": f"Unknown agent: {agent_name}"}

        metrics = self._agent_metrics[agent_name]
        recent = metrics.recent_scores[-50:] if metrics.recent_scores else []

        return {
            "agent_name": agent_name,
            "total_annotations": metrics.total_annotations,
            "correct_predictions": metrics.correct_predictions,
            "accuracy": round(metrics.accuracy, 4),
            "recent_accuracy": round(sum(recent) / len(recent), 4) if recent else 0.5,
            "current_weight": self._agent_weights.get(agent_name, 0.25),
            "task_breakdown": {
                task: {
                    "total": data["total"],
                    "correct": data["correct"],
                    "accuracy": round(data["correct"] / data["total"], 4)
                    if data["total"] > 0
                    else 0,
                }
                for task, data in metrics.task_type_metrics.items()
            },
        }

    def get_overall_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_records = len(self._records)
            records_with_truth = sum(
                1 for r in self._records if r.true_label is not None
            )

            correct_final = sum(
                1
                for r in self._records
                if r.true_label is not None and r.final_label == r.true_label
            )
            system_accuracy = (
                correct_final / records_with_truth if records_with_truth > 0 else 0
            )

            agent_comparison = {}
            for agent_name, metrics in self._agent_metrics.items():
                if metrics.total_annotations > 0:
                    agent_comparison[agent_name] = {
                        "accuracy": round(metrics.accuracy, 4),
                        "weight": round(self._agent_weights.get(agent_name, 0.25), 4),
                    }

            return {
                "total_annotations": total_records,
                "annotations_with_truth": records_with_truth,
                "system_accuracy": round(system_accuracy, 4),
                "agent_weights": self._agent_weights,
                "agent_performance": agent_comparison,
                "window_days": self.window_days,
            }

    def get_weight_distribution(self) -> Dict[str, float]:
        return self._agent_weights.copy()

    def apply_ground_truth(
        self,
        task_id: str,
        true_label: str,
    ):
        with self._lock:
            for record in self._records:
                if record.task_id == task_id:
                    record.true_label = true_label
                    self._update_agent_metrics(
                        record.agent_predictions,
                        true_label,
                        None,
                    )
                    break

    def cleanup_old_records(self, days: int = 90):
        cutoff = datetime.now() - timedelta(days=days)
        with self._lock:
            self._records = [r for r in self._records if r.timestamp > cutoff]

    def _save_to_disk(self):
        try:
            data = {
                "weights": self._agent_weights,
                "records_count": len(self._records),
                "agent_metrics": {
                    name: {
                        "total": metrics.total_annotations,
                        "correct": metrics.correct_predictions,
                        "accuracy": metrics.accuracy,
                    }
                    for name, metrics in self._agent_metrics.items()
                },
                "last_updated": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metrics: {e}")

    def _load_from_disk(self):
        try:
            if not os.path.exists(self.storage_path):
                return

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            if "weights" in data:
                self._agent_weights = data["weights"]

            if "agent_metrics" in data:
                for name, metrics_data in data["agent_metrics"].items():
                    if name in self._agent_metrics:
                        self._agent_metrics[name].total_annotations = metrics_data.get(
                            "total", 0
                        )
                        self._agent_metrics[
                            name
                        ].correct_predictions = metrics_data.get("correct", 0)
                        self._agent_metrics[name].accuracy = metrics_data.get(
                            "accuracy", 0.5
                        )

        except Exception as e:
            print(f"Warning: Failed to load metrics: {e}")

    def export_metrics(self, path: str = "data/metrics_report.json"):
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_overall_stats(),
            "agent_details": {
                name: self.get_agent_performance(name)
                for name in self._agent_metrics.keys()
            },
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report
