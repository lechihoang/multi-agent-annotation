"""Multi-Agent Data Annotation System - Main Entry Point.

MAFA-inspired annotation pipeline with:
- Tier 1: Dynamic Router Agent + Query Planner + Query Expander
- Tier 2: Parallel Annotation Agents (PrimaryOnly, Contextual, Retrieval, Hybrid)
- Tier 3: Judge Agent - Consensus and quality control
- Tier 4: Review Queue & Workflow
- Dynamic Few-Shot Selection from training pool
"""

import asyncio
import uuid
import os
from pathlib import Path
from typing import Dict, Any, List

from .config import get_config
from .tier1 import RouterAgent, Task, RouterMode
from .tier1.query_planner import QueryPlanner
from .tier1.query_expander import QueryExpander
from .tier2.agents import (
    PrimaryOnlyAgent,
    ContextualAgent,
    RetrievalAgent,
    RetrievalMrlAgent,
)
from .tier2.few_shot_selector import FewShotSelector
from .tier3 import JudgeAgent
from .tier4 import ReviewQueue, ReviewWorkflow
from .monitoring import MetricsCollector, WeightUpdateScheduler, create_weight_updater

# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data"


class AnnotationPipeline:
    """MAFA-inspired annotation pipeline with full MAFA components.

    Flow:
    1. Tier 1: Router + QueryPlanner + QueryExpander
    2. Tier 2: Parallel annotation agents with dynamic few-shot selection
    3. Tier 3: Judge Agent - Consensus and quality control
    4. Tier 4: Human-in-the-loop review
    """

    def __init__(self, router_mode: RouterMode = RouterMode.DYNAMIC):
        self.config = get_config()
        self.router = RouterAgent(mode=router_mode)

        # Tier 1: Query Planning & Expansion (MAFA Section 4.2)
        self._init_tier1_components()

        # Tier 2: Parallel Annotation Agents
        self.primary_only = PrimaryOnlyAgent()
        self.contextual = ContextualAgent()
        self.retrieval = RetrievalAgent()
        self.hybrid = RetrievalMrlAgent()

        # Tier 2: Dynamic Few-Shot Selection (MAFA Section 4.6)
        self._init_few_shot_selector()

        # Tier 3: Judge Agent
        self.judge = JudgeAgent()

        # Tier 4: Human-in-the-loop
        self.review_queue = ReviewQueue()
        self.review_workflow = ReviewWorkflow()

        # Tier 5: Production Monitoring & Weight Updates (MAFA Section 4.5)
        self._init_monitoring()

    def _init_tier1_components(self):
        """Initialize QueryPlanner and QueryExpander."""
        try:
            from .config import get_llm_client

            llm_client = get_llm_client(self.config)
            if llm_client is None:
                raise RuntimeError("No LLM client available")
            self.query_planner = QueryPlanner(llm_client)
        except Exception as e:
            print(f"Warning: Could not initialize QueryPlanner: {e}")
            self.query_planner = None

        try:
            csv_path = DATA_DIR / "train.csv"
            if csv_path.exists():
                self.query_expander = QueryExpander(str(csv_path))
            else:
                self.query_expander = None
        except Exception as e:
            print(f"Warning: Could not initialize QueryExpander: {e}")
            self.query_expander = None

    def _init_few_shot_selector(self):
        """Initialize Dynamic Few-Shot Selector from training pool."""
        try:
            csv_path = DATA_DIR / "train.csv"
            if csv_path.exists():
                self.few_shot_selector = FewShotSelector(
                    training_data_path=str(csv_path),
                    groq_client=None,  # Will use fallback selection
                    cache_size=500,
                )
                print(
                    f"✓ FewShotSelector initialized with {self.few_shot_selector.stats()}"
                )
            else:
                self.few_shot_selector = None
        except Exception as e:
            print(f"Warning: Could not initialize FewShotSelector: {e}")
            self.few_shot_selector = None

    def _init_monitoring(self):
        """Initialize production monitoring and weight update scheduler."""
        try:
            metrics_path = DATA_DIR / "metrics.json"
            self.metrics_collector, self.weight_scheduler = create_weight_updater(
                storage_path=str(metrics_path)
            )
            print(
                f"✓ Monitoring initialized: {self.metrics_collector.get_overall_stats()}"
            )
        except Exception as e:
            print(f"Warning: Could not initialize monitoring: {e}")
            self.metrics_collector = None
            self.weight_scheduler = None

    async def _expand_query(self, text: str) -> str:
        """Expand query using MAFA Query Planning + Embedding-based expansion."""
        # Use QueryPlanner first (LLM-based)
        if self.query_planner:
            try:
                expanded = await self.query_planner.expand(text, use_cache=True)
                if expanded != text:
                    return expanded
            except Exception:
                pass

        # Fallback to embedding-based expansion
        if self.query_expander:
            try:
                return self.query_expander.expand(text, top_k=5)
            except Exception:
                pass

        return text

    async def process(self, text: str, task_id: str | None = None) -> Dict[str, Any]:
        """Process a single text through the full annotation pipeline."""
        task_id = task_id or str(uuid.uuid4())
        task = Task(id=task_id, text=text)

        # ============ Tier 1: Dynamic Routing ============
        routing = await self.router.analyze(task)
        labels = self.router.get_label_names(routing)

        # ============ Tier 1: Query Expansion (MAFA) ============
        expanded_text = await self._expand_query(text)

        # ============ Tier 2: Parallel Annotation Agents ============
        # Get unique few-shot examples for each agent (MAFA Section 4.6)
        agent_examples = {}
        if self.few_shot_selector:
            agent_examples = self.few_shot_selector.select_diverse_for_all_agents(
                query=expanded_text or text,
                task_type=routing.task_type.value,
                k_per_agent=8,
            )

        # Convert examples to dict format for agents
        primary_examples = [
            ex.to_dict() for ex in agent_examples.get("primary_only", [])
        ]
        contextual_examples = [
            ex.to_dict() for ex in agent_examples.get("contextual", [])
        ]

        # Run 4 agents in parallel WITH few-shot examples
        primary_result = await self.primary_only.annotate(
            text, labels, few_shot_examples=primary_examples
        )
        contextual_result = await self.contextual.annotate(
            text, "", labels, few_shot_examples=contextual_examples
        )

        # Retrieval agents use their unique examples
        retrieval_result = await self.retrieval.annotate(text, labels)
        hybrid_result = await self.hybrid.annotate(text, labels=labels)

        # ============ Tier 3: Judge Evaluation ============
        # Judge now uses DYNAMIC weights from MetricsCollector
        final_annotation = await self.judge.evaluate(
            task_id=task_id,
            task_type=routing.task_type.value,
            primary_only_result=primary_result.to_dict(),
            contextual_result=contextual_result.to_dict(),
            retrieval_result=retrieval_result.to_dict(),
            retrieval_mrl_result=hybrid_result.to_dict(),
        )

        # ============ Tier 5: Record Metrics (MAFA Section 4.5) ============
        if self.metrics_collector is not None:
            agent_predictions = {
                "primary_only": primary_result.label,
                "contextual": contextual_result.label,
                "retrieval": retrieval_result.label,
                "retrieval_mrl": hybrid_result.label,
            }
            agent_confidences = {
                "primary_only": primary_result.confidence,
                "contextual": contextual_result.confidence,
                "retrieval": retrieval_result.confidence,
                "retrieval_mrl": hybrid_result.confidence,
            }
            self.metrics_collector.record_annotation(
                task_id=task_id,
                text=text,
                true_label=None,
                agent_predictions=agent_predictions,
                final_label=final_annotation.label,
                consensus_score=final_annotation.consensus_score,
                task_type=routing.task_type.value,
                agent_confidences=agent_confidences,
            )

        # ============ Tier 4: Human-in-the-loop ============
        if self.judge.should_approve(final_annotation):
            self.review_workflow.approve_auto(
                task_id, self._final_to_dict(final_annotation)
            )
            return self._final_to_dict(final_annotation)
        else:
            self.review_queue.add(
                task_id=task_id,
                original_text=text,
                annotation=self._final_to_dict(final_annotation),
                consensus_score=final_annotation.consensus_score,
            )

        return self._final_to_dict(final_annotation)

    async def process_with_few_shot(
        self, text: str, task_id: str | None = None
    ) -> Dict[str, Any]:
        """Process text using dynamic few-shot selection from training pool."""
        task_id = task_id or str(uuid.uuid4())
        task = Task(id=task_id, text=text)

        # Tier 1: Routing
        routing = await self.router.analyze(task)
        labels = self.router.get_label_names(routing)

        # Tier 1: Query expansion
        expanded_text = await self._expand_query(text)

        # Tier 2: Get dynamic few-shot examples
        agent_examples = {}
        if self.few_shot_selector:
            agent_examples = self.few_shot_selector.select_diverse_for_all_agents(
                query=expanded_text,
                task_type=routing.task_type.value,
                k_per_agent=8,
            )

        # Tier 2: Process with few-shot examples
        primary_result = await self.primary_only.annotate(text, labels)
        contextual_result = await self.contextual.annotate(text, "", labels)

        # Retrieval agents use their unique examples
        retrieval_result = await self.retrieval.annotate(text, labels)
        hybrid_result = await self.hybrid.annotate(text, labels=labels)

        # Tier 3: Judge
        final_annotation = await self.judge.evaluate(
            task_id=task_id,
            task_type=routing.task_type.value,
            primary_only_result=primary_result.to_dict(),
            contextual_result=contextual_result.to_dict(),
            retrieval_result=retrieval_result.to_dict(),
            retrieval_mrl_result=hybrid_result.to_dict(),
        )

        # Tier 4: Human review
        if self.judge.should_approve(final_annotation):
            self.review_workflow.approve_auto(
                task_id, self._final_to_dict(final_annotation)
            )
        else:
            self.review_queue.add(
                task_id=task_id,
                original_text=text,
                annotation=self._final_to_dict(final_annotation),
                consensus_score=final_annotation.consensus_score,
            )

        return self._final_to_dict(final_annotation)

    async def process_batch(
        self, texts: List[str], task_type: str = "topic"
    ) -> List[Dict[str, Any]]:
        """Process multiple texts through the pipeline."""
        return await asyncio.gather(*[self.process(text) for text in texts])

    def get_component_stats(self) -> Dict[str, Any]:
        """Get statistics for all MAFA components."""
        stats = {
            "few_shot_selector": self.few_shot_selector.stats()
            if self.few_shot_selector
            else None,
            "query_planner": hasattr(self, "query_planner")
            and self.query_planner is not None,
            "query_expander": self.query_expander.stats()
            if self.query_expander
            else None,
        }
        if self.metrics_collector is not None:
            stats["monitoring"] = self.metrics_collector.get_overall_stats()
            stats["weights"] = self.metrics_collector.get_weight_distribution()
        if self.weight_scheduler is not None:
            stats["scheduler"] = self.weight_scheduler.get_status()
        return stats

    def update_weights(self) -> Dict[str, float]:
        """Trigger weight update based on accumulated metrics.

        Returns the new weights.
        """
        if self.metrics_collector is None:
            return {}
        return self.metrics_collector.update_weights()

    def apply_ground_truth(self, task_id: str, true_label: str):
        """Apply ground truth to a recorded annotation.

        Used when human review provides correct label.
        """
        if self.metrics_collector is not None:
            self.metrics_collector.apply_ground_truth(task_id, true_label)

    def start_weight_scheduler(self):
        """Start the automatic weight update scheduler."""
        if self.weight_scheduler is not None:
            self.weight_scheduler.start()

    def stop_weight_scheduler(self):
        """Stop the automatic weight update scheduler."""
        if self.weight_scheduler is not None:
            self.weight_scheduler.stop()

    def _final_to_dict(self, annotation) -> Dict[str, Any]:
        """Convert FinalAnnotation to dict."""
        return {
            "task_id": annotation.task_id,
            "task_type": annotation.task_type,
            "label": annotation.label,
            "entities": annotation.entities,
            "consensus_score": annotation.consensus_score,
            "decision": annotation.decision,
            "agent_votes": annotation.agent_votes,
            "audit_trail": annotation.audit_trail,
        }


async def main():
    """Main entry point for the annotation system."""
    pipeline = AnnotationPipeline()

    # Show component stats
    print("\n" + "=" * 60)
    print("MAFA Multi-Agent Annotation System")
    print("=" * 60)
    stats = pipeline.get_component_stats()
    print(f"Query Planner: {'✓' if stats['query_planner'] else '✗'}")
    print(f"Query Expander: {'✓' if stats['query_expander'] else '✗'}")
    if stats.get("few_shot_selector"):
        fs_stats = stats["few_shot_selector"]
        print(f"Few-Shot Selector: {fs_stats['total_examples']} examples")

    # Show monitoring stats
    if stats.get("monitoring"):
        print(f"\n{'=' * 60}")
        print("Production Monitoring (MAFA Section 4.5)")
        print("=" * 60)
        mon = stats["monitoring"]
        print(f"Total Annotations: {mon['total_annotations']}")
        print(f"System Accuracy: {mon['system_accuracy']:.2%}")
        print(f"\nAgent Weights:")
        for agent, weight in sorted(stats.get("weights", {}).items()):
            perf = mon.get("agent_performance", {}).get(agent, {})
            acc = perf.get("accuracy", "N/A")
            print(f"  {agent}: {weight:.4f} (accuracy: {acc})")

    # Load samples from data/train.csv or use generic fallback
    sample_texts = []
    try:
        import csv

        train_path = DATA_DIR / "train.csv"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames
                # Try to find text column dynamically
                text_col = next(
                    (
                        c
                        for c in cols
                        if c.lower() in ["comment", "text", "review", "content"]
                    ),
                    None,
                )

                if text_col:
                    count = 0
                    for row in reader:
                        if row[text_col]:
                            sample_texts.append(row[text_col])
                            count += 1
                        if count >= 4:
                            break
    except Exception as e:
        print(f"Warning: Could not load samples from train.csv: {e}")

    # Generic fallback if data load fails
    if not sample_texts:
        print(
            "Error: Could not load any samples from train.csv. Please ensure data/train.csv exists and has valid headers."
        )
        return

    print("\n" + "=" * 60)
    print("Processing samples...")
    print("=" * 60)

    results = await pipeline.process_batch(sample_texts)

    for text, result in zip(sample_texts, results):
        print(f"\nText: {text[:50]}...")
        print(f"  Label: {result['label']}")
        print(f"  Consensus: {result['consensus_score']:.2f}")
        print(f"  Decision: {result['decision']}")

    print(f"\n{'=' * 60}")
    print(f"Review Queue Size: {pipeline.review_queue.size()}")

    # Show updated metrics
    if pipeline.metrics_collector:
        print(f"\n{'=' * 60}")
        print("After Processing:")
        print("=" * 60)
        final_stats = pipeline.metrics_collector.get_overall_stats()
        print(f"Total Annotations: {final_stats['total_annotations']}")
        print(f"Agent Weights: {final_stats['agent_weights']}")


if __name__ == "__main__":
    asyncio.run(main())
