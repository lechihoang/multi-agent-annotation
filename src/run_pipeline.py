"""MAFA Pipeline - Continuous Processing Loop.

Full MAFA pipeline with continuous processing:
1. Process texts in batch
2. Record metrics for each annotation
3. Periodically update weights based on accuracy
4. Human review integration
5. Auto-save state

Usage:
    python -m src.run_pipeline --continuous --interval 60
"""

import asyncio
import argparse
import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

from .main import AnnotationPipeline, DATA_DIR
from .config import get_config


class MAFALoop:
    """Continuous MAFA processing loop with monitoring."""

    def __init__(
        self,
        input_file: str = str(DATA_DIR / "train.csv"),
        output_file: str = str(DATA_DIR / "annotations.json"),
        ground_truth_file: Optional[str] = None,
        batch_size: int = 50,
        weight_update_interval: int = 10,  # Update weights every N batches
        auto_review: bool = False,
    ):
        """Initialize the MAFA processing loop.

        Args:
            input_file: Path to input CSV with texts to annotate
            output_file: Path to save annotations
            ground_truth_file: Path to ground truth labels (if available)
            batch_size: Number of texts to process per batch
            weight_update_interval: Update weights every N batches
            auto_review: Automatically approve high-confidence results
        """
        self.input_file = input_file
        self.output_file = output_file
        self.ground_truth_file = ground_truth_file
        self.batch_size = batch_size
        self.weight_update_interval = weight_update_interval
        self.auto_review = auto_review

        self.pipeline = AnnotationPipeline()
        self.annotations: List[Dict] = []
        self.ground_truth_map: Dict[str, str] = {}

        # Load ground truth if available
        if ground_truth_file and os.path.exists(ground_truth_file):
            self._load_ground_truth()

        # Statistics
        self.stats = {
            "total_processed": 0,
            "auto_approved": 0,
            "human_review": 0,
            "total_batches": 0,
            "weight_updates": 0,
            "start_time": None,
        }

    def _load_ground_truth(self):
        """Load ground truth labels from file."""
        try:
            with open(self.ground_truth_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_id = row.get("task_id", row.get("id"))
                    label = row.get("label", row.get("Toxicity"))
                    if task_id and label:
                        self.ground_truth_map[task_id] = label
            print(f"✓ Loaded {len(self.ground_truth_map)} ground truth labels")
        except Exception as e:
            print(f"Warning: Could not load ground truth: {e}")

    def _load_texts(self) -> List[Dict]:
        """Load texts from input file."""
        texts = []
        try:
            with open(self.input_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    texts.append(
                        {
                            "id": row.get("Unnamed: 0", str(len(texts))),
                            "text": row.get("Comment", ""),
                            "title": row.get("Title", ""),
                            "true_label": row.get("Toxicity", ""),
                        }
                    )
        except Exception as e:
            print(f"Error loading texts: {e}")
        return texts

    def _apply_ground_truth(self, annotation: Dict):
        """Apply ground truth to annotation and update metrics."""
        task_id = annotation.get("task_id")
        true_label = self.ground_truth_map.get(task_id)

        if true_label and self.pipeline.metrics_collector:
            self.pipeline.apply_ground_truth(task_id, true_label)

            # Check if prediction was correct
            predicted = annotation.get("label")
            is_correct = predicted == true_label
            print(
                f"  Ground truth: {true_label}, Predicted: {predicted} -> {'✓' if is_correct else '✗'}"
            )

    async def run_batch(self, texts: List[Dict]) -> List[Dict]:
        """Process a single batch of texts."""
        results = []
        for item in texts:
            try:
                result = await self.pipeline.process(item["text"])
                result["task_id"] = item.get("id", f"task_{len(results)}")
                result["original_text"] = item["text"]

                # Apply ground truth if available
                self._apply_ground_truth(result)

                results.append(result)
                self.stats["total_processed"] += 1

                if self.pipeline.judge.should_approve(
                    type("FakeAnnotation", (), result)()
                ):
                    self.stats["auto_approved"] += 1
                else:
                    self.stats["human_review"] += 1

            except Exception as e:
                print(f"Error processing text: {e}")

        return results

    async def run_continuous(self, max_batches: int = 1000):
        """Run continuous processing loop.

        Args:
            max_batches: Maximum number of batches to process (0 = infinite)
        """
        self.stats["start_time"] = datetime.now()

        # Load all texts
        all_texts = self._load_texts()
        if not all_texts:
            print("No texts to process!")
            return

        print(f"\n{'=' * 70}")
        print("MAFA Continuous Processing Loop")
        print("=" * 70)
        print(f"Total texts: {len(all_texts)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Weight update interval: {self.weight_update_interval} batches")
        print(f"Ground truth available: {len(self.ground_truth_map)} labels")
        print(f"Auto-review: {'Yes' if self.auto_review else 'No'}")
        print(f"\nStarting at {self.stats['start_time'].isoformat()}")
        print("=" * 70)

        # Start weight scheduler
        self.pipeline.start_weight_scheduler()

        batch_num = 0
        text_idx = 0

        try:
            while text_idx < len(all_texts):
                batch_num += 1
                self.stats["total_batches"] = batch_num

                # Get batch
                batch = all_texts[text_idx : text_idx + self.batch_size]
                text_idx += self.batch_size

                print(f"\n[Batch {batch_num}] Processing {len(batch)} texts...")

                # Process batch
                results = await self.run_batch(batch)
                self.annotations.extend(results)

                # Save state
                self._save_state()

                # Print batch stats
                stats = self.pipeline.get_component_stats()
                if stats.get("monitoring"):
                    mon = stats["monitoring"]
                    print(
                        f"  -> Total: {mon['total_annotations']}, Accuracy: {mon['system_accuracy']:.1%}"
                    )

                # Periodic weight update
                if batch_num % self.weight_update_interval == 0:
                    print(f"\n{'=' * 50}")
                    print(f"[Weight Update - Batch {batch_num}]")
                    print("=" * 50)

                    new_weights = self.pipeline.update_weights()
                    self.stats["weight_updates"] += 1

                    # Show agent performance
                    if stats.get("monitoring"):
                        for agent, weight in sorted(new_weights.items()):
                            perf = (
                                stats["monitoring"]
                                .get("agent_performance", {})
                                .get(agent, {})
                            )
                            acc = perf.get("accuracy", "N/A")
                            print(f"  {agent}: {weight:.4f} (accuracy: {acc})")

                # Check max batches
                if max_batches > 0 and batch_num >= max_batches:
                    print(f"\nReached max batches ({max_batches})")
                    break

                # Small delay between batches
                await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Stop scheduler
            self.pipeline.stop_weight_scheduler()

            # Final stats
            self._print_final_stats()

    def _save_state(self):
        """Save current state to disk."""
        try:
            # Save annotations
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)

            # Export metrics
            if self.pipeline.metrics_collector:
                self.pipeline.metrics_collector.export_metrics()

        except Exception as e:
            print(f"Warning: Failed to save state: {e}")

    def _print_final_stats(self):
        """Print final statistics."""
        end_time = datetime.now()
        duration = end_time - self.stats["start_time"]

        print(f"\n{'=' * 70}")
        print("Final Statistics")
        print("=" * 70)
        print(f"Duration: {duration}")
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"Batches: {self.stats['total_batches']}")
        print(f"Auto-approved: {self.stats['auto_approved']}")
        print(f"Human review: {self.stats['human_review']}")
        print(f"Weight updates: {self.stats['weight_updates']}")

        # Final metrics
        stats = self.pipeline.get_component_stats()
        if stats.get("monitoring"):
            mon = stats["monitoring"]
            print(f"\nSystem Accuracy: {mon['system_accuracy']:.1%}")
            print(f"\nFinal Agent Weights:")
            for agent, weight in sorted(stats.get("weights", {}).items()):
                perf = mon.get("agent_performance", {}).get(agent, {})
                acc = perf.get("accuracy", "N/A")
                print(f"  {agent}: {weight:.4f} (accuracy: {acc})")

        print(f"\nSaved {len(self.annotations)} annotations to: {self.output_file}")
        print("=" * 70)


async def main():
    """Main entry point for continuous processing."""
    parser = argparse.ArgumentParser(description="MAFA Continuous Processing Loop")
    parser.add_argument(
        "--input", "-i", default=str(DATA_DIR / "train.csv"), help="Input CSV file"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DATA_DIR / "annotations.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--ground-truth", "-g", default=None, help="Ground truth CSV file"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=20, help="Batch size")
    parser.add_argument(
        "--weight-interval",
        "-w",
        type=int,
        default=5,
        help="Weight update interval (batches)",
    )
    parser.add_argument(
        "--max-batches", "-m", type=int, default=0, help="Max batches (0 = infinite)"
    )
    parser.add_argument(
        "--auto-review",
        "-a",
        action="store_true",
        help="Auto-approve high-confidence results",
    )

    args = parser.parse_args()

    loop = MAFALoop(
        input_file=args.input,
        output_file=args.output,
        ground_truth_file=args.ground_truth,
        batch_size=args.batch_size,
        weight_update_interval=args.weight_interval,
        auto_review=args.auto_review,
    )

    await loop.run_continuous(max_batches=args.max_batches)


if __name__ == "__main__":
    asyncio.run(main())
