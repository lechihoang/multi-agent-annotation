#!/usr/bin/env python3
"""ARQ Batch Runner - Sử dụng đúng ARQ agents với full output capture.

Differences from run_batch.py:
- Sử dụng ARQ agents thực sự (PrimaryOnlyAgent, ContextualAgent, etc.)
- Capture đầy đủ reasoning từ mỗi agent
- Lưu raw LLM responses để debug
- NO FALLBACK - Validate JSON output
"""

import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import shutil
import sys
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, get_llm_client
from src.agents.tier1.query_expander import QueryExpander
from src.agents.tier2.primary_only import PrimaryOnlyAgent
from src.agents.tier2.contextual import ContextualAgent
from src.agents.tier2.retrieval import RetrievalAgent
from src.agents.tier2.retrieval_mrl import RetrievalMrlAgent
from src.agents.tier3.judge import JudgeAgent
from src.agents.tier4.review import ReviewQueue


DATA_DIR = Path(__file__).parent.parent / "data"


class ARQBatchRunner:
    """Batch runner sử dụng đúng ARQ agents."""

    def __init__(self, calls_per_minute: int = 40):
        self.config = get_config()
        self.delay = 60.0 / calls_per_minute
        self.last_call_time = 0

        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )
        logger.add(str(DATA_DIR / "run.log"), rotation="10 MB", level="DEBUG")

        logger.info(f"{'=' * 60}")
        logger.info(f"ARQ Batch Runner (4 ARQ agents)")
        logger.info(f"Provider: {self.config.provider.type}")
        logger.info(f"{'=' * 60}")

        # Initialize ARQ agents
        self.agents = {
            "primary": PrimaryOnlyAgent(),
            "contextual": ContextualAgent(),
            "retrieval": RetrievalAgent(),
            "hybrid": RetrievalMrlAgent(),
        }

        try:
            csv_path = DATA_DIR / "train.csv"
            self.query_expander = (
                QueryExpander(str(csv_path)) if csv_path.exists() else None
            )
            if self.query_expander:
                logger.info("✓ QueryExpander loaded")
        except Exception as e:
            logger.warning(f"⚠ QueryExpander failed: {e}")
            self.query_expander = None

        self.judge = JudgeAgent()
        self.review_queue = ReviewQueue()

        # Storage for raw responses
        self.raw_responses: List[Dict] = []

    async def _rate_limit(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self.last_call_time = time.time()

    def _build_prompt_for_agent(self, agent, text: str) -> str:
        """Build prompt based on agent type."""
        agent_name = type(agent).__name__

        # Get labels from config dynamically
        labels_config = getattr(self.config.task, "labels", None)
        if labels_config is None:
            raise ValueError("Task labels not configured in config.yaml")

        if isinstance(labels_config, dict):
            config_labels = list(labels_config.keys())
        elif isinstance(labels_config, list):
            config_labels = labels_config
        else:
            raise ValueError(f"Invalid format for task.labels: {type(labels_config)}")

        if "PrimaryOnlyAgent" in agent_name:
            # PrimaryOnlyAgent: _build_prompt(text, labels, few_shot)
            return agent._build_prompt(
                text=text,
                labels=config_labels,
                few_shot_examples=None,
            )
        elif "ContextualAgent" in agent_name:
            # ContextualAgent: _build_prompt(text, title, labels, few_shot)
            return agent._build_prompt(
                text=text,
                title="",
                labels=config_labels,
                few_shot_examples=None,
            )
        elif "RetrievalAgent" in agent_name or "RetrievalMrlAgent" in agent_name:
            # Retrieval agents: _build_mafa_prompt(text, nearest, labels)
            # Use empty nearest list for now (can be enhanced with actual retrieval)
            return agent._build_mafa_prompt(
                text=text,
                nearest=[],
                labels=config_labels,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_name}")

    async def _tier1_batch(self, texts: List[str]) -> List[str]:
        """Tier 1: Query expansion."""
        if self.query_expander:
            return [self.query_expander.expand(t) for t in texts]

        prompt = f"Mở rộng query cho {len(texts)} comments:\n"
        for i, t in enumerate(texts):
            prompt += f'{i + 1}. "{t}"\n'
        prompt += '\nJSON: [{"expanded": "..."}, ...]'

        await self._rate_limit()
        resp = await get_llm_client(self.config).chat(
            [{"role": "user", "content": prompt}]
        )
        content = resp.content.strip()

        if content.startswith("```json"):
            content = content[7:-3]

        data = json.loads(content)
        if isinstance(data, list):
            return [item.get("expanded", texts[i]) for i, item in enumerate(data)]

        raise ValueError(
            f"Invalid Tier 1 response format (not a list): {content[:100]}..."
        )

    async def _run_agent(self, text: str, agent_name: str, agent) -> Dict[str, Any]:
        """Chạy một agent và capture đầy đủ output."""
        start_time = time.time()

        # Build prompt based on agent type
        prompt = self._build_prompt_for_agent(agent, text)

        # Call LLM
        await self._rate_limit()

        client = getattr(agent, "_llm_client", getattr(agent, "_groq", None))

        if client is None:
            raise ValueError(f"Agent {agent_name} has no initialized LLM client")

        response = await client.chat([{"role": "user", "content": prompt}])
        raw_response = response.content

        # Parse ARQ response
        parsed = agent._parse_response(raw_response)

        elapsed = time.time() - start_time

        result = {
            "agent": agent_name,
            "text": text,
            "label": parsed.get("topic", "unknown"),
            "confidence": parsed.get("confidence", 0.5),
            "confidence_level": parsed.get("confidence_level", "MEDIUM"),
            "reasoning": parsed.get("reasoning", "")[:500],
            "elapsed_seconds": round(elapsed, 2),
            "raw_response": raw_response[:1000],
            "prompt_used": prompt[:500],
        }

        return result

    async def process_batch(self, texts: List[str], batch_num: int = 1) -> List[Dict]:
        n = len(texts)
        results = []

        # ===== TIER 1 =====
        logger.info(f"  [Tier 1] Query expansion for {n} samples...")
        expanded = await self._tier1_batch(texts)

        for i, t in enumerate(texts):
            results.append(
                {
                    "text": t,
                    "expanded": expanded[i],
                    "batch": batch_num,
                    "task_id": f"b{batch_num}_{i}",
                }
            )

        # ===== TIER 2: 4 ARQ AGENTS =====
        logger.info(f"  [Tier 2] Running 4 ARQ agents...")

        batch_responses = {
            "primary": [],
            "contextual": [],
            "retrieval": [],
            "hybrid": [],
        }

        for i, text in enumerate(texts):
            task_id = f"b{batch_num}_{i}"
            logger.debug(f'    Sample {i + 1}/{n}: "{text[:50]}..."')

            # Log Tier 1 info to raw responses
            self.raw_responses.append(
                {
                    "task_id": task_id,
                    "tier": 1,
                    "agent": "query_expander",
                    "input": text,
                    "output": expanded[i],
                    "method": "embedding" if self.query_expander else "llm_fallback",
                }
            )

            # Run all 4 agents in parallel for this text
            tasks = [
                self._run_agent(text, name, agent)
                for name, agent in self.agents.items()
            ]

            agent_results = await asyncio.gather(*tasks)

            for r in agent_results:
                batch_responses[r["agent"]].append(
                    {
                        "label": r["label"],
                        "confidence": r["confidence"],
                        "confidence_level": r.get("confidence_level"),
                        "reasoning": r.get("reasoning", ""),
                    }
                )

                # Store raw response for debugging (Tier 2)
                self.raw_responses.append(
                    {
                        "task_id": task_id,
                        "tier": 2,
                        "agent": r["agent"],
                        "prompt": r.get("prompt_used", "")[:500],
                        "raw_response": r.get("raw_response", ""),
                        "parsed": {
                            "label": r.get("label"),
                            "confidence": r.get("confidence"),
                            "reasoning": r.get("reasoning", "")[:200],
                        },
                        "error": r.get("error"),
                    }
                )

        # Attach tier2 results to each sample
        for i, r in enumerate(results):
            r["tier2"] = {
                "primary": batch_responses["primary"][i],
                "contextual": batch_responses["contextual"][i],
                "retrieval": batch_responses["retrieval"][i],
                "hybrid": batch_responses["hybrid"][i],
            }

        # ===== TIER 3 =====
        logger.info(f"  [Tier 3] Judge consensus...")

        for i, r in enumerate(results):
            tier2 = r["tier2"]

            # Calculate weighted average confidence
            confs = [
                tier2["primary"]["confidence"],
                tier2["contextual"]["confidence"],
                tier2["retrieval"]["confidence"],
                tier2["hybrid"]["confidence"],
            ]
            avg_conf = sum(confs) / 4

            # Judge decision (simplified)
            if avg_conf >= 0.85:
                decision = "approve"
            elif avg_conf < 0.60:
                decision = "escalate"
            else:
                decision = "review"

            # Determine final label (majority vote)
            labels = [
                tier2[a]["label"]
                for a in ["primary", "contextual", "retrieval", "hybrid"]
            ]
            from collections import Counter

            label_counts = Counter(labels)
            final_label = label_counts.most_common(1)[0][0]

            r["tier3"] = {
                "label": final_label,
                "confidence": avg_conf,
                "decision": decision,
                "agent_agreement": dict(label_counts),
            }

            # Log Tier 3 info to raw responses
            self.raw_responses.append(
                {
                    "task_id": r["task_id"],
                    "tier": 3,
                    "agent": "judge",
                    "input": {"avg_confidence": avg_conf, "votes": dict(label_counts)},
                    "output": {"final_label": final_label, "decision": decision},
                }
            )

            # Tier 4: Review queue
            try:
                self.review_queue.add(
                    task_id=r["task_id"],
                    original_text=r["text"],
                    annotation=r["tier3"],
                    consensus_score=avg_conf,
                )
            except:
                pass

        return results


async def run(
    input_file: str = str(DATA_DIR / "train.csv"),
    output_file: str = str(DATA_DIR / "batch_arq_results.csv"),
    raw_output_file: str = str(DATA_DIR / "debug_trace.json"),
    batch_size: int = 5,
    max_samples: int = 5,
    calls_per_minute: int = 40,
    batch_delay: int = 60,
):
    runner = ARQBatchRunner(calls_per_minute)

    # Output file path
    output_path = Path(output_file)

    # Check existing progress to resume
    existing_count = 0
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                rows = list(reader)
                existing_count = len(rows) - 1

                if existing_count < 0:
                    existing_count = 0

            if existing_count > 0:
                logger.info(f"Found existing output at {output_path.absolute()}")
                logger.info(
                    f"Total rows in file: {len(rows)} (Header + {existing_count} samples)"
                )
                logger.info(f"Resuming from sample {existing_count + 1}...")
            else:
                logger.info(
                    f"Output file exists but is empty (only header). Starting from beginning."
                )

        except Exception as e:
            logger.error(f"Error checking existing file: {e}")
            existing_count = 0
    else:
        logger.info(
            f"No existing output file found at {output_path.absolute()}. Starting new."
        )

    # Load texts
    texts = []
    skipped = 0
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        # STRICT MODE: Use column from config ONLY
        config_col = getattr(runner.config.task.columns, "text", None)
        if config_col is None:
            raise ValueError(
                "Text column not configured in config.yaml (task.columns.text)"
            )

        if config_col not in cols:
            logger.error(
                f"Configured text column '{config_col}' not found in {input_file}. Available columns: {cols}"
            )
            logger.error(
                "Please update config.yaml (task.columns.text) or check your input file."
            )
            raise ValueError(f"Column '{config_col}' missing in input file")

        text_col = config_col

        for row in reader:
            # Skip processed samples
            if skipped < existing_count:
                skipped += 1
                continue

            t = row.get(text_col, "").strip()
            if t:
                texts.append(t)

            if max_samples > 0 and len(texts) >= max_samples:
                break

    print(f"\nLoaded {len(texts)} new samples to process")
    print(f"Batch size: {batch_size}")
    print(f"Rate: {calls_per_minute}/min")
    print(f"{'=' * 60}\n")

    total_start_time = time.time()

    all_results = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_num = start // batch_size + 1
        total = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"Batch {batch_num}/{total} ({len(batch)} samples)")
        results = await runner.process_batch(batch, batch_num)
        all_results.extend(results)

        # Save results to CSV (Simplified format)
        csv_file = str(Path(output_file).with_suffix(".csv"))
        csv_headers = ["text", "final_label", "confidence", "decision"]

        file_exists = Path(csv_file).exists()
        mode = "a" if file_exists else "w"

        try:
            with open(csv_file, mode, newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                if not file_exists:
                    writer.writeheader()

                for r in results:
                    writer.writerow(
                        {
                            "text": r["text"],
                            "final_label": r["tier3"]["label"],
                            "confidence": r["tier3"]["confidence"],
                            "decision": r["tier3"]["decision"],
                        }
                    )
            logger.info(f"Saved {len(results)} results to {csv_file}")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")

        try:
            with open(raw_output_file, "w", encoding="utf-8") as f:
                json.dump(runner.raw_responses, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved debug trace to {raw_output_file}")
        except Exception as e:
            logger.error(f"Error saving debug trace: {e}")

        if batch_delay > 0:
            logger.info(
                f"Waiting {batch_delay}s before next batch to avoid 429 errors..."
            )
            await asyncio.sleep(batch_delay)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    decisions = {"approve": 0, "review": 0, "escalate": 0}
    for r in all_results:
        d = r["tier3"].get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1

    total = len(all_results)
    print(f"Total: {total}")
    print(
        f"Approve:  {decisions['approve']} ({decisions['approve'] / total * 100:.1f}%)"
    )
    print(f"Review:   {decisions['review']} ({decisions['review'] / total * 100:.1f}%)")
    print(
        f"Escalate: {decisions['escalate']} ({decisions['escalate'] / total * 100:.1f}%)"
    )

    csv_out = str(Path(output_file).with_suffix(".csv"))
    print(f"\nSaved (CSV): {csv_out}")
    print(f"Saved (Debug Trace): {raw_output_file}")

    # Time statistics
    total_duration = time.time() - total_start_time
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)

    print(f"\n{'=' * 60}")
    print(f"TIMING STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total Processing Time: {minutes}m {seconds}s")

    if total > 0:
        avg_time = total_duration / total
        print(f"Average Time per Sample: {avg_time:.2f}s")
        print(f"Estimated Throughput: {60 / avg_time:.2f} samples/minute")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/train.csv")
    parser.add_argument(
        "--output", "-o", default="data/batch_arq_results.csv"
    )  # Default to CSV
    parser.add_argument("--raw", "-R", default="data/debug_trace.json")
    parser.add_argument("--batch-size", "-b", type=int, default=5)
    parser.add_argument("--max", "-m", type=int, default=5)
    parser.add_argument("--rate", "-r", type=int, default=40)
    parser.add_argument(
        "--batch-delay",
        "-d",
        type=int,
        default=60,
        help="Delay in seconds between batches to avoid rate limits",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            args.input,
            args.output,
            args.raw,
            args.batch_size,
            args.max,
            args.rate,
            args.batch_delay,
        )
    )
