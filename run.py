"""
DREAM: Multi-Agent Debate Annotation

Usage:
    uv run python run.py --input data/unlabeled.csv --output data/annotated.csv
    uv run python run.py --input data/unlabeled.csv --concurrency 5 --limit 10

Based on: arXiv:2602.06526 - DREAM: Debate-based RElevance Assessment with Multi-agents

Flow:
  1. Two agents with OPPOSING STANCES debate (complaint vs non-complaint)
  2. Multi-round reciprocal critique (R=2 rounds by default)
  3. Agreement → use agreed label | Disagreement → LLM adjudicator
"""

import argparse
import asyncio
import csv
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from src.graphs import annotate_with_dream, DreamResult
from src.config import get_config

DATA_DIR = Path("data")


def load_done_indices(output_path: Path) -> set[int]:
    """Load done indices from output file for resume."""
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get("idx", "")
            if idx.isdigit():
                done.add(int(idx))
    logger.info(f"Resume: found {len(done)} already done in {output_path}")
    return done


def load_input_csv(path: Path) -> list[dict]:
    """Load input CSV with auto-detect columns."""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # Auto-detect text column
        text_col = next((c for c in cols if c.lower() in ["review", "comment", "text", "content"]), None)
        if not text_col:
            raise ValueError(f"Cannot find text column in {path}. Columns: {cols}")
        # Auto-detect index column
        idx_col = next((c for c in cols if c.lower() in ["idx", "id", "index"]), None)

        rows = []
        for i, row in enumerate(reader):
            rows.append({
                "idx": row.get(idx_col, str(i)) if idx_col else str(i),
                "text": row.get(text_col, ""),
            })
    return rows


def append_result_csv(result: DreamResult, idx: str, path: Path):
    """Append single result to CSV (immediate write)."""
    fieldnames = ["idx", "review", "final_label", "confidence", "reached_agreement", "agreement_round", "reasoning"]
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "idx": idx,
            "review": result.review,
            "final_label": result.final_label,
            "confidence": result.confidence,
            "reached_agreement": result.reached_agreement,
            "agreement_round": result.agreement_round or "",
            "reasoning": result.reasoning,
        })


async def run_batch(
    rows: list[dict],
    output_path: Path,
    concurrency: int,
) -> list[DreamResult]:
    """Run annotation with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Optional[DreamResult]] = [None] * len(rows)
    start = time.monotonic()

    async def process(i: int, row: dict):
        async with semaphore:
            try:
                result = await annotate_with_dream(row["text"], task_id=row["idx"])
            except Exception as e:
                logger.error(f"[{row['idx']}] Error: {e}")
                result = DreamResult(
                    task_id=row["idx"],
                    review=row["text"],
                    final_label="0",
                    confidence=0.0,
                    reasoning=f"Error: {str(e)}",
                    reached_agreement=False,
                )
            results[i] = result
            append_result_csv(result, row["idx"], output_path)

            # Progress
            elapsed = time.monotonic() - start
            done = sum(1 for r in results if r is not None)
            rate = done / elapsed * 60 if elapsed > 0 else 0

            agreements = sum(1 for r in results if r and r.reached_agreement)
            logger.info(f"[{done}/{len(rows)}] {rate:.1f} samples/min | agree: {agreements}/{done}")

    await asyncio.gather(*[process(i, row) for i, row in enumerate(rows)])
    return [r for r in results if r is not None]


async def main():
    parser = argparse.ArgumentParser(description="DREAM Annotation")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "unlabeled.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "annotated.csv")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent (default: 5)")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore existing output)")
    parser.add_argument("--verbose", action="store_true", help="Debug logs")
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG", colorize=True)

    # Load config for rate limit info
    config = get_config()
    rate_limit = config.nvidia.rate_limit
    max_rounds = config.dream.debate.max_rounds

    # Calculate optimal concurrency
    # DREAM: 2 agents × max_rounds + adjudicator (worst case)
    worst_case_calls = 2 * max_rounds + 1
    optimal = int(rate_limit / 60 / worst_case_calls * 0.9)
    optimal = max(1, min(optimal, args.concurrency))

    logger.info("=" * 50)
    logger.info("DREAM: Multi-Agent Debate Annotation")
    logger.info(f"Input:     {args.input}")
    logger.info(f"Output:    {args.output}")
    logger.info(f"Rate limit: {rate_limit} req/min")
    logger.info(f"Concurrency: {optimal} (requested: {args.concurrency})")
    logger.info("=" * 50)

    # Load input
    rows = load_input_csv(args.input)
    logger.info(f"Input total: {len(rows)} samples")

    # Resume: skip already done
    if args.no_resume:
        done_indices = set()
    else:
        done_indices = load_done_indices(args.output)

    pending = [r for r in rows if int(r["idx"]) not in done_indices]
    logger.info(f"Pending: {len(pending)} (skipped {len(done_indices)} already done)")

    if args.limit > 0:
        pending = pending[:args.limit]
        logger.info(f"Limited to: {args.limit} samples")

    if not pending:
        logger.info("Nothing to annotate.")
        return

    # Run
    results = await run_batch(pending, args.output, optimal)

    # Statistics
    label_1 = sum(1 for r in results if r.final_label == "1")
    label_0 = sum(1 for r in results if r.final_label == "0")
    agreements = sum(1 for r in results if r.reached_agreement)

    logger.info("=" * 50)
    logger.info("DONE!")
    logger.info(f"Processed: {len(results)}")
    logger.info(f"Label 1:   {label_1} ({label_1/len(results)*100:.1f}%)")
    logger.info(f"Label 0:   {label_0} ({label_0/len(results)*100:.1f}%)")
    logger.info(f"Agreement: {agreements} ({agreements/len(results)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
