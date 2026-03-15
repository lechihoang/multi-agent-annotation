"""
DREAM Relabel Runner

Usage:
    uv run python run_relabel.py --input data/train.csv --output data/train_relabeled.csv
    uv run python run_relabel.py --input data/val.csv --output data/val_relabeled.csv --concurrency 8
"""

import argparse
import asyncio
import csv
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from src.graphs import DreamResult, relabel_dataset


DATA_DIR = Path("data")


def load_input_csv(path: Path, text_col: str = "review", label_col: str = "label"):
    """Load input CSV and return texts, labels, indices."""
    texts, labels, indices = [], [], []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            texts.append(row.get(text_col, ""))
            labels.append(row.get(label_col, ""))
            indices.append(str(i))
    return texts, labels, indices


def count_done(output_path: Path) -> int:
    """Count already done samples."""
    if not output_path.exists():
        return 0
    with open(output_path, encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


async def main():
    parser = argparse.ArgumentParser(description="DREAM Relabel Dataset")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    parser.add_argument(
        "--text-column", type=str, default="review", help="Column name for text"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column name for existing label",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Concurrent annotations (default: 5)"
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0=all)")
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from existing output"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG", colorize=True)

    # Calculate optimal concurrency from rate limit
    rate_limit = 40  # requests per minute from config
    # DREAM: 2 agents × max_rounds (typically 2) + 1 adjudicator if needed
    expected_calls_per_sample = 4  # Agent1_R1, Agent2_R1, Agent1_R2, Agent2_R2 (worst case)
    optimal_concurrency = int((rate_limit / 60) / expected_calls_per_sample * 0.9)
    optimal_concurrency = max(1, min(optimal_concurrency, args.concurrency))

    logger.info(f"Rate limit: {rate_limit} req/min")
    logger.info(f"Expected calls/sample (worst case): {expected_calls_per_sample}")
    logger.info(
        f"Using concurrency: {optimal_concurrency} (requested: {args.concurrency})"
    )

    # Load input
    texts, labels, indices = load_input_csv(
        args.input, text_col=args.text_column, label_col=args.label_column
    )
    logger.info(f"Input: {len(texts)} samples from {args.input}")

    # Check existing output
    existing = count_done(args.output)
    logger.info(f"Output: {args.output} ({existing} already done)")

    # Apply limit
    if args.limit > 0:
        texts, labels, indices = (
            texts[: args.limit],
            labels[: args.limit],
            indices[: args.limit],
        )
        logger.info(f"Limited to: {args.limit} samples")

    if not texts:
        logger.info("No samples to process")
        return

    start_time = time.monotonic()

    # Run relabeling
    results = await relabel_dataset(
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        label_column=args.label_column,
        concurrency=optimal_concurrency,
        limit=args.limit,
        resume=not args.no_resume,
    )

    elapsed = time.monotonic() - start_time

    # Statistics
    if results:
        label_1 = sum(1 for r in results if r.final_label == "1")
        label_0 = sum(1 for r in results if r.final_label == "0")
        agreements = sum(1 for r in results if r.reached_agreement)
        adjudicated = sum(1 for r in results if r.used_adjudicator)

        # Agreement with old labels
        disagreements = sum(
            1 for r in results if r.old_label and r.final_label != r.old_label
        )

        logger.info(f"=== DREAM Results ===")
        logger.info(f"Processed: {len(results)} samples")
        logger.info(f"Time: {elapsed / 60:.1f} minutes")
        logger.info(f"Rate: {len(results) / elapsed * 60:.1f} samples/min")
        logger.info(f"Label 1: {label_1} ({label_1 / len(results) * 100:.1f}%)")
        logger.info(f"Label 0: {label_0} ({label_0 / len(results) * 100:.1f}%)")
        logger.info(f"Agreements: {agreements} ({agreements / len(results) * 100:.1f}%)")
        logger.info(f"Adjudicated: {adjudicated} ({adjudicated / len(results) * 100:.1f}%)")
        logger.info(
            f"Disagreements (vs old): {disagreements} ({disagreements / len(results) * 100:.1f}%)"
        )
    else:
        logger.info("No new results processed")


if __name__ == "__main__":
    asyncio.run(main())
