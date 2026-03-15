"""
DREAM Annotation Runner (Multi-Agent Debate)

Usage:
    uv run python run_annotation.py --input data/unlabeled.csv --output data/annotated.csv
    uv run python run_annotation.py --input data/unlabeled.csv --concurrency 8 --verbose

Based on: arXiv:2602.06526 - DREAM: Debate-based RElevance Assessment with Multi-agents

Flow:
  1. Two agents with OPPOSING STANCES debate (relevant vs irrelevant)
  2. Multi-round reciprocal critique (R=2 rounds)
  3. Agreement → auto label | Disagreement → LLM adjudicator
"""

import argparse
import asyncio
import csv
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from src.graphs import annotate_with_dream, DreamResult

DATA_DIR = Path("data")


def load_done_reviews(output_path: Path) -> set[str]:
    """Đọc các review đã annotated từ output file (để resume)."""
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            review = row.get("review", "").strip()
            if review and row.get("final_label", ""):
                done.add(review)
    logger.info(f"Resume: found {len(done)} already annotated in {output_path}")
    return done


def load_input_csv(path: Path) -> tuple[list[str], list[str]]:
    texts, ids = [], []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        text_col = next((c for c in cols if c.lower() in ["review", "comment", "text", "content"]), None)
        id_col = next((c for c in cols if c.lower() in ["id", "task_id"]), None)
        if not text_col:
            raise ValueError(f"Cannot find text column in {path}. Columns: {cols}")
        for row in reader:
            texts.append(row[text_col])
            ids.append(row.get(id_col, str(uuid.uuid4())) if id_col else str(uuid.uuid4()))
    return texts, ids


def append_result_csv(result: DreamResult, path: Path):
    """Ghi ngay từng kết quả vào file (append mode)."""
    fieldnames = [
        "task_id", "review", "final_label", "confidence",
        "reached_agreement", "agreement_round", "reasoning"
    ]
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "task_id": result.task_id,
            "review": result.review,
            "final_label": result.final_label,
            "confidence": result.confidence,
            "reached_agreement": result.reached_agreement,
            "agreement_round": result.agreement_round or "",
            "reasoning": result.reasoning,
        })


async def annotate_all(
    texts: list[str],
    task_ids: list[str],
    output_path: Path,
    concurrency: int = 5,
    total_in_file: int = 0,
) -> list[DreamResult]:
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(texts)
    completed = 0
    errors = 0
    start = time.monotonic()

    async def run(i: int, text: str, tid: str):
        nonlocal completed, errors
        async with semaphore:
            try:
                results[i] = await annotate_with_dream(text, tid)
            except Exception as e:
                logger.error(f"[{i}] Error: {e}")
                results[i] = DreamResult(
                    task_id=tid,
                    review=text,
                    final_label="0",
                    confidence=0.0,
                    reasoning=f"Error: {str(e)}",
                    reached_agreement=False,
                )
                errors += 1
            finally:
                completed += 1
                append_result_csv(results[i], output_path)

                elapsed = time.monotonic() - start
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                done_total = total_in_file + completed

                # Stats for this batch
                agreements = sum(1 for r in results[:completed] if r and r.reached_agreement)

                logger.info(
                    f"Annotated {done_total} | batch: {completed}/{len(texts)} "
                    f"| errors: {errors} | agree: {agreements}/{completed} | "
                    f"{rate:.1f} samples/min"
                )

    await asyncio.gather(*[run(i, t, tid) for i, (t, tid) in enumerate(zip(texts, task_ids))])
    return results


async def main():
    parser = argparse.ArgumentParser(description="DREAM Annotation Pipeline")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "unlabeled.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "annotated.csv")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent annotations")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows to process (0=all)")
    parser.add_argument("--verbose", action="store_true", help="Print debug logs")
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG", colorize=True)

    logger.info(f"Input:       {args.input}")
    logger.info(f"Output:      {args.output}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info("=" * 50)
    logger.info("DREAM: Multi-Agent Debate-based Annotation")
    logger.info("Based on: arXiv:2602.06526")
    logger.info("=" * 50)

    # Resume: skip already annotated
    done_reviews = load_done_reviews(args.output)
    total_in_file = len(done_reviews)

    texts, ids = load_input_csv(args.input)
    logger.info(f"Input total: {len(texts)}")

    # Filter out already done
    pending = [(t, i) for t, i in zip(texts, ids) if t.strip() not in done_reviews]
    logger.info(f"Pending:     {len(pending)} (skipped {len(texts) - len(pending)} already done)")

    if args.limit > 0:
        pending = pending[: args.limit]
        logger.info(f"Limit:       {args.limit}")

    if not pending:
        logger.info("Nothing to annotate.")
        return

    texts_p, ids_p = zip(*pending)
    results = await annotate_all(
        list(texts_p), list(ids_p),
        output_path=args.output,
        concurrency=args.concurrency,
        total_in_file=total_in_file,
    )

    # Final statistics
    label_1 = sum(1 for r in results if r.final_label == "1")
    label_0 = sum(1 for r in results if r.final_label == "0")
    agreements = sum(1 for r in results if r.reached_agreement)
    errors = sum(1 for r in results if not r.final_label)

    logger.info("=" * 50)
    logger.info("DONE!")
    logger.info(f"Processed: {len(results)}")
    logger.info(f"Label 1:   {label_1} ({label_1/len(results)*100:.1f}%)")
    logger.info(f"Label 0:   {label_0} ({label_0/len(results)*100:.1f}%)")
    logger.info(f"Agreement: {agreements} ({agreements/len(results)*100:.1f}%)")
    logger.info(f"Errors:    {errors}")
    logger.info(f"Total in output file: {total_in_file + len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
