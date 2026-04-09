"""
DREAM — Multi-Agent Debate Annotation Runner.
Usage:
  python run.py --input data/ViCTSD_unlabeled.csv \
    --output data/ViCTSD_annotated.csv --limit 5 --concurrency 2
Based on arXiv:2602.06526 - DREAM: Multi-Agent Debate for NLP Classification
"""

import argparse
import asyncio
import csv
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from src.config import get_config
from src.pipeline import annotate, DreamResult

DATA_DIR = Path("data")


def _log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = row.get("task_id", "").strip()
            if tid and row.get("final_label", ""):
                done.add(tid)
    _log(f"[INFO] Resume: {len(done)} already done in {output_path}")
    return done


def load_csv(path: Path, text_col: str):
    texts, ids = [], []
    with open(path, encoding="utf-8-sig") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            texts.append(row[text_col])
            ids.append(str(uuid.uuid4()))
    return texts, ids


def write_result(result: DreamResult, path: Path):
    fieldnames = [
        "task_id", "text", "final_label", "confidence",
        "reasoning", "reached_agreement", "agreement_round",
        "used_moderator", "used_adjudicator", "needs_human",
        "moderator_agreements", "moderator_disagreements", "moderator_closer",
        "adjudication_confidence", "adjudication_reasoning",
        "agent_a_argument", "agent_a_evidence",
        "agent_b_argument", "agent_b_evidence",
        "debate_rounds_count",
    ]
    write_header = not path.exists()

    mod = result.moderator_summary
    adj = result.adjudication

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "task_id": result.task_id,
            "text": result.text,
            "final_label": result.final_label,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "reached_agreement": result.reached_agreement,
            "agreement_round": result.agreement_round or "",
            "used_moderator": result.used_moderator,
            "used_adjudicator": result.used_adjudicator,
            "needs_human": result.needs_human,
            "moderator_agreements": mod.agreements if mod else "",
            "moderator_disagreements": mod.disagreements if mod else "",
            "moderator_closer": mod.closer_to_label if mod else "",
            "adjudication_confidence": adj.confidence if adj else "",
            "adjudication_reasoning": adj.reasoning if adj else "",
            "agent_a_argument": result.agent_a_final_argument,
            "agent_a_evidence": result.agent_a_final_evidence,
            "agent_b_argument": result.agent_b_final_argument,
            "agent_b_evidence": result.agent_b_final_evidence,
            "debate_rounds_count": len(result.debate_rounds),
        })


async def annotate_batch(
    texts: list[str],
    task_ids: list[str],
    output_path: Path,
    concurrency: int,
    total_done: int,
    verbose: bool = False,
) -> list[DreamResult]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[DreamResult | None] = [None] * len(texts)
    completed = errors = 0
    start = time.monotonic()

    async def run(i: int, text: str, tid: str):
        nonlocal completed, errors
        async with semaphore:
            try:
                result: DreamResult = await annotate(text, tid)
                results[i] = result
            except Exception as e:
                if verbose:
                    _log(f"[ERROR] [{i}] {e}")
                results[i] = DreamResult(
                    task_id=tid, text=text,
                    final_label="0", confidence=0.0,
                    reasoning=f"Error: {str(e)}",
                    reached_agreement=False,
                )
                errors += 1
            finally:
                if results[i]:
                    write_result(results[i], output_path)
                completed += 1
                elapsed = time.monotonic() - start
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                done_total = total_done + completed
                agree = sum(1 for r in results if r and r.reached_agreement)
                human = sum(1 for r in results if r and r.needs_human)
                if verbose:
                    _log(f"[INFO] {done_total} done | batch {completed}/{len(texts)} "
                         f"| errors={errors} | agree={agree} | human_escal={human} "
                         f"| {rate:.1f}/min")

    await asyncio.gather(*[
        run(i, t, tid)
        for i, (t, tid) in enumerate(zip(texts, task_ids))
    ])
    return [r for r in results if r is not None]


async def main():
    parser = argparse.ArgumentParser(description="DREAM — Multi-Agent Debate Annotation")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "annotated.csv")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose

    # Non-verbose: loguru to stderr (background)
    # Verbose: plain print() + flush=True for Kaggle cell visibility
    if verbose:
        _log("[INFO] Starting DREAM pipeline...")

    config = get_config(str(args.config))

    if verbose:
        _log(f"[INFO] Config: {config.task.name} | model: {config.nvidia.model}")
        _log(f"[INFO] Debate: {config.dream.debate.max_rounds} rounds | "
             f"Moderator: {config.dream.moderator.enabled} | "
             f"Escalation: {config.dream.escalation.enabled}")
    else:
        logger.info(f"Config: {config.task.name} | model: {config.nvidia.model}")
        logger.info(f"Debate: {config.dream.debate.max_rounds} rounds | "
                    f"Moderator: {config.dream.moderator.enabled} | "
                    f"Escalation: {config.dream.escalation.enabled}")

    text_col = config.task.text_column
    texts, ids = load_csv(args.input, text_col)

    done_ids = load_done_ids(args.output)
    pending = [(t, i) for t, i in zip(texts, ids) if i not in done_ids]
    total_done = len(done_ids)

    if verbose:
        _log(f"[INFO] Input: {len(texts)} | Pending: {len(pending)} | Done: {total_done}")
    else:
        logger.info(f"Input: {len(texts)} | Pending: {len(pending)} | Done: {total_done}")

    if args.limit > 0:
        pending = pending[:args.limit]
        if verbose:
            _log(f"[INFO] Limit: {args.limit}")
        else:
            logger.info(f"Limit: {args.limit}")

    if not pending:
        if verbose:
            _log("[INFO] Nothing to do.")
        else:
            logger.info("Nothing to do.")
        return

    texts_p, ids_p = zip(*pending)
    results = await annotate_batch(
        texts=list(texts_p),
        task_ids=list(ids_p),
        output_path=args.output,
        concurrency=args.concurrency,
        total_done=total_done,
        verbose=verbose,
    )

    label_1 = sum(1 for r in results if r.final_label == "1")
    label_0 = sum(1 for r in results if r.final_label == "0")
    agree = sum(1 for r in results if r.reached_agreement)
    human = sum(1 for r in results if r.needs_human)
    errors = sum(1 for r in results if r.confidence == 0.0)
    total = len(results)

    if verbose:
        _log("=" * 50)
        _log("[INFO] DONE")
        _log(f"[INFO] Processed: {total}")
        _log(f"[INFO] Label 1:   {label_1} ({label_1/total*100:.1f}%)")
        _log(f"[INFO] Label 0:   {label_0} ({label_0/total*100:.1f}%)")
        _log(f"[INFO] Agreement: {agree} ({agree/total*100:.1f}%)")
        _log(f"[INFO] Human esc: {human} ({human/total*100:.1f}%)")
        _log(f"[INFO] Errors:    {errors}")
    else:
        logger.info("=" * 50)
        logger.info("DONE")
        logger.info(f"  Processed: {total}")
        logger.info(f"  Label 1:   {label_1} ({label_1/total*100:.1f}%)")
        logger.info(f"  Label 0:   {label_0} ({label_0/total*100:.1f}%)")
        logger.info(f"  Agreement: {agree} ({agree/total*100:.1f}%)")
        logger.info(f"  Human esc: {human} ({human/total*100:.1f}%)")
        logger.info(f"  Errors:    {errors}")


if __name__ == "__main__":
    asyncio.run(main())
