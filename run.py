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
import logging

from dotenv import load_dotenv

load_dotenv()

from src.config import get_config
from src.pipeline import annotate, DreamResult

DATA_DIR = Path("data")


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = row.get("task_id", "").strip()
            final_label = str(row.get("final_label", "")).strip()
            reasoning = str(row.get("reasoning", "")).strip()
            needs_human = str(row.get("needs_human", "")).strip().lower() == "true"
            
            # If it's an error, don't mark as done, so we can retry
            if "Error:" in reasoning:
                continue
                
            if tid and (final_label != "" or needs_human):
                done.add(tid)
    sys.stderr.write(f"[INFO] Resume: {len(done)} already done in {output_path}\n")
    sys.stderr.flush()
    return done


def load_human_done_ids(human_file: Path) -> set[str]:
    if not human_file.exists():
        return set()
    done = set()
    with open(human_file, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = str(row.get("task_id", "")).strip()
            final_label = str(row.get("final_label", "")).strip()
            if tid and final_label != "":
                done.add(tid)
    if done:
        sys.stderr.write(f"[INFO] Human labels loaded: {len(done)} from {human_file}\n")
        sys.stderr.flush()
    return done


def write_human_review_file(result: DreamResult, human_file: Path):
    fieldnames = [
        "task_id", "text", "final_label", "confidence", "reasoning",
        "agent_a_argument", "agent_a_evidence",
        "agent_b_argument", "agent_b_evidence",
        "human_label", "human_note",
    ]
    write_header = not human_file.exists()
    with open(human_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "task_id": result.task_id,
            "text": result.text,
            "final_label": result.final_label,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "agent_a_argument": result.agent_a_final_argument,
            "agent_a_evidence": result.agent_a_final_evidence,
            "agent_b_argument": result.agent_b_final_argument,
            "agent_b_evidence": result.agent_b_final_evidence,
            "human_label": "",
            "human_note": "",
        })


def prune_human_review_file(human_file: Path, done_human_ids: set[str]):
    if not human_file.exists() or not done_human_ids:
        return
    with open(human_file, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []
    kept = [r for r in rows if str(r.get("task_id", "")).strip() not in done_human_ids]
    if len(kept) == len(rows):
        return
    with open(human_file, "w", encoding="utf-8", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
    sys.stderr.write(
        f"[INFO] Human review queue pruned: removed {len(rows) - len(kept)} resolved rows from {human_file}\n"
    )
    sys.stderr.flush()


def _text_to_id(text: str, row_index: int, source_name: str) -> str:
    """Deterministic task_id from (source, row_index, text) for robust resume with duplicate texts."""
    payload = f"{source_name}|{row_index}|{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, payload))


def load_csv(path: Path, text_col: str):
    texts, ids = [], []
    with open(path, encoding="utf-8-sig") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            text = row[text_col]
            texts.append(text)
            ids.append(_text_to_id(text, idx, path.name))
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
            "used_moderator": mod is not None,
            "used_adjudicator": adj is not None,
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


def write_human_queue(result: DreamResult, path: Path):
    fieldnames = [
        "task_id", "text", "final_label", "confidence", "reasoning",
        "reached_agreement", "agreement_round", "needs_human",
        "agent_a_argument", "agent_a_evidence",
        "agent_b_argument", "agent_b_evidence",
        "debate_rounds_count",
    ]
    write_header = not path.exists()
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
            "needs_human": result.needs_human,
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
    escalated_output_path: Path,
    concurrency: int,
    total_done: int,
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
                sys.stderr.write(f"[ERROR] [{i}] {e}\n")
                sys.stderr.flush()
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
                    if results[i].needs_human:
                        write_human_queue(results[i], escalated_output_path)
                        write_human_review_file(results[i], escalated_output_path.with_name(
                            f"{escalated_output_path.stem}_for_human_labeling{escalated_output_path.suffix}"
                        ))
                completed += 1
                elapsed = time.monotonic() - start
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                done_total = total_done + completed
                agree = sum(1 for r in results if r and r.reached_agreement)
                human = sum(1 for r in results if r and r.needs_human)
                sys.stderr.write(
                    f"[INFO] {done_total} done | batch {completed}/{len(texts)} "
                    f"| errors={errors} | agree={agree} | human_escal={human} "
                    f"| {rate:.1f}/min\n"
                )
                sys.stderr.flush()

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
    args = parser.parse_args()

    config = get_config(str(args.config))
    # Set logging level
    numeric_level = getattr(logging, config.logging_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger(__name__)

    sys.stderr.write(f"[INFO] Config: {config.task.name} | model: {config.nvidia.model}\n")
    sys.stderr.write(f"[INFO] Debate: {config.dream.debate.max_rounds} rounds | "
                     f"Moderator: {config.dream.moderator.enabled} | "
                     f"Escalation mode: {config.dream.escalation.mode}\n")
    sys.stderr.flush()
    logger.info(
        "Starting run | input=%s output=%s concurrency=%s",
        args.input,
        args.output,
        args.concurrency,
    )

    text_col = config.task.text_column
    texts, ids = load_csv(args.input, text_col)
    escalated_output_path = Path(config.dream.escalation.export_file)

    done_ids = load_done_ids(args.output)
    human_done_file = escalated_output_path.with_name(
        f"{escalated_output_path.stem}_for_human_labeling{escalated_output_path.suffix}"
    )
    done_human_ids = load_human_done_ids(human_done_file)
    prune_human_review_file(human_done_file, done_human_ids)
    done_ids |= done_human_ids
    pending = [(t, i) for t, i in zip(texts, ids) if i not in done_ids]
    total_done = len(done_ids)
    sys.stderr.write(f"[INFO] Input: {len(texts)} | Pending: {len(pending)} | Done: {total_done}\n")
    sys.stderr.flush()

    if args.limit > 0:
        pending = pending[:args.limit]
        sys.stderr.write(f"[INFO] Limit: {args.limit}\n")
        sys.stderr.flush()

    if not pending:
        sys.stderr.write("[INFO] Nothing to do.\n")
        sys.stderr.flush()
        return

    texts_p, ids_p = zip(*pending)
    results = await annotate_batch(
        texts=list(texts_p),
        task_ids=list(ids_p),
        output_path=args.output,
        escalated_output_path=escalated_output_path,
        concurrency=args.concurrency,
        total_done=total_done,
    )

    label_1 = sum(1 for r in results if r.final_label == "1")
    label_0 = sum(1 for r in results if r.final_label == "0")
    agree = sum(1 for r in results if r.reached_agreement)
    human = sum(1 for r in results if r.needs_human)
    errors = sum(1 for r in results if r.reasoning.startswith("Error:"))
    total = len(results)

    sys.stderr.write("=" * 50 + "\n")
    sys.stderr.write("[INFO] DONE\n")
    sys.stderr.write(f"[INFO] Processed: {total}\n")
    sys.stderr.write(f"[INFO] Label 1:   {label_1} ({label_1/total*100:.1f}%)\n")
    sys.stderr.write(f"[INFO] Label 0:   {label_0} ({label_0/total*100:.1f}%)\n")
    sys.stderr.write(f"[INFO] Agreement: {agree} ({agree/total*100:.1f}%)\n")
    sys.stderr.write(f"[INFO] Human esc: {human} ({human/total*100:.1f}%)\n")
    sys.stderr.write(f"[INFO] Errors:    {errors}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    asyncio.run(main())
