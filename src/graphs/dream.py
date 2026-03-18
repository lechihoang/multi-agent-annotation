"""
DREAM: Debate-based RElevance Assessment with Multi-agents

Entry point for Vietnamese complaint detection annotation.

Based on: arXiv:2602.06526 - Completing Missing Annotation
"""

import csv
import json
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import get_config
from src.graphs.models import DreamResult, DebateRound
from src.graphs.debate import run_debate_round, run_adjudicator


def _export_for_human(
    task_id: str,
    review: str,
    debate_rounds,
    rel_arg: str,
    rel_evidence: str,
    irr_arg: str,
    irr_evidence: str,
    adjudicator_reasoning: str = "",
    export_file: str = "data/escalated_for_human.csv",
):
    """
    Export unresolved case to file for human review.

    Per DREAM paper: Human annotator receives:
    - The review
    - Debate history (both agents' reasoning and evidence)
    - Adjudicator reasoning (if available)
    - This helps understand source of disagreement for better judgment
    """
    # Build debate history string
    history_parts = []
    for rnd in debate_rounds:
        history_parts.append(
            f"Vòng {rnd.round_num}:\n"
            f"  Agent_A (Nhãn {rnd.relevant_turn.label}):\n"
            f"    Bằng chứng: {rnd.relevant_turn.evidence}\n"
            f"    Lập luận: {rnd.relevant_turn.argument}\n\n"
            f"  Agent_B (Nhãn {rnd.irrelevant_turn.label}):\n"
            f"    Bằng chứng: {rnd.irrelevant_turn.evidence}\n"
            f"    Lập luận: {rnd.irrelevant_turn.argument}"
        )
    history_str = "\n\n".join(history_parts)

    row = {
        "task_id": task_id,
        "review": review,
        "debate_history": history_str,
        "relevant_argument": rel_arg,
        "relevant_evidence": rel_evidence,
        "irrelevant_argument": irr_arg,
        "irrelevant_evidence": irr_evidence,
        "adjudicator_reasoning": adjudicator_reasoning,
        "human_label": "",  # To be filled by human
    }

    file_path = Path(export_file)
    file_exists = file_path.exists()

    with open(file_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "review",
                "debate_history",
                "relevant_argument",
                "relevant_evidence",
                "irrelevant_argument",
                "irrelevant_evidence",
                "human_label",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"[DREAM] Exported to {export_file} for human review")


async def annotate_with_dream(
    review: str,
    task_id: Optional[str] = None,
    config=None,
) -> DreamResult:
    """
    Annotate a Vietnamese review using DREAM pipeline.

    Flow:
      1. Initialize two agents with opposing stances
      2. Multi-round debate with reciprocal critique
      3. If agreement reached → use that label
      4. If disagreement → use LLM adjudicator
    """
    if config is None:
        config = get_config()

    task_id = task_id or str(uuid.uuid4())

    logger.info(f"[DREAM] Annotating task {task_id}")

    debate_rounds: list[DebateRound] = []
    max_rounds = config.dream.debate.max_rounds
    reached_agreement = False
    agreement_round = None

    try:
        # Multi-round debate
        for round_num in range(1, max_rounds + 1):
            # Get previous history for this round
            previous_history = []
            for rnd in debate_rounds:
                previous_history.append(rnd.relevant_turn)
                previous_history.append(rnd.irrelevant_turn)

            # Run debate round
            debate_round = await run_debate_round(
                review=review,
                round_num=round_num,
                previous_history=previous_history,
                config=config,
            )
            debate_rounds.append(debate_round)

            # Check if agents agreed
            if debate_round.reached_agreement:
                reached_agreement = True
                agreement_round = round_num
                logger.info(f"[DREAM] Agreement at round {agreement_round}")
                break

        # Final decision
        if reached_agreement:
            # Use agreed label
            final_label = debate_rounds[-1].agreed_label
            confidence = 0.9
            reasoning = f"Agents agreed at round {agreement_round}"

            # Get final arguments from last round
            last_round = debate_rounds[-1] if debate_rounds else None
            rel_arg = last_round.relevant_turn.argument if last_round else ""
            rel_evi = last_round.relevant_turn.evidence if last_round else ""
            irr_arg = last_round.irrelevant_turn.argument if last_round else ""
            irr_evi = last_round.irrelevant_turn.evidence if last_round else ""

            return DreamResult(
                task_id=task_id,
                review=review,
                final_label=final_label,
                confidence=confidence,
                reasoning=reasoning,
                reached_agreement=True,
                agreement_round=agreement_round,
                debate_rounds=debate_rounds,
                used_adjudicator=False,
                relevant_argument=rel_arg,
                relevant_evidence=rel_evi,
                irrelevant_argument=irr_arg,
                irrelevant_evidence=irr_evi,
            )
        else:
            # Disagreement: handle based on config
            logger.info("[DREAM] No agreement reached")

            if config.dream.escalation.use_llm_adjudicator:
                # Use LLM as adjudicator first
                adjudication = await run_adjudicator(
                    review=review,
                    debate_rounds=debate_rounds,
                    config=config,
                )

                # Get final arguments from last round
                last_round = debate_rounds[-1] if debate_rounds else None
                rel_arg = last_round.relevant_turn.argument if last_round else ""
                rel_evi = last_round.relevant_turn.evidence if last_round else ""
                irr_arg = last_round.irrelevant_turn.argument if last_round else ""
                irr_evi = last_round.irrelevant_turn.evidence if last_round else ""

                # Check if adjudicator confidence is below threshold → escalate to human
                threshold = config.dream.escalation.confidence_threshold
                if adjudication.confidence < threshold:
                    logger.warning(f"[DREAM] Adjudicator confidence {adjudication.confidence} < {threshold}, escalating to human")

                    # Export to file for human review
                    _export_for_human(
                        task_id=task_id,
                        review=review,
                        debate_rounds=debate_rounds,
                        rel_arg=rel_arg,
                        rel_evidence=rel_evi,
                        irr_arg=irr_arg,
                        irr_evidence=irr_evi,
                        adjudicator_reasoning=adjudication.reasoning,
                        export_file=config.dream.escalation.export_file,
                    )

                    return DreamResult(
                        task_id=task_id,
                        review=review,
                        final_label="",  # Empty - needs human review
                        confidence=adjudication.confidence,
                        reasoning=f"ESCALATED_TO_HUMAN: Adjudicator confidence {adjudication.confidence} < {threshold}",
                        reached_agreement=False,
                        debate_rounds=debate_rounds,
                        used_adjudicator=True,
                        adjudication=adjudication,
                        relevant_argument=rel_arg,
                        relevant_evidence=rel_evi,
                        irrelevant_argument=irr_arg,
                        irrelevant_evidence=irr_evi,
                        adjudicator_reasoning=adjudication.reasoning,
                    )

                # Adjudicator confident enough → use its decision
                return DreamResult(
                    task_id=task_id,
                    review=review,
                    final_label=adjudication.final_label,
                    confidence=adjudication.confidence,
                    reasoning=adjudication.reasoning,
                    reached_agreement=False,
                    debate_rounds=debate_rounds,
                    used_adjudicator=True,
                    adjudication=adjudication,
                    relevant_argument=rel_arg,
                    relevant_evidence=rel_evi,
                    irrelevant_argument=irr_arg,
                    irrelevant_evidence=irr_evi,
                    adjudicator_reasoning=adjudication.reasoning,
                )
            else:
                # Human escalation: Export to file for human review
                # Per DREAM paper: human receives review + debate history for better judgment
                logger.warning(f"[DREAM] Escalating to human: {task_id}")

                # Get final arguments from last round
                last_round = debate_rounds[-1] if debate_rounds else None
                rel_arg = last_round.relevant_turn.argument if last_round else ""
                rel_evi = last_round.relevant_turn.evidence if last_round else ""
                irr_arg = last_round.irrelevant_turn.argument if last_round else ""
                irr_evi = last_round.irrelevant_turn.evidence if last_round else ""

                # Export to file for human review
                export_file = config.dream.escalation.export_file
                _export_for_human(
                    task_id=task_id,
                    review=review,
                    debate_rounds=debate_rounds,
                    rel_arg=rel_arg,
                    rel_evidence=rel_evi,
                    irr_arg=irr_arg,
                    irr_evidence=irr_evi,
                    adjudicator_reasoning="",
                    export_file=export_file,
                )

                return DreamResult(
                    task_id=task_id,
                    review=review,
                    final_label="",  # Empty - needs human review
                    confidence=0.0,
                    reasoning="ESCALATED_TO_HUMAN - See debate history in escalated file",
                    reached_agreement=False,
                    debate_rounds=debate_rounds,
                    used_adjudicator=False,
                    relevant_argument=rel_arg,
                    relevant_evidence=rel_evi,
                    irrelevant_argument=irr_arg,
                    irrelevant_evidence=irr_evi,
                )

    except Exception as e:
        logger.error(f"[DREAM] Error: {e}")
        return DreamResult(
            task_id=task_id,
            review=review,
            final_label="0",
            confidence=0.0,
            reasoning=f"Error: {str(e)}",
            reached_agreement=False,
            debate_rounds=debate_rounds,
            used_adjudicator=False,
        )


# =============================================================================
# Batch Processing
# =============================================================================


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import asyncio

    # Test
    test_review = "Giao hàng chậm quá, rất không hài lòng!"

    result = asyncio.run(annotate_with_dream(test_review))
    print(result.model_dump_json(indent=2, ensure_ascii=False))
