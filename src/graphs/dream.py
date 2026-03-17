"""
DREAM: Debate-based RElevance Assessment with Multi-agents

Entry point for Vietnamese complaint detection annotation.

Based on: arXiv:2602.06526 - Completing Missing Annotation
"""

import csv
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import get_config
from src.graphs.models import DreamResult, DebateRound
from src.graphs.debate import run_debate_round, run_adjudicator


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
            # Disagreement: use adjudicator
            logger.info("[DREAM] No agreement, using adjudicator")

            if config.dream.escalation.use_llm_adjudicator:
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
                # Human escalation not implemented
                return DreamResult(
                    task_id=task_id,
                    review=review,
                    final_label="0",
                    confidence=0.5,
                    reasoning="Human escalation not implemented",
                    reached_agreement=False,
                    debate_rounds=debate_rounds,
                    used_adjudicator=False,
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
