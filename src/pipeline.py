"""
DREAM Pipeline — Multi-Agent Debate for NLP Classification.
Task-agnostic: configure via config.yaml for any binary classification task.

Flow:
  1. Multi-round debate (Agent_A vs Agent_B, concurrent rounds)
  2. Agreement → auto-label | Disagreement → Moderator → Adjudicator
  3. Low adjudicator confidence → human escalation
"""

import uuid
import logging
from typing import Optional

from src.config import get_config
from src.models import DreamResult, DebateRound, ModeratorSummary, AdjudicationResult
from src.debate import (
    run_debate_round,
    run_moderator,
    run_adjudicator,
    calibrate_confidence,
)

logger = logging.getLogger(__name__)

def _escalation_mode(config) -> str:
    mode = (config.dream.escalation.mode or "fully_automatic").strip().lower()
    if mode not in {"paper_strict_human", "fully_automatic"}:
        logger.warning("Unknown escalation mode '%s', fallback to fully_automatic", mode)
        return "fully_automatic"
    return mode

def _make_result(
    text: str,
    task_id: str,
    final_label: str,
    confidence: float,
    reasoning: str,
    reached_agreement: bool,
    agreement_round: Optional[int],
    rounds: list[DebateRound],
    moderator: Optional[ModeratorSummary],
    adjudication: Optional[AdjudicationResult],
    needs_human: bool,
) -> DreamResult:
    """Build DreamResult from components."""
    last = rounds[-1] if rounds else None

    return DreamResult(
        task_id=task_id,
        text=text,
        final_label=final_label,
        confidence=confidence,
        reasoning=reasoning,
        reached_agreement=reached_agreement,
        agreement_round=agreement_round,
        debate_rounds=rounds,
        used_moderator=moderator is not None,
        moderator_summary=moderator,
        used_adjudicator=adjudication is not None,
        adjudication=adjudication,
        needs_human=needs_human,
        agent_a_final_argument=last.agent_a_turn.argument if last else "",
        agent_a_final_evidence=last.agent_a_turn.evidence if last else "",
        agent_b_final_argument=last.agent_b_turn.argument if last else "",
        agent_b_final_evidence=last.agent_b_turn.evidence if last else "",
    )


async def annotate(
    text: str,
    task_id: Optional[str] = None,
    config=None,
) -> DreamResult:
    """
    Annotate a single text using DREAM pipeline.

    Args:
        text: Input text to classify
        task_id: Optional task ID (auto-generated if not provided)
        config: Config object (loaded from YAML if not provided)

    Returns:
        DreamResult with final_label, confidence, reasoning, and full debate trace
    """
    if config is None:
        config = get_config()

    task_id = task_id or str(uuid.uuid4())
    logger.debug(f"[DREAM] task={task_id} text_len={len(text)}")

    rounds: list[DebateRound] = []
    max_rounds = config.dream.debate.max_rounds
    mode = _escalation_mode(config)

    try:
        # Multi-round debate
        for round_num in range(1, max_rounds + 1):
            debate_round = await run_debate_round(
                text=text,
                round_num=round_num,
                history=rounds,
                config=config,
            )
            rounds.append(debate_round)

            if debate_round.reached_agreement:
                confidence = calibrate_confidence(
                    reached_agreement=True,
                    agreement_round=round_num,
                    rounds=rounds,
                    adjudication=None,
                )
                reasoning = (
                    f"Agents agreed at Round {round_num} on Label {debate_round.agreed_label}. "
                    f"Confidence: {confidence}"
                )
                return _make_result(
                    text=text,
                    task_id=task_id,
                    final_label=debate_round.agreed_label,
                    confidence=confidence,
                    reasoning=reasoning,
                    reached_agreement=True,
                    agreement_round=round_num,
                    rounds=rounds,
                    moderator=None,
                    adjudication=None,
                    needs_human=False,
                )

        # No agreement after max rounds
        logger.info(f"[DREAM] No agreement after {max_rounds} rounds | task={task_id}")

        if mode == "paper_strict_human":
            reasoning = (
                f"No agreement after {max_rounds} rounds. "
                "Marked for human review by paper_strict_human mode."
            )
            return _make_result(
                text=text,
                task_id=task_id,
                final_label=rounds[-1].agent_a_turn.label,
                confidence=0.0,
                reasoning=reasoning,
                reached_agreement=False,
                agreement_round=None,
                rounds=rounds,
                moderator=None,
                adjudication=None,
                needs_human=True,
            )

        moderator: Optional[ModeratorSummary] = None
        moderator_text = "(No moderator)"
        if config.dream.moderator.enabled:
            moderator = await run_moderator(text, rounds, config)
            moderator_text = (
                f"Agreements: {moderator.agreements} | "
                f"Disagreements: {moderator.disagreements} | "
                f"Closer to: {moderator.closer_to_label}"
            )
            logger.info(f"[Moderator] {moderator_text}")

        adjudication = await run_adjudicator(
            text=text,
            rounds=rounds,
            moderator_notes=moderator_text,
            config=config,
        )

        confidence = calibrate_confidence(
            reached_agreement=False,
            agreement_round=None,
            rounds=rounds,
            adjudication=adjudication,
        )

        reasoning = (
            f"No agreement after {max_rounds} rounds. "
            f"Adjudicator → Label {adjudication.final_label} "
            f"(confidence={adjudication.confidence:.2f}). "
            f"Moderator: {moderator_text[:100]}"
        )

        return _make_result(
            text=text,
            task_id=task_id,
            final_label=adjudication.final_label,
            confidence=confidence,
            reasoning=reasoning,
            reached_agreement=False,
            agreement_round=None,
            rounds=rounds,
            moderator=moderator,
            adjudication=adjudication,
            needs_human=False,
        )

    except Exception as e:
        logger.error(f"[DREAM] Error task={task_id}: {e}")
        return _make_result(
            text=text,
            task_id=task_id,
            final_label="0",
            confidence=0.0,
            reasoning=f"Error: {str(e)}",
            reached_agreement=False,
            agreement_round=None,
            rounds=[],
            moderator=None,
            adjudication=None,
            needs_human=True,
        )


# Alias
annotate_with_dream = annotate
