"""
Debate logic for DREAM pipeline.
Handles: agent turns, rounds, moderator, adjudicator.
"""

import asyncio
import logging
from typing import List

from .models import (
    DebateTurn,
    DebateRound,
    ModeratorSummary,
    AdjudicationResult,
)
from src.api.nim_client import get_nim_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_labels_text(config) -> str:
    """Format label definitions as readable text."""
    parts = []
    for key, desc in config.task.labels.items():
        parts.append(f"  Label {key}: {desc}")
    return "\n".join(parts)


def _build_agent_system(config, agent: str) -> str:
    """Build system prompt for Agent_A or Agent_B."""
    if agent == "Agent_A":
        template = config.dream.agent_a_system
    else:
        template = config.dream.agent_b_system
    return template.format(guidelines=config.dream.guidelines)


def _format_history(rounds: List[DebateRound]) -> str:
    """Format debate history for prompt injection."""
    if not rounds:
        return "(No previous rounds)"
    parts = []
    for rnd in rounds:
        parts.append(
            f"Round {rnd.round_num}:\n"
            f"  Agent_A ({rnd.agent_a_turn.label}): Evidence: {rnd.agent_a_turn.evidence} | "
            f"Argument: {rnd.agent_a_turn.argument}\n"
            f"  Agent_B ({rnd.agent_b_turn.label}): Evidence: {rnd.agent_b_turn.evidence} | "
            f"Argument: {rnd.agent_b_turn.argument}"
        )
    return "\n\n".join(parts)


def _build_agent_user_prompt(
    config,
    text: str,
    stance_label: str,
    history: List[DebateRound],
) -> str:
    """Build user prompt for an agent turn."""
    labels_text = _build_labels_text(config)
    history_text = _format_history(history)

    if not history:
        template = config.dream.agent_user_template
        return template.format(
            text=text,
            labels=labels_text,
            stance=stance_label,
        )
    else:
        template = config.dream.agent_roundN_template
        return template.format(
            text=text,
            labels=labels_text,
            stance=stance_label,
            history=history_text,
        )


# ---------------------------------------------------------------------------
# Agent turns
# ---------------------------------------------------------------------------

async def run_agent_turn(
    text: str,
    agent: str,
    stance_label: str,
    history: List[DebateRound],
    config,
) -> DebateTurn:
    """Run a single turn for one agent."""
    llm = get_nim_client()

    system_prompt = _build_agent_system(config, agent)
    user_prompt = _build_agent_user_prompt(config, text, stance_label, history)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug(f"[{agent}] stance={stance_label}, history_rounds={len(history)}")

    max_tokens = config.dream.llm.max_tokens
    if history:
        max_tokens = max(256, int(config.dream.llm.max_tokens * 0.75))

    result = await llm.chat_structured(
        messages=messages,
        response_model=DebateTurn,
        temperature=config.dream.llm.temperature,
        max_tokens=max_tokens,
        max_retries=5,
    )

    logger.info(
        f"[{agent}] → label={result.label} | "
        f"evidence: {result.evidence[:80]}... | "
        f"arg: {result.argument[:80]}..."
    )
    return result


# ---------------------------------------------------------------------------
# Debate round
# ---------------------------------------------------------------------------

async def run_debate_round(
    text: str,
    round_num: int,
    history: List[DebateRound],
    config,
) -> DebateRound:
    """
    Run one complete debate round (both agents).
    Round 1: both agents start blind (no history).
    Round N: both agents see all previous rounds.
    """
    # Run both agents in parallel — they both see the same history
    agent_a_task = run_agent_turn(
        text=text,
        agent="Agent_A",
        stance_label="Label 1",
        history=history,
        config=config,
    )
    agent_b_task = run_agent_turn(
        text=text,
        agent="Agent_B",
        stance_label="Label 0",
        history=history,
        config=config,
    )

    agent_a_turn, agent_b_turn = await asyncio.gather(agent_a_task, agent_b_task)

    reached_agreement = agent_a_turn.label == agent_b_turn.label

    logger.info(
        f"[Round {round_num}] Agent_A={agent_a_turn.label} | "
        f"Agent_B={agent_b_turn.label} | agreed={reached_agreement}"
    )

    return DebateRound(
        round_num=round_num,
        agent_a_turn=agent_a_turn,
        agent_b_turn=agent_b_turn,
        reached_agreement=reached_agreement,
        agreed_label=agent_a_turn.label if reached_agreement else None,
    )


# ---------------------------------------------------------------------------
# Moderator
# ---------------------------------------------------------------------------

async def run_moderator(text: str, rounds: List[DebateRound], config) -> ModeratorSummary:
    """Extract key points of agreement/disagreement to help the adjudicator."""
    llm = get_nim_client()

    history_text = _format_history(rounds)

    system_prompt = config.dream.moderator.system
    user_prompt = config.dream.moderator.user_template.format(
        text=text,
        history=history_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug("[Moderator] Running...")
    result = await llm.chat_structured(
        messages=messages,
        response_model=ModeratorSummary,
        temperature=0.0,
        max_tokens=512,
        max_retries=5,
    )

    logger.info(f"[Moderator] → closer_to_label={result.closer_to_label}")
    return result


# ---------------------------------------------------------------------------
# Adjudicator
# ---------------------------------------------------------------------------

async def run_adjudicator(
    text: str,
    rounds: List[DebateRound],
    moderator_notes: str,
    config,
) -> AdjudicationResult:
    """
    Final decision when agents disagree after max rounds.
    Per paper: temperature = 0.0 (Appendix A.4).
    Adjudicator sees: text + full debate history + evidence + moderator notes.
    """
    llm = get_nim_client()

    history_text = _format_history(rounds)
    labels_text = _build_labels_text(config)

    system_prompt = config.dream.adjudicator.system.format(
        guidelines=config.dream.guidelines
    )
    user_prompt = config.dream.adjudicator.user_template.format(
        text=text,
        labels=labels_text,
        history=history_text,
        moderator_notes=moderator_notes,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug("[Adjudicator] Running...")
    result = await llm.chat_structured(
        messages=messages,
        response_model=AdjudicationResult,
        temperature=config.dream.adjudicator.temperature,
        max_tokens=config.dream.adjudicator.max_tokens,
        max_retries=5,
    )

    logger.info(
        f"[Adjudicator] → label={result.final_label} | "
        f"confidence={result.confidence} | "
        f"reasoning: {result.reasoning[:100]}..."
    )
    return result


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

def calibrate_confidence(
    reached_agreement: bool,
    agreement_round: int | None,
    adjudication: AdjudicationResult | None,
) -> float:
    """
    Confidence score derived purely from debate structure per the DREAM paper.

    Key principle (paper §3.1):
      "our approach leverages inter-agent agreement as a direct indicator of
       reliability … avoiding calibration training or threshold tuning."

    Signal source is observable debate *behavior*, not self-assessed content:
      - Round 1 agreement  → 0.95  (immediate consensus, strongest signal)
      - Round 2 agreement  → 0.85  (survived one round of challenge)
      - Round R≥3          → decreasing by 0.05 per extra round
      - Adjudicator path   → adjudicator's own stated confidence (already a
                             structured LLM output, not heuristic)
      - No agreement       → 0.0   (genuinely uncertain → escalate to human)
    """
    if reached_agreement and agreement_round is not None:
        # Agreement speed is the only structural signal the paper defines.
        # Paper default max rounds = 2; handle R≥3 gracefully.
        base = 0.95 - (agreement_round - 1) * 0.10   # R1→0.95, R2→0.85, R3→0.75 …
        return round(max(base, 0.60), 3)

    if adjudication is not None:
        # Adjudicator explicitly outputs its confidence as part of its
        # structured response — use it directly, no secondary heuristic.
        return round(float(adjudication.confidence), 3)

    # No agreement, no adjudicator → maximum uncertainty
    return 0.0
