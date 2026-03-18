"""
Debate logic for DREAM pipeline.

Handles:
- Agent debate turns
- Agreement checking
- Adjudication
"""

from typing import List

from loguru import logger

from .models import DebateTurn, DebateRound, AdjudicationResult
from src.api.nim_client import get_nim_client


def _build_system_prompt(config, stance: str) -> str:
    """Build system prompt based on agent stance."""
    if stance == "relevant":
        template = config.dream.agent_complaint_system
    else:
        template = config.dream.agent_non_complaint_system

    # Fill in guidelines
    return template.format(guidelines=config.dream.guidelines)


def _build_user_prompt(
    review: str,
    agent_name: str,
    stance: str,
    history: List[DebateTurn],
    config,
) -> str:
    """Build user prompt for debate turn.

    History format adapted from paper:
    - Paper uses: AgentA: Yes. ... AgentB: No. ...
    - We format: Agent_Complaint: argument + evidence, Agent_NonComplaint: argument + evidence
    """
    target_label = "1" if stance == "relevant" else "0"

    if not history:
        # First round - use template from config
        template = config.dream.agent_user_template
        # Include guidelines in the prompt
        return template.format(
            review=review,
            stance=target_label,
            guidelines=config.dream.guidelines
        )
    else:
        # Subsequent rounds - use roundN template
        template = config.dream.agent_roundN_template

        # Format history following paper style: include both agents' positions
        # Group turns by pairs (relevant, irrelevant)
        history_parts = []
        for turn in history:
            label_desc = "COMPLAINT (Label 1)" if turn.label == "1" else "NON-COMPLAINT (Label 0)"
            history_parts.append(
                f"{turn.agent} ({label_desc}):\n"
                f"  Evidence: {turn.evidence}\n"
                f"  Argument: {turn.argument}"
            )
        history_str = "\n\n".join(history_parts)

        return template.format(
            review=review,
            stance=target_label,
            history=history_str,
            guidelines=config.dream.guidelines
        )


async def run_agent_turn(
    review: str,
    agent_name: str,
    stance: str,
    history: List[DebateTurn],
    config,
) -> DebateTurn:
    """Run a single debate turn for an agent."""

    from .models import DebateTurn as DebateTurnModel

    llm = get_nim_client()

    system_prompt = _build_system_prompt(config, stance)
    user_prompt = _build_user_prompt(review, agent_name, stance, history, config)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug(f"[{agent_name}] Running turn with stance={stance}")

    result = await llm.chat_structured(
        messages=messages,
        response_model=DebateTurnModel,
        temperature=config.dream.llm.temperature,
        max_tokens=config.dream.llm.max_tokens,
    )

    logger.info(f"[{agent_name}] label={result.label}")
    logger.info(f"[{agent_name}] argument: {result.argument[:200]}...")
    logger.info(f"[{agent_name}] evidence: {result.evidence[:100]}...")
    return result


async def run_adjudicator(
    review: str,
    debate_rounds: List[DebateRound],
    config,
) -> AdjudicationResult:
    """Run LLM adjudicator when agents disagree."""

    from .models import AdjudicationResult as AdjudicationModel

    llm = get_nim_client()

    # Build history string
    history_parts = []
    for rnd in debate_rounds:
        history_parts.append(
            f"Round {rnd.round_num}:\n"
            f"  {rnd.relevant_turn.agent}: {rnd.relevant_turn.argument}\n"
            f"  {rnd.irrelevant_turn.agent}: {rnd.irrelevant_turn.argument}"
        )
    history_str = "\n\n".join(history_parts)

    system_prompt = config.dream.adjudicator_system.format(
        guidelines=config.dream.guidelines
    )
    template = config.dream.adjudicator_user_template
    user_prompt = template.format(
        review=review,
        history=history_str,
        guidelines=config.dream.guidelines
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug("[Adjudicator] Running...")

    result = await llm.chat_structured(
        messages=messages,
        response_model=AdjudicationModel,
        temperature=0.3,  # Lower temp for adjudicator
        max_tokens=2048,
    )

    logger.debug(f"[Adjudicator] → label={result.final_label}, confidence={result.confidence}")
    return result


async def run_debate_round(
    review: str,
    round_num: int,
    previous_history: List[DebateTurn],
    config,
) -> DebateRound:
    """Run a complete debate round (both agents)."""

    # Agent Relevant (starts with stance: relevant → label 1)
    relevant_turn = await run_agent_turn(
        review=review,
        agent_name="Agent_A",
        stance="relevant",
        history=previous_history,
        config=config,
    )

    # Agent B (starts with stance: irrelevant → label 0)
    irrelevant_turn = await run_agent_turn(
        review=review,
        agent_name="Agent_B",
        stance="irrelevant",
        history=previous_history,
        config=config,
    )

    # Check agreement
    reached_agreement = relevant_turn.label == irrelevant_turn.label

    logger.info(
        f"[Round {round_num}] Agent_Relevant={relevant_turn.label}, "
        f"Agent_Irrelevant={irrelevant_turn.label}, "
        f"Agreement={reached_agreement}"
    )

    return DebateRound(
        round_num=round_num,
        relevant_turn=relevant_turn,
        irrelevant_turn=irrelevant_turn,
        reached_agreement=reached_agreement,
        agreed_label=relevant_turn.label if reached_agreement else None,
    )
