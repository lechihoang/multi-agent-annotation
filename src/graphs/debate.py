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
) -> str:
    """Build user prompt for debate turn."""
    target_label = "1" if stance == "relevant" else "0"
    base_prompt = f"""Review cần phân loại:

{review}

**Stance hiện tại**: Label {target_label}
Bạn có thể giữ stance này hoặc THAY ĐỔI stance nếu bị thuyết phục.

Evidence: <trích đoạn từ review>
Argument: <lập luận của bạn>
Label: <0 hoặc 1>"""

    if not history:
        # First round - just the review
        return base_prompt
    else:
        # Subsequent rounds - include history
        history_str = "\n\n".join([
            f"**{turn.agent}** (Label {turn.label}):\n{turn.argument}\nEvidence: {turn.evidence}"
            for turn in history
        ])
        return f"""{base_prompt}

**Lịch sử tranh luận trước đó**:
{history_str}

**Nhiệm vụ thêm**:
- Đọc argument của đối phương
- Phản bác hoặc thừa nhận điểm hợp lý
- Có thể giữ stance hoặc THAY ĐỔI stance nếu bị thuyết phục
- Tìm thêm bằng chứng nếu cần"""


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
    user_prompt = _build_user_prompt(review, agent_name, stance, history)

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
    user_prompt = f"""Review cần phân loại:

{review}

**Lịch sử tranh luận**:
{history_str}

Hai Agent không đồng ý. Hãy đưa ra quyết định cuối cùng dựa trên lập luận của cả hai bên."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug("[Adjudicator] Running...")

    result = await llm.chat_structured(
        messages=messages,
        response_model=AdjudicationModel,
        temperature=0.3,  # Lower temp for adjudicator
        max_tokens=1024,
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
        agent_name="Agent_Relevant",
        stance="relevant",
        history=previous_history,
        config=config,
    )

    # Agent Irrelevant (starts with stance: irrelevant → label 0)
    irrelevant_turn = await run_agent_turn(
        review=review,
        agent_name="Agent_Irrelevant",
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
