"""
Pydantic models for DREAM pipeline.

Defines structured output schemas for LLM responses.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class DebateTurn(BaseModel):
    """Single turn in the debate."""

    agent: Optional[str] = Field(default="Agent", description="Agent name: Agent_Relevant or Agent_Irrelevant")
    label: str = Field(description="Label: 0 or 1", pattern=r"^[01]$")
    argument: str = Field(description="Argument/reasoning in Vietnamese")
    evidence: str = Field(description="Evidence quoted from the review")


class DebateRound(BaseModel):
    """A complete debate round with both agents."""

    round_num: int = Field(description="Round number (1-indexed)")
    relevant_turn: DebateTurn = Field(description="Turn from Agent_Relevant")
    irrelevant_turn: DebateTurn = Field(description="Turn from Agent_Irrelevant")
    reached_agreement: bool = Field(description="Whether agents agreed in this round")
    agreed_label: Optional[str] = Field(
        default=None,
        description="The label both agents agreed on (if reached_agreement=True)"
    )


class AdjudicationResult(BaseModel):
    """Result from LLM adjudicator when agents disagree."""

    final_label: str = Field(description="Final label: 0 or 1", pattern=r"^[01]$")
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Reasoning for the decision")


class DreamResult(BaseModel):
    """Final result from DREAM pipeline."""

    task_id: str = Field(description="Unique task identifier")
    review: str = Field(description="Original review text")
    old_label: Optional[str] = Field(default=None, description="Original label (for relabeling)")
    final_label: str = Field(description="Final label: 0 or 1", pattern=r"^[01]$")
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Reasoning for the decision")

    # Debate metadata
    reached_agreement: bool = Field(description="Whether agents reached agreement")
    agreement_round: Optional[int] = Field(
        default=None,
        description="Round where agreement was reached (1 or 2)"
    )
    debate_rounds: List[DebateRound] = Field(
        default_factory=list,
        description="Complete debate history"
    )

    # Adjudication
    used_adjudicator: bool = Field(description="Whether LLM adjudicator was used")
    adjudication: Optional[AdjudicationResult] = Field(
        default=None,
        description="Adjudication result if used"
    )
