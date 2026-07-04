"""
Pydantic models for DREAM pipeline.
Task-agnostic: works for any binary/multi-class NLP classification.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class DebateTurn(BaseModel):
    """Single turn from one agent."""

    label: str = Field(description="Label decision: 0 or 1")
    evidence: str = Field(description="Quoted evidence from the text")
    argument: str = Field(description="Reasoning/argument in Vietnamese or English")



class DebateRound(BaseModel):
    """One complete round of debate with both agents."""

    round_num: int = Field(description="Round number (1-indexed)")
    agent_a_turn: DebateTurn = Field(description="Turn from Agent_A")
    agent_b_turn: DebateTurn = Field(description="Turn from Agent_B")
    reached_agreement: bool = Field(description="Whether agents agreed this round")
    agreed_label: Optional[str] = Field(
        default=None,
        description="Agreed label if reached_agreement=True",
        pattern=r"^[01]$",
    )


class ModeratorSummary(BaseModel):
    """Output from the moderator when agents disagree."""

    agreements: str = Field(description="Points both agents agree on")
    disagreements: str = Field(description="Key points of disagreement")
    closer_to_label: str = Field(
        description="Which label seems more supported: 0 or 1",
        pattern=r"^[01]$",
    )


class AdjudicationResult(BaseModel):
    """Final decision from the adjudicator."""

    final_label: str = Field(description="Final label: 0 or 1", pattern=r"^[01]$")
    confidence: float = Field(description="Confidence 0.0-1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Reasoning for the decision")


class DreamResult(BaseModel):
    """Final output from the DREAM pipeline."""

    task_id: str = Field(description="Unique task identifier")
    text: str = Field(description="Original input text")
    final_label: str = Field(description="Final label: 0 or 1", pattern=r"^[01]$")

    # Confidence — dynamically calibrated
    confidence: float = Field(description="Calibrated confidence 0.0-1.0", ge=0.0, le=1.0)

    # How the result was reached
    reasoning: str = Field(description="Summary reasoning for the decision")
    reached_agreement: bool = Field(description="Agents agreed without adjudicator")
    agreement_round: Optional[int] = Field(default=None, description="Round of agreement (1 or 2)")

    # Full debate trace
    debate_rounds: List[DebateRound] = Field(
        default_factory=list,
        description="Complete debate history"
    )

    # Moderator (if used)
    moderator_summary: Optional[ModeratorSummary] = Field(default=None)

    # Adjudicator (if agents disagreed)
    adjudication: Optional[AdjudicationResult] = Field(default=None)

    # Escalation
    needs_human: bool = Field(default=False, description="Exported for human review")

    # Raw agent outputs (for debugging)
    agent_a_final_argument: str = Field(default="")
    agent_a_final_evidence: str = Field(default="")
    agent_b_final_argument: str = Field(default="")
    agent_b_final_evidence: str = Field(default="")
