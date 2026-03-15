"""DREAM pipeline for Vietnamese complaint detection."""

from .models import DreamResult, DebateTurn, DebateRound, AdjudicationResult
from .dream import annotate_with_dream

__all__ = [
    "DreamResult",
    "DebateTurn",
    "DebateRound",
    "AdjudicationResult",
    "annotate_with_dream",
]
