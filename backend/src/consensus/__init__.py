"""
Consensus mechanism for multi-agent annotation.

Provides weighted voting, confidence aggregation, and threshold management.
"""

from .confidence import ConfidenceAggregator, get_confidence_aggregator
from .thresholds import (
    RoutingAction,
    ThresholdConfig,
    ThresholdManager,
    get_threshold_manager,
)
from .voting import ConsensusEngine, get_consensus_engine

__all__ = [
    # Voting
    "ConsensusEngine",
    "get_consensus_engine",
    # Confidence
    "ConfidenceAggregator",
    "get_confidence_aggregator",
    # Thresholds
    "ThresholdManager",
    "ThresholdConfig",
    "RoutingAction",
    "get_threshold_manager",
]
