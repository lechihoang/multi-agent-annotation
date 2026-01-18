"""
Annotation agents for the multi-agent system.

Provides specialized agents for intent classification, entity extraction,
FAQ matching, and task routing.
"""

from .base import BaseAnnotationAgent
from .entity_agent import EntityAgent, create_entity_agent
from .faq_agent import FAQAgent, create_faq_agent
from .intent_agent import IntentAgent, create_intent_agent
from .router_agent import RouterAgent, create_router_agent

__all__ = [
    # Base
    "BaseAnnotationAgent",
    # Tier 2 Agents
    "IntentAgent",
    "create_intent_agent",
    "EntityAgent",
    "create_entity_agent",
    "FAQAgent",
    "create_faq_agent",
    # Tier 1 Agent
    "RouterAgent",
    "create_router_agent",
]
