"""
Intent Classification Agent.

Classifies user intents using DeepSeek-R1 or Llama models.
Tier 2 agent with weight 0.35.
"""

from typing import Any

import structlog

from ..config.settings import settings
from ..models import Annotation, ChatResponse, IntentAnnotation
from .base import BaseAnnotationAgent

logger = structlog.get_logger()


class IntentAgent(BaseAnnotationAgent):
    """
    Intent Classification Agent.

    Uses DeepSeek-R1 (Groq) for complex reasoning and intent classification.
    Falls back to Llama models on HuggingFace if needed.

    Weight: 0.35 (highest priority for downstream routing)
    """

    name = "intent_agent"
    weight = settings.intent_agent_weight
    tier = 2

    # Default intent categories
    DEFAULT_INTENTS = [
        "inquiry",
        "request",
        "complaint",
        "feedback",
        "greeting",
        "confirmation",
        "cancellation",
        "support",
        "purchase",
        "other",
    ]

    def __init__(
        self,
        primary_provider: str = "groq",
        fallback_provider: str = "huggingface",
        primary_model: str = "deepseek-r1-distill-llama-70b",
        fallback_model: str = "meta-llama/Llama-3.2-8B-Instruct",
        intent_categories: list[str] | None = None,
    ):
        """
        Initialize Intent Agent.

        Args:
            primary_provider: Primary API provider
            fallback_provider: Fallback provider
            primary_model: Model for primary provider
            fallback_model: Model for fallback
            intent_categories: Custom intent categories
        """
        super().__init__(
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            primary_model=primary_model,
            fallback_model=fallback_model,
        )
        self.intent_categories = intent_categories or self.DEFAULT_INTENTS

    def get_system_prompt(self) -> str:
        """Get system prompt for intent classification."""
        prompts = self._prompts.get("intent_classification", {})

        if "system" in prompts:
            return prompts["system"]

        # Default prompt
        categories_str = ", ".join(self.intent_categories)
        return f"""You are an expert intent classification agent. Your task is to analyze user text
and classify the primary intent with high accuracy.

Available intent categories: {categories_str}

Guidelines:
- Identify the main purpose or goal behind the text
- Consider context and implicit meanings
- Provide a confidence score (0.0 to 1.0)
- List alternative intents if applicable

Output format (JSON only):
{{
  "intent": "primary_intent_name",
  "confidence": 0.95,
  "reasoning": "Brief explanation",
  "alternatives": [
    {{"intent": "alternative_intent", "confidence": 0.3}}
  ]
}}

IMPORTANT: Your response must be valid JSON only. Do not include any text before or after the JSON."""

    def get_user_prompt(self, text: str, **kwargs) -> str:
        """Get user prompt with text to classify."""
        prompts = self._prompts.get("intent_classification", {})

        if "user" in prompts:
            return prompts["user"].format(text=text)

        return f"""Classify the intent of the following text:

Text: {text}

Provide your classification in valid JSON format."""

    def parse_response(self, response: ChatResponse) -> dict[str, Any]:
        """Parse intent classification response."""
        parsed = self._parse_json_response(response.content)

        # Validate and normalize
        intent = parsed.get("intent", "other")
        confidence = parsed.get("confidence", 0.5)

        # Normalize intent to lowercase
        if isinstance(intent, str):
            intent = intent.lower().strip()

        # Validate intent is in known categories
        if intent not in self.intent_categories:
            logger.warning(
                "unknown_intent_classified",
                intent=intent,
                known_categories=self.intent_categories,
            )

        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", ""),
            "alternatives": parsed.get("alternatives", []),
        }

    async def annotate(self, text: str, **kwargs) -> Annotation:
        """
        Classify intent of the given text.

        Args:
            text: Text to classify
            **kwargs: Additional parameters

        Returns:
            Annotation with intent classification result
        """
        messages = self._build_messages(text, **kwargs)

        response = await self._execute_with_fallback(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            json_mode=True,
        )

        result = self.parse_response(response)

        logger.info(
            "intent_classified",
            intent=result.get("intent"),
            confidence=result.get("confidence"),
            latency_ms=round(response.latency_ms, 2),
        )

        return self._create_annotation(result, response.latency_ms)

    async def classify_intent(self, text: str) -> IntentAnnotation:
        """
        Convenience method that returns IntentAnnotation directly.

        Args:
            text: Text to classify

        Returns:
            IntentAnnotation with classification result
        """
        annotation = await self.annotate(text)
        result = annotation.result

        return IntentAnnotation(
            intent=result.get("intent", "other"),
            confidence=annotation.confidence,
            reasoning=result.get("reasoning"),
            alternatives=result.get("alternatives", []),
        )


# Factory function
def create_intent_agent(**kwargs) -> IntentAgent:
    """Create an IntentAgent with optional configuration."""
    return IntentAgent(**kwargs)
