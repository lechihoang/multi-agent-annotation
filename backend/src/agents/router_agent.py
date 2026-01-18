"""
Router Agent - Tier 1.

Analyzes task complexity and routes to appropriate API/model.
Responsible for initial task classification and batching.
"""

from typing import Any

import structlog

from ..api import get_api_manager
from ..models import ChatResponse, ComplexityLevel, Task
from .base import BaseAnnotationAgent

logger = structlog.get_logger()


class RouterAgent(BaseAnnotationAgent):
    """
    Router Agent (Tier 1).

    Analyzes incoming tasks and determines:
    - Task complexity (HIGH/MEDIUM/LOW)
    - Optimal API provider routing
    - Batching strategy for similar tasks

    This agent uses lightweight models for fast classification.
    """

    name = "router_agent"
    weight = 1.0  # Router doesn't participate in consensus
    tier = 1

    # Complexity indicators
    HIGH_COMPLEXITY_INDICATORS = [
        "multi-step reasoning",
        "ambiguous context",
        "domain-specific terminology",
        "requires inference",
        "complex grammar",
        "nested clauses",
    ]

    MEDIUM_COMPLEXITY_INDICATORS = [
        "clear context",
        "common patterns",
        "moderate length",
        "single topic",
    ]

    LOW_COMPLEXITY_INDICATORS = [
        "short text",
        "FAQ-like",
        "simple classification",
        "greeting",
        "yes/no question",
    ]

    def __init__(
        self,
        primary_provider: str = "ollama",
        fallback_provider: str = "huggingface",
        primary_model: str = "llama3.2",
        fallback_model: str = "meta-llama/Llama-3.2-8B-Instruct",
        length_thresholds: dict[str, int] | None = None,
    ):
        """
        Initialize Router Agent.

        Args:
            primary_provider: Primary provider (local for speed)
            fallback_provider: Fallback provider
            primary_model: Model for primary provider
            fallback_model: Model for fallback
            length_thresholds: Custom text length thresholds
        """
        super().__init__(
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            primary_model=primary_model,
            fallback_model=fallback_model,
        )

        self.length_thresholds = length_thresholds or {
            "low": 100,  # < 100 chars = low complexity
            "medium": 500,  # 100-500 chars = medium
            # > 500 chars = high complexity
        }

        self._api_manager = get_api_manager()

    def get_system_prompt(self) -> str:
        """Get system prompt for complexity classification."""
        prompts = self._prompts.get("complexity_classification", {})

        if "system" in prompts:
            return prompts["system"]

        return """You are a task complexity classifier. Analyze the given text and determine
its complexity level for annotation purposes.

Complexity Levels:
- HIGH: Requires deep reasoning, ambiguous context, domain expertise, long text
- MEDIUM: Standard annotation task, clear context, moderate length
- LOW: Simple pattern matching, short text, FAQ-like queries

Factors to consider:
- Text length and structure
- Presence of technical/domain terms
- Ambiguity level
- Required reasoning depth

Output format (JSON only):
{
  "complexity": "HIGH",
  "confidence": 0.90,
  "factors": ["factor1", "factor2"],
  "reasoning": "Brief explanation"
}

IMPORTANT: Your response must be valid JSON only."""

    def get_user_prompt(self, text: str, **kwargs) -> str:
        """Get user prompt for complexity classification."""
        prompts = self._prompts.get("complexity_classification", {})

        if "user" in prompts:
            return prompts["user"].format(text=text)

        return f"""Classify the complexity of the following text for annotation:

Text: {text}

Provide your classification in valid JSON format."""

    def parse_response(self, response: ChatResponse) -> dict[str, Any]:
        """Parse complexity classification response."""
        parsed = self._parse_json_response(response.content)

        complexity_str = parsed.get("complexity", "MEDIUM").upper()

        # Map to enum
        complexity_map = {
            "HIGH": ComplexityLevel.HIGH,
            "MEDIUM": ComplexityLevel.MEDIUM,
            "LOW": ComplexityLevel.LOW,
        }
        complexity = complexity_map.get(complexity_str, ComplexityLevel.MEDIUM)

        return {
            "complexity": complexity.value,
            "confidence": parsed.get("confidence", 0.5),
            "factors": parsed.get("factors", []),
            "reasoning": parsed.get("reasoning", ""),
        }

    async def annotate(self, text: str, **kwargs) -> Any:
        """
        Classify complexity of text.

        Note: Router doesn't return standard Annotation,
        but rather routing decision.
        """
        return await self.classify_complexity(text)

    def classify_complexity_heuristic(self, text: str) -> ComplexityLevel:
        """
        Quick heuristic-based complexity classification.

        Used when LLM call is not needed (for simple cases).

        Args:
            text: Text to classify

        Returns:
            ComplexityLevel
        """
        text_length = len(text)

        # Length-based heuristic
        if text_length < self.length_thresholds["low"]:
            return ComplexityLevel.LOW
        elif text_length > self.length_thresholds["medium"]:
            return ComplexityLevel.HIGH

        # Check for complexity indicators
        text_lower = text.lower()

        # Check for domain-specific terms (simplified)
        technical_terms = [
            "api",
            "database",
            "algorithm",
            "configuration",
            "integration",
            "authentication",
            "authorization",
            "deployment",
        ]
        has_technical = any(term in text_lower for term in technical_terms)

        # Check for question words indicating complexity
        complex_patterns = [
            "how does",
            "why does",
            "explain",
            "compare",
            "difference between",
            "analyze",
        ]
        has_complex_pattern = any(p in text_lower for p in complex_patterns)

        if has_technical or has_complex_pattern:
            return ComplexityLevel.MEDIUM if text_length < 300 else ComplexityLevel.HIGH

        return ComplexityLevel.MEDIUM

    async def classify_complexity(
        self,
        text: str,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        """
        Classify task complexity.

        Args:
            text: Text to classify
            use_llm: Whether to use LLM for classification

        Returns:
            Dict with complexity, confidence, and routing info
        """
        # Start with heuristic
        heuristic_complexity = self.classify_complexity_heuristic(text)

        if not use_llm:
            # Use heuristic only
            routing = self._get_routing(heuristic_complexity)

            logger.info(
                "complexity_classified_heuristic",
                complexity=heuristic_complexity.value,
                text_length=len(text),
            )

            return {
                "complexity": heuristic_complexity,
                "confidence": 0.7,  # Lower confidence for heuristic
                "routing": routing,
                "method": "heuristic",
            }

        # Use LLM for more accurate classification
        try:
            messages = self._build_messages(text)
            response = await self._execute_with_fallback(
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                json_mode=True,
            )

            result = self.parse_response(response)
            complexity = ComplexityLevel(result["complexity"])
            routing = self._get_routing(complexity)

            logger.info(
                "complexity_classified_llm",
                complexity=complexity.value,
                confidence=result.get("confidence"),
                latency_ms=round(response.latency_ms, 2),
            )

            return {
                "complexity": complexity,
                "confidence": result.get("confidence", 0.8),
                "routing": routing,
                "factors": result.get("factors", []),
                "reasoning": result.get("reasoning", ""),
                "method": "llm",
            }

        except Exception as e:
            logger.warning(
                "llm_classification_failed_using_heuristic",
                error=str(e),
            )

            routing = self._get_routing(heuristic_complexity)
            return {
                "complexity": heuristic_complexity,
                "confidence": 0.6,
                "routing": routing,
                "method": "heuristic_fallback",
            }

    def _get_routing(self, complexity: ComplexityLevel) -> dict[str, Any]:
        """Get routing configuration for complexity level."""
        routing_config = {
            ComplexityLevel.HIGH: {
                "primary_provider": "groq",
                "primary_model": "deepseek-r1-distill-llama-70b",
                "fallback_provider": "huggingface",
                "agents": ["intent_agent", "entity_agent", "faq_agent"],
            },
            ComplexityLevel.MEDIUM: {
                "primary_provider": "groq",
                "primary_model": "llama-3.3-70b-versatile",
                "fallback_provider": "huggingface",
                "agents": ["intent_agent", "entity_agent", "faq_agent"],
            },
            ComplexityLevel.LOW: {
                "primary_provider": "huggingface",
                "primary_model": "meta-llama/Llama-3.2-8B-Instruct",
                "fallback_provider": "ollama",
                "agents": ["intent_agent", "faq_agent"],  # Skip entity for simple
            },
        }
        return routing_config.get(complexity, routing_config[ComplexityLevel.MEDIUM])

    def batch_tasks(
        self,
        tasks: list[Task],
    ) -> dict[ComplexityLevel, list[Task]]:
        """
        Group tasks by complexity for batch processing.

        Args:
            tasks: List of tasks to batch

        Returns:
            Dict of complexity level to tasks
        """
        batches: dict[ComplexityLevel, list[Task]] = {
            ComplexityLevel.HIGH: [],
            ComplexityLevel.MEDIUM: [],
            ComplexityLevel.LOW: [],
        }

        for task in tasks:
            complexity = self.classify_complexity_heuristic(task.text)
            batches[complexity].append(task)

        logger.info(
            "tasks_batched",
            high=len(batches[ComplexityLevel.HIGH]),
            medium=len(batches[ComplexityLevel.MEDIUM]),
            low=len(batches[ComplexityLevel.LOW]),
        )

        return batches


# Factory function
def create_router_agent(**kwargs) -> RouterAgent:
    """Create a RouterAgent with optional configuration."""
    return RouterAgent(**kwargs)
