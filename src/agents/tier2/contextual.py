"""Contextual Agent - MAFA Tier 2 Agent 2.

Agent B: Full-Context analysis using ARQ-style structured prompting.
Incorporates secondary information (Title, metadata) for deeper understanding.

MAFA Section 4.2.1: ARQ prompts with domain-specific queries for contextual analysis.
"""

from typing import Dict, Any, List, Optional
import json

from ...config import get_config, get_llm_client
from .arq_prompts import ARQPromptBuilder


class ContextualAnnotation:
    """Result from contextual analysis agent."""

    def __init__(
        self, label: str, confidence: float, reasoning: str, context_used: str
    ):
        self.label = label
        self.confidence = confidence
        self.reasoning = reasoning
        self.context_used = context_used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "context_used": self.context_used,
        }


class ContextualAgent:
    """MAFA Agent 2: Full-Context Ranker.

    - Uses structured prompting with secondary context (Title)
    - Enables deeper semantic understanding
    - Better for queries requiring contextual interpretation
    - Weight: 0.25 in MAFA ensemble
    """

    def __init__(self):
        self.config = get_config()
        self.weight = self.config.agents.contextual
        self._llm_client = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM client based on provider configuration."""
        self._llm_client = get_llm_client(self.config)
        if self._llm_client is None:
            print("Warning: No LLM client available (neither Groq nor NIM)")

    def _build_prompt(
        self,
        text: str,
        title: str,
        labels: List[str],
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build ARQ-style structured prompt with contextual analysis.

        MAFA Section 4.2.1: ARQ prompts with title/context analysis.
        Output MUST be valid JSON (NO FALLBACK).
        """
        arq_prompt = ARQPromptBuilder.build_toxicity_arq(
            text=text,
            examples=few_shot_examples or [],
            context=title,
            agent_type="contextual",
        )
        return ARQPromptBuilder.to_prompt(arq_prompt)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse ARQ-style response from LLM.

        NO FALLBACK - Response MUST be valid JSON.
        """
        result = ARQPromptBuilder.parse_response(content)

        return {
            "topic": result.get("final_label", "unknown"),
            "confidence": result.get("confidence_score", 0.5),
            "confidence_level": result.get("confidence", "MEDIUM"),
            "reasoning": result.get("reasoning", ""),
            "title_influence": result.get("reasoning", ""),
        }

    async def annotate(
        self,
        text: str,
        title: str = "",
        labels: List[str] | None = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> ContextualAnnotation:
        """Annotate text using contextual analysis with few-shot examples.

        Args:
            text: Text to classify
            title: Title/context for the text
            labels: Classification labels
            few_shot_examples: Optional few-shot examples from FewShotSelector
        """
        # If labels not provided or empty, use dynamic inference
        if labels is None or len(labels) == 0:
            return await self._annotate_dynamic(text, title)

        if not title:
            title = "[No title provided]"

        prompt = self._build_prompt(text, title, labels, few_shot_examples)

        try:
            if self._llm_client is not None:
                messages = [{"role": "user", "content": prompt}]
                response = await self._llm_client.chat(messages)
                content = response.content
            else:
                raise RuntimeError("LLM client not available")

            result = self._parse_response(content)

        except Exception as e:
            result = {
                "topic": "unknown",
                "confidence": 0.5,
                "reasoning": f"Error: {str(e)}",
                "title_influence": "",
            }

        return ContextualAnnotation(
            label=result.get("topic", "unknown"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            context_used=result.get("title_influence", ""),
        )

    async def _annotate_dynamic(
        self, text: str, title: str = ""
    ) -> ContextualAnnotation:
        """Annotate text without predefined labels - dynamic inference."""
        context = f"Title: {title}\n" if title else ""
        prompt = f"""You are a topic classification expert. Analyze the text with context and determine:
1. What is the main topic/category?
2. How does the context influence your decision?

{context}
TEXT: {text}

Respond ONLY with valid JSON:
{{"topic": "...", "confidence": 0.0-1.0, "reasoning": "...", "title_influence": "..."}}"""

        try:
            if self._llm_client is not None:
                messages = [{"role": "user", "content": prompt}]
                response = await self._llm_client.chat(messages)
                result = self._parse_response(response.content)
            else:
                result = {
                    "topic": "unknown",
                    "confidence": 0.5,
                    "confidence_level": "MEDIUM",
                    "reasoning": "LLM client not available",
                    "title_influence": "",
                }
        except Exception as e:
            result = {
                "topic": "unknown",
                "confidence": 0.5,
                "confidence_level": "MEDIUM",
                "reasoning": f"Error: {str(e)}",
                "title_influence": "",
            }

        return ContextualAnnotation(
            label=result.get("topic", "unknown"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            context_used=result.get("title_influence", ""),
        )

    def get_weight(self) -> float:
        return self.weight

    def unload(self):
        """Unload client (no-op for cloud API)."""
        self._llm_client = None
