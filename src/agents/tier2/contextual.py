

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

    def __init__(self):
        self.config = get_config()
        self.weight = self.config.agents.contextual
        self._llm_client = None
        self._init_llm()

    def _init_llm(self):
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
        arq_prompt = ARQPromptBuilder.build_toxicity_arq(
            text=text,
            examples=few_shot_examples or [],
            context=title,
            agent_type="contextual",
        )
        return ARQPromptBuilder.to_prompt(arq_prompt)

    def _parse_response(self, content: str) -> Dict[str, Any]:
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
        if labels is None or len(labels) == 0:
            raise ValueError("Labels must be provided for ContextualAgent")

        if not title:
            title = "[No title provided]"

        prompt = self._build_prompt(text, title, labels, few_shot_examples)

        if self._llm_client is None:
            raise RuntimeError("LLM client not available")

        messages = [{"role": "user", "content": prompt}]
        response = await self._llm_client.chat(messages)
        content = response.content

        result = self._parse_response(content)

        return ContextualAnnotation(
            label=result.get("topic", "unknown"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            context_used=result.get("title_influence", ""),
        )

    def get_weight(self) -> float:
        return self.weight

    def unload(self):
        self._llm_client = None
