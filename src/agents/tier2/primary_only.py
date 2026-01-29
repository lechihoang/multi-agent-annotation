

from typing import Dict, Any, List, Optional
import json

from ...config import get_config, get_llm_client
from .arq_prompts import ARQPromptBuilder, ARQPrompt


class PrimaryOnlyAnnotation:
    """Result from primary-only analysis agent with ARQ-style output."""

    def __init__(
        self,
        label: str,
        confidence: float,
        reasoning: str,
        confidence_level: str = "MEDIUM",
    ):
        self.label = label
        self.confidence = confidence
        self.confidence_level = confidence_level
        self.reasoning = reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "reasoning": self.reasoning,
        }


class PrimaryOnlyAgent:

    def __init__(self):
        self.config = get_config()
        self._llm_client = None
        self._init_llm()

    def _init_llm(self):
        self._llm_client = get_llm_client(self.config)
        if self._llm_client is None:
            print("Warning: No LLM client available (neither Groq nor NIM)")

    def _build_prompt(
        self,
        text: str,
        labels: List[str],
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        arq_prompt = ARQPromptBuilder.build_toxicity_arq(
            text=text, examples=few_shot_examples or [], agent_type="primary"
        )
        return ARQPromptBuilder.to_prompt(arq_prompt)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        result = ARQPromptBuilder.parse_response(content)

        return {
            "topic": result.get("final_label", "unknown"),
            "confidence": result.get("confidence_score", 0.5),
            "confidence_level": result.get("confidence", "MEDIUM"),
            "reasoning": result.get("reasoning", ""),
        }

    async def annotate(
        self,
        text: str,
        labels: List[str] | None = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> PrimaryOnlyAnnotation:
        if labels is None or len(labels) == 0:
            raise ValueError("Labels must be provided for PrimaryOnlyAgent")

        prompt = self._build_prompt(text, labels, few_shot_examples)

        if self._llm_client is None:
            raise RuntimeError("LLM client not available")

        messages = [{"role": "user", "content": prompt}]
        response = await self._llm_client.chat(messages)
        content = response.content

        result = self._parse_response(content)

        return PrimaryOnlyAnnotation(
            label=result.get("topic", "unknown"),
            confidence=result.get("confidence", 0.5),
            confidence_level=result.get("confidence_level", "MEDIUM"),
            reasoning=result.get("reasoning", ""),
        )

    def get_weight(self) -> float:
        return self.config.agents.primary_only

    def unload(self):
        self._llm_client = None
