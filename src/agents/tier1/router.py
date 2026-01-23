"""Tier 1: Router Agent - Dynamic task routing using TaskParser.

Uses TaskParser to parse user prompt and extract task definition.
Fully dynamic - no hardcoded task types or labels.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import json

from ...config import get_config, get_llm_client
from .parser import TaskParser, TaskType, TaskDefinition, LabelDefinition


class RouterMode(Enum):
    """Router operation mode."""

    DYNAMIC = "dynamic"  # Parse from prompt
    PREDEFINED = "predefined"  # Use provided config


@dataclass
class Task:
    """Input task for routing."""

    id: str
    text: str
    metadata: Dict[str, Any] | None = None


@dataclass
class RoutingResult:
    """Result from router agent."""

    task_id: str
    task_type: TaskType
    labels: List[LabelDefinition]
    confidence: float
    reasoning: str
    text_column: str
    input_file: str
    output_column: str
    mode: RouterMode = RouterMode.DYNAMIC


class RouterAgent:
    """MAFA Tier 1: Dynamic router using TaskParser.

    Key features:
    - Parses user prompt to extract task_type, labels, columns
    - Uses Groq API to infer label descriptions
    - Supports various CSV formats
    - No hardcoded task types or labels
    """

    def __init__(
        self,
        mode: RouterMode = RouterMode.DYNAMIC,
        llm_client: Any | None = None,
    ):
        self.config = get_config()
        self.mode = mode
        self._llm_client = llm_client
        self._parser = TaskParser(groq_client=llm_client)

        # Initialize LLM if not provided
        if self._llm_client is None:
            self._init_llm()

    def _init_llm(self):
        """Initialize LLM client."""
        try:
            self._llm_client = get_llm_client(self.config)
            self._parser = TaskParser(groq_client=self._llm_client)
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            self._llm_client = None

    def _build_prompt(self, text: str, labels: List[str]) -> str:
        """Build prompt for classification (fallback when no parser result)."""
        labels_str = ", ".join(labels)

        return f"""You are a classification expert. Classify the text into ONE of the provided labels.

LABELS: {labels_str}

TEXT: {text}

Respond ONLY with valid JSON:
{{"label": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from model response."""
        content = content.strip()

        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{[^{}]*\}", content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {"label": "unknown", "confidence": 0.5, "reasoning": "Parse error"}

    async def analyze(
        self,
        task: Task,
        prompt: str = "",
        csv_path: str = "",
    ) -> RoutingResult:
        """Analyze task and return routing decision.

        If prompt and csv_path are provided, uses TaskParser for full parsing.
        Otherwise falls back to simple label inference.
        """
        # Use TaskParser if prompt and csv provided
        if prompt and csv_path:
            return await self._parse_with_prompt(task, prompt, csv_path)

        # Fallback: simple inference
        return await self._simple_inference(task)

    async def _parse_with_prompt(
        self,
        task: Task,
        prompt: str,
        csv_path: str,
    ) -> RoutingResult:
        """Parse prompt to get task definition."""
        try:
            # Parse task definition from prompt
            task_def = await self._parser.parse_with_inference(prompt, csv_path)

            return RoutingResult(
                task_id=task.id,
                task_type=task_def.task_type,
                labels=task_def.labels,
                confidence=0.9,  # High confidence when using explicit prompt
                reasoning=f"Parsed from prompt: {task_def.task_description}",
                text_column=task_def.text_column,
                input_file=task_def.input_file,
                output_column=task_def.output_column,
                mode=RouterMode.DYNAMIC,
            )

        except Exception as e:
            print(f"Prompt parsing error: {e}")
            return await self._simple_inference(task)

    async def _simple_inference(self, task: Task) -> RoutingResult:
        """Simple inference when no prompt provided."""
        text = task.text.lower()

        # Detect task type
        if any(
            kw in text for kw in ["good", "bad", "great", "terrible", "love", "hate"]
        ):
            task_type = TaskType.SENTIMENT
        elif any(kw in text for kw in ["who", "where", "when", "name", "city"]):
            task_type = TaskType.NER
        elif any(kw in text for kw in ["can you", "how to", "i want", "help"]):
            task_type = TaskType.INTENT
        else:
            task_type = TaskType.CLASSIFICATION

        # Return with no labels (Tier 2 will handle dynamic inference)
        return RoutingResult(
            task_id=task.id,
            task_type=task_type,
            labels=[],
            confidence=0.5,
            reasoning="Simple inference (no prompt provided)",
            text_column="text",
            input_file="",
            output_column="predicted_label",
            mode=RouterMode.DYNAMIC,
        )

    def parse_prompt(self, prompt: str, csv_path: str) -> TaskDefinition:
        """Parse prompt to get task definition (sync version)."""
        import asyncio

        return asyncio.run(self._parser.parse_with_inference(prompt, csv_path))

    def get_label_names(self, result: RoutingResult) -> List[str]:
        """Extract just the label names from a routing result."""
        return [label.name for label in result.labels]


# Backward compatibility
DynamicRouterAgent = RouterAgent
