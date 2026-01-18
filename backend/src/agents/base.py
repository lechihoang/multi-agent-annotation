"""
Base agent interface for annotation agents.

Defines the abstract interface that all annotation agents must implement.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import structlog
import yaml

from ..api import BaseAPIClient, get_api_manager
from ..config.settings import settings
from ..models import Annotation, ChatResponse

logger = structlog.get_logger()


class BaseAnnotationAgent(ABC):
    """
    Abstract base class for annotation agents.

    All specialized agents (Intent, Entity, FAQ) inherit from this class.
    """

    name: str = "base_agent"
    weight: float = 0.33
    tier: int = 2

    def __init__(
        self,
        primary_provider: str = "groq",
        fallback_provider: str | None = "huggingface",
        primary_model: str | None = None,
        fallback_model: str | None = None,
    ):
        """
        Initialize base agent.

        Args:
            primary_provider: Primary API provider
            fallback_provider: Fallback API provider
            primary_model: Model for primary provider
            fallback_model: Model for fallback provider
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.primary_model = primary_model
        self.fallback_model = fallback_model

        self._api_manager = get_api_manager()
        self._prompts = self._load_prompts()

    def _load_prompts(self) -> dict[str, Any]:
        """Load prompts from YAML configuration."""
        try:
            with open(settings.prompts_yaml_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"failed_to_load_prompts: {e}")
            return {}

    @abstractmethod
    async def annotate(self, text: str, **kwargs) -> Annotation:
        """
        Perform annotation on the given text.

        Args:
            text: Text to annotate
            **kwargs: Additional parameters

        Returns:
            Annotation with results and confidence
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    def get_user_prompt(self, text: str, **kwargs) -> str:
        """Get the user prompt with the text to annotate."""
        pass

    @abstractmethod
    def parse_response(self, response: ChatResponse) -> dict[str, Any]:
        """Parse the LLM response into structured output."""
        pass

    async def _execute_with_fallback(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        json_mode: bool = True,
    ) -> ChatResponse:
        """
        Execute LLM call with automatic fallback.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_mode: Whether to request JSON output

        Returns:
            ChatResponse from successful provider
        """
        providers_to_try = [self.primary_provider]
        if self.fallback_provider:
            providers_to_try.append(self.fallback_provider)

        response_format = {"type": "json_object"} if json_mode else None
        last_error: Exception | None = None

        for provider_name in providers_to_try:
            try:
                client = self._api_manager.get_client(provider_name)
                if not client.is_available:
                    continue

                model = (
                    self.primary_model
                    if provider_name == self.primary_provider
                    else self.fallback_model
                )

                logger.info(
                    "agent_executing",
                    agent=self.name,
                    provider=provider_name,
                    model=model,
                )

                response = await client.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )

                return response

            except Exception as e:
                logger.warning(
                    "agent_provider_failed",
                    agent=self.name,
                    provider=provider_name,
                    error=str(e),
                )
                last_error = e
                continue

        raise RuntimeError(
            f"All providers failed for {self.name}. Last error: {last_error}"
        )

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """
        Parse JSON from LLM response, handling common issues.

        Args:
            content: Raw response content

        Returns:
            Parsed JSON dict
        """
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in the text
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(
            "failed_to_parse_json",
            agent=self.name,
            content_preview=content[:200],
        )

        return {"error": "Failed to parse response", "raw_content": content}

    def _build_messages(
        self,
        text: str,
        **kwargs,
    ) -> list[dict[str, str]]:
        """Build message list for LLM call."""
        return [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.get_user_prompt(text, **kwargs)},
        ]

    def _create_annotation(
        self,
        result: dict[str, Any],
        latency_ms: float,
    ) -> Annotation:
        """Create Annotation object from parsed result."""
        confidence = result.get("confidence", 0.5)

        # Ensure confidence is a float
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 0.5

        return Annotation(
            agent_name=self.name,
            confidence=max(0.0, min(1.0, confidence)),
            weight=self.weight,
            result=result,
            latency_ms=latency_ms,
        )
