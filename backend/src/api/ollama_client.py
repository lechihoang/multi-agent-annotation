"""
Ollama client for local LLM inference.

Zero-cost processing using locally hosted models.
"""

import time
from typing import Any

import structlog
from ollama import AsyncClient

from ..config.settings import settings
from ..models import ChatResponse
from .base import (
    APIClientError,
    BaseAPIClient,
    ConnectionError,
    ModelNotFoundError,
)

logger = structlog.get_logger()


class OllamaClient(BaseAPIClient):
    """
    Ollama client for local inference.

    Provides zero-cost inference using locally running Ollama server.
    Useful for validation, preprocessing, and fallback scenarios.
    """

    name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        default_model: str | None = None,
    ):
        """
        Initialize Ollama client.

        Args:
            host: Ollama server URL (defaults to localhost:11434)
            default_model: Default model to use
        """
        self.host = host or settings.ollama_host
        self.default_model = default_model or settings.default_ollama_model
        self.client = AsyncClient(host=self.host)
        self.is_available = True  # Assume available, verify with health_check

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """
        Execute chat completion with Ollama.

        Args:
            messages: List of message dicts
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            response_format: Optional format (supports JSON)

        Returns:
            ChatResponse with completion result
        """
        model = model or self.default_model
        start_time = time.perf_counter()

        try:
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }

            # Handle JSON mode
            format_spec = None
            if response_format and response_format.get("type") == "json_object":
                format_spec = "json"

            response = await self.client.chat(
                model=model,
                messages=messages,
                options=options,
                format=format_spec,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            content = response.message.content or ""
            usage = {
                "prompt_tokens": response.prompt_eval_count or 0,
                "completion_tokens": response.eval_count or 0,
                "total_tokens": (response.prompt_eval_count or 0)
                + (response.eval_count or 0),
            }

            logger.info(
                "ollama_completion_success",
                model=model,
                latency_ms=round(latency_ms, 2),
                tokens=usage.get("total_tokens", 0),
            )

            return ChatResponse(
                content=content,
                model=model,
                usage=usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_str = str(e).lower()

            logger.error(
                "ollama_completion_error",
                model=model,
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )

            if "not found" in error_str or "does not exist" in error_str:
                raise ModelNotFoundError(self.name, model)

            if "connection" in error_str or "refused" in error_str:
                self.is_available = False
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.host}", self.name, e
                )

            raise APIClientError(str(e), self.name, e)

    async def health_check(self) -> bool:
        """
        Check Ollama server availability.

        Returns:
            True if server is responding
        """
        try:
            # List models to verify connectivity
            models = await self.client.list()
            self.is_available = True
            return True
        except Exception as e:
            logger.warning("ollama_health_check_failed", error=str(e))
            self.is_available = False
            return False

    async def list_models(self) -> list[str]:
        """
        List available Ollama models.

        Returns:
            List of model names
        """
        try:
            response = await self.client.list()
            return [m.model for m in response.models]
        except Exception as e:
            logger.warning("ollama_list_models_failed", error=str(e))
            return []

    async def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        try:
            logger.info("ollama_pulling_model", model=model)
            await self.client.pull(model)
            logger.info("ollama_model_pulled", model=model)
            return True
        except Exception as e:
            logger.error("ollama_pull_failed", model=model, error=str(e))
            return False

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> str:
        """
        Simple text generation (non-chat).

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text
        """
        model = model or self.default_model

        try:
            response = await self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            return response.response
        except Exception as e:
            logger.error("ollama_generate_error", error=str(e))
            raise APIClientError(str(e), self.name, e)
