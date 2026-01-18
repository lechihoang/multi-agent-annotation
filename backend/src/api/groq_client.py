"""
Groq API client for high-speed LLM inference.

Supports DeepSeek-R1 and Llama models with rate limiting and error handling.
"""

import time
from typing import Any

import structlog
from groq import AsyncGroq, RateLimitError as GroqRateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config.settings import settings
from ..models import ChatResponse
from .base import (
    APIClientError,
    BaseAPIClient,
    ConnectionError,
    ModelNotFoundError,
    RateLimitError,
)

logger = structlog.get_logger()


class GroqClient(BaseAPIClient):
    """
    Groq API client for fast inference.

    Models supported:
    - deepseek-r1-distill-llama-70b (reasoning)
    - llama-3.3-70b-versatile (general)
    - llama-3.1-8b-instant (fast fallback)
    """

    name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
    ):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key (uses env if not provided)
            default_model: Default model to use
        """
        self.api_key = api_key or settings.groq_api_key
        self.default_model = default_model or settings.default_groq_model
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None
        self.is_available = bool(self.api_key)

        if not self.is_available:
            logger.warning("Groq client initialized without API key")

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """
        Execute chat completion with Groq API.

        Args:
            messages: List of message dicts
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            response_format: Optional JSON mode

        Returns:
            ChatResponse with completion result

        Raises:
            APIClientError: On API errors
            RateLimitError: On rate limit
        """
        if not self.client:
            raise ConnectionError("Groq client not initialized (missing API key)", self.name)

        model = model or self.default_model
        start_time = time.perf_counter()

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if response_format:
                kwargs["response_format"] = response_format

            response = await self.client.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            logger.info(
                "groq_completion_success",
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

        except GroqRateLimitError as e:
            logger.warning("groq_rate_limit", model=model, error=str(e))
            raise RateLimitError(self.name, retry_after=60)

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "groq_completion_error",
                model=model,
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )

            if "model" in str(e).lower() and "not found" in str(e).lower():
                raise ModelNotFoundError(self.name, model)

            raise APIClientError(str(e), self.name, e)

    async def health_check(self) -> bool:
        """
        Check Groq API availability.

        Returns:
            True if API is responding
        """
        if not self.client:
            return False

        try:
            # Use a minimal request to check connectivity
            response = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            return bool(response.choices)
        except Exception as e:
            logger.warning("groq_health_check_failed", error=str(e))
            return False

    async def list_models(self) -> list[str]:
        """
        List available Groq models.

        Returns:
            List of model IDs
        """
        if not self.client:
            return []

        try:
            models = await self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.warning("groq_list_models_failed", error=str(e))
            return []
