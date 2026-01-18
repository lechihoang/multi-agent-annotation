"""
HuggingFace Inference API client.

Supports HuggingFace Inference Providers for pattern matching and general tasks.
"""

import time
from typing import Any

import structlog
from huggingface_hub import AsyncInferenceClient
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


class HuggingFaceClient(BaseAPIClient):
    """
    HuggingFace Inference API client.

    Uses the Inference Providers API with OpenAI-compatible interface.
    Supports models like Llama-3.2-8B-Instruct.
    """

    name = "huggingface"

    def __init__(
        self,
        token: str | None = None,
        default_model: str | None = None,
    ):
        """
        Initialize HuggingFace client.

        Args:
            token: HuggingFace token (uses env if not provided)
            default_model: Default model to use
        """
        self.token = token or settings.hf_token
        self.default_model = default_model or settings.default_hf_model
        self.client = AsyncInferenceClient(token=self.token) if self.token else None
        self.is_available = bool(self.token)

        if not self.is_available:
            logger.warning("HuggingFace client initialized without token")

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
        Execute chat completion with HuggingFace Inference API.

        Args:
            messages: List of message dicts
            model: Model identifier (HuggingFace model ID)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            response_format: Optional JSON mode

        Returns:
            ChatResponse with completion result
        """
        if not self.client:
            raise ConnectionError(
                "HuggingFace client not initialized (missing token)", self.name
            )

        model = model or self.default_model
        start_time = time.perf_counter()

        try:
            response = await self.client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            logger.info(
                "huggingface_completion_success",
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
                "huggingface_completion_error",
                model=model,
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )

            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(self.name, retry_after=60)

            if "not found" in error_str or "404" in error_str:
                raise ModelNotFoundError(self.name, model)

            raise APIClientError(str(e), self.name, e)

    async def health_check(self) -> bool:
        """
        Check HuggingFace API availability.

        Returns:
            True if API is responding
        """
        if not self.client:
            return False

        try:
            # Check model status
            status = await self.client.get_model_status(self.default_model)
            return status.state == "Loadable" or status.state == "Loaded"
        except Exception as e:
            logger.warning("huggingface_health_check_failed", error=str(e))
            # Try a simple request instead
            try:
                await self.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
                return True
            except Exception:
                return False

    async def text_generation(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        """
        Simple text generation (non-chat).

        Args:
            prompt: Input prompt
            model: Model identifier
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if not self.client:
            raise ConnectionError(
                "HuggingFace client not initialized", self.name
            )

        model = model or self.default_model

        try:
            response = await self.client.text_generation(
                prompt=prompt,
                model=model,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            return response
        except Exception as e:
            logger.error("huggingface_text_generation_error", error=str(e))
            raise APIClientError(str(e), self.name, e)
