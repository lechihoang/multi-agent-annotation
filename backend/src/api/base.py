"""
Base API client interface for all LLM providers.

Defines the abstract interface that all API clients must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..models import ChatResponse


class BaseAPIClient(ABC):
    """Abstract base class for LLM API clients."""

    name: str = "base"
    is_available: bool = False

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """
        Execute a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (uses default if None)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            response_format: Optional format spec (e.g., {"type": "json_object"})

        Returns:
            ChatResponse with content, model, usage, and latency
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the API is available and responding.

        Returns:
            True if healthy, False otherwise
        """
        pass

    async def structured_output(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> ChatResponse:
        """
        Get structured JSON output from the model.

        Args:
            messages: List of message dicts
            schema: JSON schema for the expected output
            model: Model identifier
            temperature: Sampling temperature

        Returns:
            ChatResponse with JSON content
        """
        return await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict[str, str]]:
        """Helper to build standard message format."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


class APIClientError(Exception):
    """Base exception for API client errors."""

    def __init__(self, message: str, provider: str, original_error: Exception | None = None):
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class RateLimitError(APIClientError):
    """Raised when rate limit is exceeded."""

    def __init__(self, provider: str, retry_after: float | None = None):
        self.retry_after = retry_after
        message = f"Rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after}s"
        super().__init__(message, provider)


class ModelNotFoundError(APIClientError):
    """Raised when requested model is not available."""

    def __init__(self, provider: str, model: str):
        self.model = model
        super().__init__(f"Model '{model}' not found", provider)


class ConnectionError(APIClientError):
    """Raised when connection to API fails."""

    pass
