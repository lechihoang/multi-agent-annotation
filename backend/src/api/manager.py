"""
API Manager for routing and fallback between providers.

Handles intelligent routing based on task complexity and provider availability.
"""

from typing import Any

import structlog

from ..models import ChatResponse, ComplexityLevel
from .base import APIClientError, BaseAPIClient, RateLimitError
from .groq_client import GroqClient
from .huggingface_client import HuggingFaceClient
from .ollama_client import OllamaClient

logger = structlog.get_logger()


class APIManager:
    """
    Manages API clients with intelligent routing and fallback.

    Routes requests to appropriate providers based on:
    - Task complexity
    - Provider availability
    - Cost optimization
    """

    def __init__(self):
        """Initialize API manager with all clients."""
        self.groq = GroqClient()
        self.huggingface = HuggingFaceClient()
        self.ollama = OllamaClient()

        self._providers: dict[str, BaseAPIClient] = {
            "groq": self.groq,
            "huggingface": self.huggingface,
            "ollama": self.ollama,
        }

        # Routing configuration based on complexity
        self._routing_config = {
            ComplexityLevel.HIGH: {
                "primary": "groq",
                "fallback": ["huggingface", "ollama"],
                "models": {
                    "groq": "deepseek-r1-distill-llama-70b",
                    "huggingface": "meta-llama/Llama-3.2-8B-Instruct",
                    "ollama": "llama3.2",
                },
            },
            ComplexityLevel.MEDIUM: {
                "primary": "groq",
                "fallback": ["huggingface", "ollama"],
                "models": {
                    "groq": "llama-3.3-70b-versatile",
                    "huggingface": "meta-llama/Llama-3.2-8B-Instruct",
                    "ollama": "llama3.2",
                },
            },
            ComplexityLevel.LOW: {
                "primary": "huggingface",
                "fallback": ["ollama", "groq"],
                "models": {
                    "huggingface": "meta-llama/Llama-3.2-8B-Instruct",
                    "ollama": "llama3.2",
                    "groq": "llama-3.1-8b-instant",
                },
            },
        }

    async def health_check_all(self) -> dict[str, bool]:
        """
        Check health of all providers.

        Returns:
            Dict of provider name to health status
        """
        results = {}
        for name, client in self._providers.items():
            results[name] = await client.health_check()
            logger.info(f"{name}_health_status", healthy=results[name])
        return results

    def get_client(self, provider: str) -> BaseAPIClient:
        """
        Get a specific client by name.

        Args:
            provider: Provider name (groq, huggingface, ollama)

        Returns:
            The API client

        Raises:
            ValueError: If provider not found
        """
        if provider not in self._providers:
            raise ValueError(f"Unknown provider: {provider}")
        return self._providers[provider]

    def get_client_for_complexity(
        self, complexity: ComplexityLevel
    ) -> tuple[BaseAPIClient, str]:
        """
        Get the optimal client for a given complexity level.

        Args:
            complexity: Task complexity level

        Returns:
            Tuple of (client, model_name)
        """
        config = self._routing_config[complexity]
        primary = config["primary"]
        client = self._providers[primary]
        model = config["models"][primary]

        if client.is_available:
            return client, model

        # Try fallbacks
        for fallback in config["fallback"]:
            client = self._providers[fallback]
            if client.is_available:
                model = config["models"][fallback]
                logger.info(
                    "using_fallback_provider",
                    complexity=complexity.value,
                    primary=primary,
                    fallback=fallback,
                )
                return client, model

        # No available provider
        raise APIClientError(
            "No available API providers",
            "manager",
        )

    async def execute_with_fallback(
        self,
        messages: list[dict[str, str]],
        complexity: ComplexityLevel = ComplexityLevel.MEDIUM,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """
        Execute a request with automatic fallback on failure.

        Args:
            messages: Chat messages
            complexity: Task complexity for routing
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            response_format: Optional JSON mode

        Returns:
            ChatResponse from successful provider

        Raises:
            APIClientError: If all providers fail
        """
        config = self._routing_config[complexity]
        providers_to_try = [config["primary"]] + config["fallback"]
        last_error: Exception | None = None

        for provider_name in providers_to_try:
            client = self._providers[provider_name]
            if not client.is_available:
                continue

            model = config["models"][provider_name]

            try:
                logger.info(
                    "attempting_provider",
                    provider=provider_name,
                    model=model,
                    complexity=complexity.value,
                )

                response = await client.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )

                logger.info(
                    "provider_success",
                    provider=provider_name,
                    latency_ms=response.latency_ms,
                )

                return response

            except RateLimitError as e:
                logger.warning(
                    "provider_rate_limited",
                    provider=provider_name,
                    retry_after=e.retry_after,
                )
                last_error = e
                continue

            except APIClientError as e:
                logger.warning(
                    "provider_failed",
                    provider=provider_name,
                    error=str(e),
                )
                last_error = e
                continue

        # All providers failed
        raise APIClientError(
            f"All providers failed. Last error: {last_error}",
            "manager",
            last_error,
        )

    async def execute_parallel(
        self,
        messages: list[dict[str, str]],
        providers: list[str] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, ChatResponse | Exception]:
        """
        Execute requests to multiple providers in parallel.

        Useful for consensus mechanisms where multiple opinions are needed.

        Args:
            messages: Chat messages
            providers: List of providers to query (defaults to all available)
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Dict of provider name to response or exception
        """
        import asyncio

        if providers is None:
            providers = [name for name, client in self._providers.items() if client.is_available]

        async def query_provider(name: str) -> tuple[str, ChatResponse | Exception]:
            client = self._providers[name]
            try:
                response = await client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return name, response
            except Exception as e:
                return name, e

        tasks = [query_provider(name) for name in providers]
        results = await asyncio.gather(*tasks)

        return dict(results)

    def get_available_providers(self) -> list[str]:
        """
        Get list of currently available providers.

        Returns:
            List of provider names
        """
        return [name for name, client in self._providers.items() if client.is_available]

    async def refresh_availability(self) -> None:
        """Refresh availability status of all providers."""
        await self.health_check_all()


# Singleton instance
_manager: APIManager | None = None


def get_api_manager() -> APIManager:
    """Get the singleton API manager instance."""
    global _manager
    if _manager is None:
        _manager = APIManager()
    return _manager
