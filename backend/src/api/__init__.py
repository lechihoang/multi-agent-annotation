"""
API clients for LLM providers.

Provides unified interface for Groq, HuggingFace, and Ollama.
"""

from .base import (
    APIClientError,
    BaseAPIClient,
    ConnectionError,
    ModelNotFoundError,
    RateLimitError,
)
from .groq_client import GroqClient
from .huggingface_client import HuggingFaceClient
from .manager import APIManager, get_api_manager
from .ollama_client import OllamaClient

__all__ = [
    # Base
    "BaseAPIClient",
    "APIClientError",
    "RateLimitError",
    "ModelNotFoundError",
    "ConnectionError",
    # Clients
    "GroqClient",
    "HuggingFaceClient",
    "OllamaClient",
    # Manager
    "APIManager",
    "get_api_manager",
]
