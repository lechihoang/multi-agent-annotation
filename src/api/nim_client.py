"""NVIDIA NIM API client for LLM inference.

NIM (NVIDIA Inference Microservice) provides access to NVIDIA-hosted models
via OpenAI-compatible API format.

Usage:
    - Set NVIDIA_API_KEY environment variable or NIM_API_KEY
    - Configure model in config.yaml
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    content: str
    usage: Dict[str, int]
    finish_reason: str


class NimClient:
    """Client for NVIDIA NIM API - OpenAI-compatible format."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize NIM client.

        Args:
            model: Model name (loaded from config.yaml if not provided)
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens to generate
            base_url: NIM API base URL (auto-detected if not provided)
            api_key: API key (uses NIM_API_KEY or NVIDIA_API_KEY env var if not provided)
        """
        try:
            from openai import OpenAI as OpenAILib

            # Get API key from env or parameter
            self._api_key = (
                api_key or os.getenv("NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
            )
            if not self._api_key:
                raise ValueError(
                    "NIM_API_KEY or NVIDIA_API_KEY environment variable is required"
                )

            # Load model from config.yaml if not provided
            if model is None:
                from src.config import get_config
                config = get_config()
                model = config.nvidia.model

            # Set base URL
            if base_url:
                self.base_url = base_url
            else:
                # Auto-detect base URL based on model
                self.base_url = self._get_base_url(model)

            self.client = OpenAILib(
                api_key=self._api_key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        except Exception as e:
            raise RuntimeError(f"NIM client init failed: {e}")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _get_base_url(self, model: str) -> str:
        """Get base URL for NIM API based on model name."""
        # Check env override first
        env_url = os.getenv("NIM_BASE_URL")
        if env_url:
            return env_url

        # NVIDIA NIM uses integrate.api.nvidia.com/v1 format
        # Model is passed in the request body, not URL
        return "https://integrate.api.nvidia.com/v1"

    async def chat(self, messages: List[Dict[str, str]]) -> ChatResponse:
        """Send chat completion request to NIM API."""
        import httpx
        import asyncio

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        max_retries = 10  # Increased for batch processing stability
        retry_delay = 30  # Increased base delay

        async with httpx.AsyncClient(timeout=120.0) as http_client:
            data = None
            response = None
            for attempt in range(max_retries):
                try:
                    response = await http_client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = int(retry_after) + 5  # Add 5s buffer
                        else:
                            # Exponential backoff with higher base: 30, 60, 120, ...
                            wait_time = retry_delay * (2**attempt)
                            # Cap wait time at 5 minutes
                            wait_time = min(wait_time, 300)

                        print(
                            f"  [429] Rate limited. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = response.json()

                    # Success - break retry loop
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        # Should have been handled above, but just in case
                        print(f"  [429] HTTPStatusError caught. Retrying...")
                        await asyncio.sleep(30)
                        continue
                    raise e
                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    print(f"  [Network] {e}. Retrying in 10s...")
                    await asyncio.sleep(10)
                    continue

            else:
                # If loop finishes without success
                if response:
                    response.raise_for_status()
                else:
                    raise RuntimeError(
                        "Failed to get response from NIM API after multiple retries"
                    )

        if data is None:
            raise RuntimeError("No data received from NIM API")

        choice = data["choices"][0]
        message = choice["message"]

        # Handle DeepSeek R1 format (reasoning_content + content)
        content = message.get("content", "")
        if not content and message.get("reasoning_content"):
            content = message.get("reasoning_content", "")

        return ChatResponse(
            content=content,
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def structured(
        self, messages: List[Dict[str, str]], schema: Dict[str, Any]
    ) -> ChatResponse:
        """Send structured output request with JSON schema."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_object",
                "schema": schema,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as http_client:
            response = await http_client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        message = choice["message"]

        # Handle DeepSeek R1 format (reasoning_content + content)
        content = message.get("content", "")
        if not content and message.get("reasoning_content"):
            content = message.get("reasoning_content", "")

        return ChatResponse(
            content=content,
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def is_available(self) -> bool:
        """Check if NIM API is available."""
        import httpx

        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }

            with httpx.Client(timeout=10.0) as http_client:
                response = http_client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                return response.status_code == 200
        except Exception:
            return False


# Factory function to get the appropriate client
def get_nim_client(
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> Optional[NimClient]:
    """Create and return NIM client if API key is available.

    Model is loaded from config.yaml if not provided.
    """
    api_key = os.getenv("NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return None

    try:
        return NimClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"Warning: Could not initialize NIM client: {e}")
        return None
