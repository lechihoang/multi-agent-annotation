

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


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

            self._api_key = (
                api_key or os.getenv("NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
            )
            if not self._api_key:
                raise ValueError(
                    "NIM_API_KEY or NVIDIA_API_KEY environment variable is required"
                )

            if model is None:
                from src.config import get_config
                config = get_config()
                model = config.nvidia.model

            if base_url:
                self.base_url = base_url
            else:
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
        env_url = os.getenv("NIM_BASE_URL")
        if env_url:
            return env_url

        return "https://integrate.api.nvidia.com/v1"

    async def chat(self, messages: List[Dict[str, str]]) -> ChatResponse:
        import httpx
        import asyncio

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

        max_retries = 10
        retry_delay = 30

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
                            wait_time = int(retry_after) + 5
                        else:
                            wait_time = retry_delay * (2**attempt)
                            wait_time = min(wait_time, 300)

                        print(
                            f"  [429] Rate limited. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = response.json()

                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        print(f"  [429] HTTPStatusError caught. Retrying...")
                        await asyncio.sleep(30)
                        continue
                    raise e
                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    print(f"  [Network] {e}. Retrying in 10s...")
                    await asyncio.sleep(10)
                    continue

            else:
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

        content = message.get("content", "")
        if not content and message.get("reasoning_content"):
            content = message.get("reasoning_content", "")

        return ChatResponse(
            content=content,
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def is_available(self) -> bool:
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


def get_nim_client(
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> Optional[NimClient]:
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
