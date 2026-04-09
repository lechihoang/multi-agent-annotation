"""
NIM Client — NVIDIA NIM API with multi-key support.
Reads NVIDIA_API_KEYS env var (comma-separated) for key pooling.
Round-robin: picks the least-recently-used key on each request.
"""

import asyncio
import time
import os
from dotenv import load_dotenv
load_dotenv()
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "meta/llama-3.3-70b-instruct"


def _parse_keys() -> list[str]:
    """Parse NVIDIA_API_KEYS env var (comma-separated) or fall back to single key."""
    keys_str = os.getenv("NVIDIA_API_KEYS", "")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if len(keys) > 1:
            logger.info(f"Key pool: {len(keys)} keys detected")
            return keys
    # Fall back to single key
    fallback = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY", "")
    if fallback:
        return [fallback]
    return []


# ---------------------------------------------------------------------------
# Per-key client
# ---------------------------------------------------------------------------

class _KeyClient:
    """One ChatOpenAI instance bound to one API key."""

    def __init__(self, api_key: str, rate_limit: int):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.llm = ChatOpenAI(
            model=MODEL,
            api_key=api_key,
            base_url=BASE_URL,
            temperature=0.0,
            max_tokens=4096,
            max_retries=0,
        )
        self._lock = asyncio.Lock()
        self._last_call = 0.0
        self._request_count = 0

    def _to_lc_messages(self, messages: list[dict]):
        return [
            SystemMessage(content=m["content"]) if m["role"] == "system"
            else HumanMessage(content=m["content"])
            for m in messages
        ]

    async def _wait(self):
        async with self._lock:
            elapsed = time.time() - self._last_call
            min_interval = 60.0 / self.rate_limit * 1.5
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_call = time.time()
            self._request_count += 1

    async def call(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None,
        temperature: float | None,
        max_tokens: int | None,
        max_retries: int,
    ) -> str | BaseModel:
        await self._wait()

        last_error = None
        current_max_tokens = max_tokens or 4096

        for attempt in range(max_retries + 1):
            try:
                kwargs = {"temperature": temperature} if temperature is not None else {}
                kwargs["max_tokens"] = current_max_tokens

                if response_model is not None:
                    structured_llm = self.llm.with_structured_output(response_model)
                    result = await structured_llm.ainvoke(
                        self._to_lc_messages(messages), **kwargs
                    )
                else:
                    result = await self.llm.ainvoke(
                        self._to_lc_messages(messages), **kwargs
                    )
                    result = result.content

                return result

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "length" in err_str or "maximum context" in err_str:
                    current_max_tokens = int(current_max_tokens * 1.5)
                    logger.warning(f"[{self.api_key[:12]}...] Retry {attempt+1} (length): {e}")
                elif "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                    logger.warning(f"[{self.api_key[:12]}...] Retry {attempt+1} (429): {e}")
                else:
                    logger.warning(f"[{self.api_key[:12]}...] Retry {attempt+1}: {e}")

                await asyncio.sleep(30 * (2 ** attempt))

        raise last_error


# ---------------------------------------------------------------------------
# Pooled client
# ---------------------------------------------------------------------------

class NimClient:
    """
    NVIDIA NIM client with automatic key pooling.
    Uses round-robin (least-recently-used) to distribute requests across keys.
    """

    def __init__(
        self,
        rate_limit: int = 40,
        max_retries: int = 10,
    ):
        keys = _parse_keys()
        if not keys:
            raise ValueError("No NVIDIA API key found. Set NVIDIA_API_KEY or NVIDIA_API_KEYS env var.")

        self._clients: list[_KeyClient] = [
            _KeyClient(k, rate_limit) for k in keys
        ]
        self._pool_size = len(self._clients)
        self._round_robin = 0
        self._lock = asyncio.Lock()
        logger.info(f"NimClient pool: {self._pool_size} keys, {rate_limit} req/min each")

    def _pick_client(self) -> _KeyClient:
        """Round-robin: pick the next client in rotation."""
        client = self._clients[self._round_robin]
        self._round_robin = (self._round_robin + 1) % self._pool_size
        return client

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 10,
    ) -> str:
        """Unstructured chat — returns raw text."""
        client = self._pick_client()
        return await client.call(
            messages=messages,
            response_model=None,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    async def chat_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 10,
    ) -> BaseModel:
        """Structured chat — returns Pydantic model."""
        client = self._pick_client()
        return await client.call(
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def stats(self) -> dict:
        """Return per-key stats."""
        return {
            f"key_{i}": {"requests": c._request_count, "last": c._last_call}
            for i, c in enumerate(self._clients)
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_nim_client: NimClient | None = None


def get_nim_client(rate_limit: int = 40, max_retries: int = 10) -> NimClient:
    global _nim_client
    if _nim_client is None:
        _nim_client = NimClient(rate_limit=rate_limit, max_retries=max_retries)
    return _nim_client


def reset_nim_client():
    global _nim_client
    _nim_client = None
