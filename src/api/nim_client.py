"""
NIM Client — NVIDIA NIM API with multi-key support.
Reads NVIDIA_API_KEY env var (single key or comma-separated keys) for key pooling.
Round-robin distribution + centralized retry behavior via BaseLLMClient.
"""

import asyncio
import time
import os
import logging
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from src.api.base_client import BaseLLMClient
from src.config import get_config

logger = logging.getLogger(__name__)


def _parse_keys() -> list[str]:
    """Parse NVIDIA_API_KEY env var (supports comma-separated values)."""
    keys_str = os.getenv("NVIDIA_API_KEY", "")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if len(keys) >= 1:
            logger.info(f"Key pool: {len(keys)} keys detected")
            return keys
    return []


class NimKeyClient(BaseLLMClient):
    """Client for a single API key."""
    def __init__(self, api_key: str, rate_limit: int, max_retries: int, model: str, base_url: str, default_max_tokens: int):
        super().__init__(max_retries=max_retries)
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.default_max_tokens = default_max_tokens
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=default_max_tokens,
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

    async def _wait_rate_limit(self):
        async with self._lock:
            elapsed = time.time() - self._last_call
            min_interval = 60.0 / self.rate_limit
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_call = time.time()
            self._request_count += 1

    async def _call_api(self, messages: list[dict], response_model: type[BaseModel] | None, temperature: float | None, max_tokens: int | None):
        await self._wait_rate_limit()
        kwargs = {"temperature": temperature} if temperature is not None else {}
        kwargs["max_tokens"] = max_tokens
        
        lc_msgs = self._to_lc_messages(messages)
        if response_model is not None:
            return await self.llm.with_structured_output(response_model).ainvoke(lc_msgs, **kwargs)
        
        result = await self.llm.ainvoke(lc_msgs, **kwargs)
        return result.content

    async def chat(
        self, 
        messages: list[dict], 
        response_model: type[BaseModel] | None = None, 
        temperature: float | None = None, 
        max_tokens: int | None = None, 
        max_retries: int | None = None
    ):
        current_max_tokens = max_tokens or self.default_max_tokens
        
        async def _invoke():
            nonlocal current_max_tokens
            try:
                return await self._call_api(messages, response_model, temperature, current_max_tokens)
            except Exception as e:
                err_str = str(e).lower()
                if "length" in err_str or "maximum context" in err_str:
                    current_max_tokens = max(128, int(current_max_tokens * 0.7))
                    logger.warning(f"[{self.api_key[:8]}...] context length hit, reduced max_tokens to {current_max_tokens}")
                raise
        
        return await self._retry(_invoke, max_retries)


class NimClient:
    """Round-robin pool of NimKeyClients."""
    def __init__(self, rate_limit: int = 40, max_retries: int = 5):
        keys = _parse_keys()
        if not keys:
            raise ValueError("No NVIDIA API key found. Set NVIDIA_API_KEY env var.")
        
        cfg = get_config()
        self._clients = [
            NimKeyClient(k, rate_limit, max_retries, cfg.nvidia.model, cfg.nvidia.base_url, cfg.nvidia.max_tokens)
            for k in keys
        ]
        self._pool_size = len(self._clients)
        self._round_robin = 0
        logger.info(f"NimClient pool: {self._pool_size} keys, {rate_limit} req/min each")

    def _pick_client(self) -> NimKeyClient:
        client = self._clients[self._round_robin]
        self._round_robin = (self._round_robin + 1) % self._pool_size
        return client

    async def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None, max_retries: int = 5) -> str:
        return await self._pick_client().chat(messages, None, temperature, max_tokens, max_retries)

    async def chat_structured(self, messages: list[dict], response_model: type[BaseModel], temperature: float | None = None, max_tokens: int | None = None, max_retries: int = 5) -> BaseModel:
        return await self._pick_client().chat(messages, response_model, temperature, max_tokens, max_retries)

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def stats(self) -> dict:
        return {
            f"key_{i}": {"requests": c._request_count, "last": c._last_call}
            for i, c in enumerate(self._clients)
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_nim_client: NimClient | None = None

def get_nim_client(rate_limit: int = 40, max_retries: int = 5) -> NimClient:
    global _nim_client
    if _nim_client is None:
        _nim_client = NimClient(rate_limit=rate_limit, max_retries=max_retries)
    return _nim_client

def reset_nim_client():
    global _nim_client
    _nim_client = None
