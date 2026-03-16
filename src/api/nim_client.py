"""
NIM Client - simple wrapper around ChatOpenAI with NVIDIA NIM.
"""

import asyncio
import time
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "meta/llama-3.3-70b-instruct"


class NimClient:
    """Simple NVIDIA NIM client using LangChain ChatOpenAI."""

    def __init__(
        self,
        model: str = MODEL,
        api_key: str = NVIDIA_API_KEY,
        base_url: str = BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 5,
        rate_limit: int = 40,
    ):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=0,  # Handle retries manually
        )
        self._lock = asyncio.Lock()
        self._min_interval = 60.0 / rate_limit * 1.5  # Conservative: 2/3 of limit
        self._last_call = 0.0
        self._max_retries = max_retries

    def _to_lc_messages(self, messages: list[dict]):
        return [
            SystemMessage(content=m["content"]) if m["role"] == "system"
            else HumanMessage(content=m["content"])
            for m in messages
        ]

    async def _rate_limit(self):
        async with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.time()

    async def chat(self, messages: list[dict]) -> str:
        await self._rate_limit()

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self.llm.ainvoke(self._to_lc_messages(messages))
                return response.content
            except Exception as e:
                last_error = e
                await asyncio.sleep(2 ** attempt)

        raise last_error

    async def chat_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        temperature: float = None,
        max_tokens: int = None,
    ) -> BaseModel:
        await self._rate_limit()

        # Manual retry with exponential backoff and increasing max_tokens
        last_error = None
        current_max_tokens = max_tokens or self.llm.max_tokens

        for attempt in range(self._max_retries + 1):
            try:
                kwargs = {"temperature": temperature} if temperature else {}
                kwargs["max_tokens"] = current_max_tokens

                structured_llm = self.llm.with_structured_output(response_model)
                return await structured_llm.ainvoke(self._to_lc_messages(messages), **kwargs)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "length" in err_str:
                    current_max_tokens = int(current_max_tokens * 1.5)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise last_error


_nim_client = None


def get_nim_client(**kwargs) -> NimClient:
    global _nim_client
    if _nim_client is None:
        _nim_client = NimClient(**kwargs)
    return _nim_client


def reset_nim_client():
    global _nim_client
    _nim_client = None
