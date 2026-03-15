"""
NIM Client using LangChain with structured output.

Uses ChatOpenAI with base_url pointing to NVIDIA NIM.
"""

import os
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from loguru import logger


@dataclass
class ChatResponse:
    content: str
    reasoning: Optional[str] = None
    usage: dict = None
    finish_reason: str = "stop"


class NimClient:
    """NVIDIA NIM client using LangChain ChatOpenAI."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config=None):
        if self._initialized:
            return

        if config is None:
            from ..config import get_config
            config = get_config()

        self.config = config
        self.nvidia = config.nvidia

        # LangChain ChatOpenAI pointing to NVIDIA NIM
        self.llm = ChatOpenAI(
            model=self.nvidia.model,
            api_key=self.nvidia.api_key,
            base_url=self.nvidia.base_url,
            temperature=self.nvidia.temperature,
            max_tokens=self.nvidia.max_tokens,
        )

        # Rate limiting
        self._lock = asyncio.Lock()
        self._min_interval = 60.0 / self.nvidia.rate_limit
        self._last_call = 0.0
        self._initialized = True

    async def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Basic chat - returns raw response."""
        # Rate limiting
        async with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.time()

        # Convert messages to LangChain format
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            else:
                lc_messages.append(HumanMessage(content=msg["content"]))

        try:
            response = await self.llm.ainvoke(
                lc_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return ChatResponse(
                content=response.content,
                usage={},
                finish_reason="stop",
            )
        except Exception as e:
            logger.error(f"NIM API error: {e}")
            raise

    async def chat_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        """Chat with structured output - uses LangChain with_structured_output."""
        # Rate limiting
        async with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.time()

        # Convert messages to LangChain format
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            else:
                lc_messages.append(HumanMessage(content=msg["content"]))

        # Create structured LLM
        structured_llm = self.llm.with_structured_output(response_model)

        try:
            result = await structured_llm.ainvoke(
                lc_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result
        except Exception as e:
            logger.error(f"Structured output failed: {e}")
            raise


# Singleton accessor
_nim_client = None


def get_nim_client(config=None) -> NimClient:
    global _nim_client
    if _nim_client is None:
        _nim_client = NimClient(config)
    return _nim_client


def reset_nim_client():
    """Reset client (useful for testing)."""
    global _nim_client
    _nim_client = None
