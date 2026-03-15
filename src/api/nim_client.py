"""
NIM Client - simple wrapper around ChatOpenAI with NVIDIA NIM.
"""

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
    ):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    def _to_lc_messages(self, messages: list[dict]):
        return [
            SystemMessage(content=m["content"]) if m["role"] == "system"
            else HumanMessage(content=m["content"])
            for m in messages
        ]

    async def chat(self, messages: list[dict]) -> str:
        response = await self.llm.ainvoke(self._to_lc_messages(messages))
        return response.content

    async def chat_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        temperature: float = None,
        max_tokens: int = None,
    ) -> BaseModel:
        kwargs = {"temperature": temperature} if temperature else {}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        structured_llm = self.llm.with_structured_output(response_model)
        return await structured_llm.ainvoke(self._to_lc_messages(messages), **kwargs)


_nim_client = None


def get_nim_client(**kwargs) -> NimClient:
    global _nim_client
    if _nim_client is None:
        _nim_client = NimClient(**kwargs)
    return _nim_client


def reset_nim_client():
    global _nim_client
    _nim_client = None
