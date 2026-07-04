"""
Base LLM client primitives inspired by ReEvo style:
- centralized retry loop
- exponential backoff with jitter
"""

import asyncio
import logging
from random import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseLLMClient:
    """Base class that provides shared async retry behavior."""

    def __init__(self, max_retries: int = 5, base_backoff_s: float = 1.5):
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s

    async def _retry(self, fn: Callable[[], Awaitable[T]], max_retries: int | None = None) -> T:
        last_error: Exception | None = None
        retries = self.max_retries if max_retries is None else max_retries
        
        # Initial jitter (Reevo style)
        await asyncio.sleep(random())
        
        for attempt in range(retries + 1):
            try:
                return await fn()
            except Exception as e:  # noqa: BLE001
                last_error = e
                err_str = str(e).lower()
                is_last = attempt >= retries

                if "length" in err_str or "maximum context" in err_str:
                    pass # Allow nim_client's reduced max_tokens to be retried

                if is_last:
                    break

                wait_s = self.base_backoff_s * (2**attempt) + random() * 0.25
                logger.warning(
                    f"LLM call failed (attempt={attempt + 1}/{retries + 1}): {e}; retry in {wait_s:.2f}s"
                )
                await asyncio.sleep(wait_s)

        assert last_error is not None
        raise last_error
