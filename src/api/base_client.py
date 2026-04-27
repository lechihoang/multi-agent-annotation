"""
Base LLM client primitives inspired by ReEvo style:
- centralized retry loop
- categorized transient errors
- exponential backoff with jitter
"""

from __future__ import annotations

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

    @staticmethod
    def _is_transient_error(err_str: str) -> bool:
        return any(
            token in err_str
            for token in (
                "429",
                "rate limit",
                "too many requests",
                "504",
                "gateway timeout",
                "timeout",
                "temporarily unavailable",
                "connection",
            )
        )

    async def _retry(self, fn: Callable[[], Awaitable[T]], max_retries: int | None = None) -> T:
        last_error: Exception | None = None
        retries = self.max_retries if max_retries is None else max_retries
        for attempt in range(retries + 1):
            try:
                return await fn()
            except Exception as e:  # noqa: BLE001
                last_error = e
                err = str(e).lower()
                transient = self._is_transient_error(err)
                is_last = attempt >= retries

                if not transient and "length" not in err and "maximum context" not in err:
                    raise

                if is_last:
                    break

                wait_s = self.base_backoff_s * (2**attempt) + random() * 0.25
                logger.warning(
                    "LLM call failed (attempt=%s/%s, transient=%s): %s; retry in %.2fs",
                    attempt + 1,
                    retries + 1,
                    transient,
                    e,
                    wait_s,
                )
                await asyncio.sleep(wait_s)

        assert last_error is not None
        raise last_error
