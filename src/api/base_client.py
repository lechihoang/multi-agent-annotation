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
        
        server_attempts = 0
        validation_attempts = 0
        
        # Initial jitter (Reevo style)
        await asyncio.sleep(random())
        
        while True:
            try:
                return await fn()
            except Exception as e:  # noqa: BLE001
                last_error = e
                err_str = str(e).lower()

                if "length" in err_str or "maximum context" in err_str:
                    pass # Allow nim_client's reduced max_tokens to be retried

                is_validation = "validation" in err_str or "parse" in err_str
                
                if is_validation:
                    validation_attempts += 1
                    if validation_attempts > 5:  # Cap validation retries at 5 to prevent infinite loops
                        break
                    wait_s = random() * 0.1
                    logger.warning(
                        f"LLM call failed (validation error, attempt={validation_attempts}/5): {e}; retry in {wait_s:.2f}s"
                    )
                else:
                    server_attempts += 1
                    if server_attempts > retries:
                        break
                    wait_s = self.base_backoff_s * (2**(server_attempts - 1)) + random() * 0.25
                    logger.warning(
                        f"LLM call failed (server error, attempt={server_attempts}/{retries}): {e}; retry in {wait_s:.2f}s"
                    )
                    
                await asyncio.sleep(wait_s)

        assert last_error is not None
        raise last_error
