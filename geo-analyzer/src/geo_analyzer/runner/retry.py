"""Exponential-backoff retry for provider calls.

Only retries `ProviderError` (which adapters raise on 429/5xx/timeout/network).
Non-provider exceptions (programming errors) propagate immediately.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from geo_analyzer.providers.base import ProviderError

_T = TypeVar("_T")


async def retry_with_backoff(
    fn: Callable[[], Awaitable[_T]],
    *,
    max_attempts: int,
    backoff_base_s: float,
) -> _T:
    """Call `fn()` up to `max_attempts` times. Sleeps backoff_base_s * 2^(attempt-1)
    between failures. Re-raises the last ProviderError if all attempts fail.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last: ProviderError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except ProviderError as e:
            last = e
            if attempt == max_attempts:
                break
            delay = backoff_base_s * (2 ** (attempt - 1))
            await asyncio.sleep(delay)
    assert last is not None
    raise last
