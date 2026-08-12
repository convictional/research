"""Local wrapper around common.ainstruct_llm.

Three concerns handled here that the shared helper doesn't:
1. Drops `temperature` for models that reject it (e.g., claude-opus-4-7).
2. Rebinds the instructor response to the caller's canonical Pydantic class —
   instructor returns an instance whose `type()` has the same module/name but
   is a different class object, which breaks stdlib pickle.
3. Retries transient API failures (529 overloaded, 429 rate-limit, 5xx,
   timeouts, connection drops) with exponential backoff so a momentary provider
   blip does not abort a multi-hour run. instructor's built-in retry fires 3×
   within ~1s with no real backoff, which cannot outlast an overload spike.
"""
import asyncio
import random
from typing import Awaitable, Callable, Type, TypeVar

from pydantic import BaseModel

from common import instruct_llm as ci
from common.instruct_llm import set_async_instructor_client  # noqa: F401 — re-exported for callers

from .settings import logger

_NO_TEMPERATURE_MODELS = {"claude-opus-4-7"}

# ── Transient-error retry policy ──────────────────────────────────────────────

_MAX_ATTEMPTS = 8
_BASE_DELAY_SECONDS = 2.0
_MAX_DELAY_SECONDS = 60.0
# HTTP statuses worth retrying: timeouts/conflicts, rate limits, and 5xx
# (529 = Anthropic "overloaded_error", non-standard but >= 500).
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 529}
# Lowercased substrings that mark a transient failure. instructor wraps the real
# error in InstructorRetryException whose message embeds the full nested text,
# so a string check is a robust SDK-agnostic fallback.
_TRANSIENT_MARKERS = ("overloaded", "rate limit", "rate_limit", "timeout", "connection error")

_T = TypeVar("_T")


def _iter_exception_chain(exc: BaseException):
    """Yield the exception and every exception linked via __cause__/__context__."""
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        stack.append(cur.__cause__)
        stack.append(cur.__context__)


def _is_transient(exc: BaseException) -> bool:
    """True if `exc` (or any error it wraps) is a retryable transient API failure.

    SDK-agnostic: checks HTTP status codes and exception-type names across the
    cause chain so it covers anthropic.* and openai.* (and instructor's wrapper)
    without importing provider-specific classes. Non-transient errors
    (BadRequestError, AuthenticationError, ValidationError, ...) return False so
    they fail fast.
    """
    transient_type_names = {
        "InternalServerError",
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "Timeout",
        "OverloadedError",
        "ServiceUnavailableError",
    }
    for err in _iter_exception_chain(exc):
        status = getattr(err, "status_code", None)
        if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
            return True
        if type(err).__name__ in transient_type_names:
            return True
        message = str(err).lower()
        if any(marker in message for marker in _TRANSIENT_MARKERS):
            return True
    return False


async def with_transient_retry(call: Callable[[], Awaitable[_T]], *, label: str) -> _T:
    """Await `call()`, retrying transient API failures with exponential backoff.

    `call` must be a zero-arg coroutine factory (a thunk) — it is re-invoked on
    each attempt so a fresh request is issued. Non-transient errors propagate
    immediately; transient ones retry up to `_MAX_ATTEMPTS` times, after which
    the last exception is re-raised.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 — re-raised below unless transient
            if attempt >= _MAX_ATTEMPTS or not _is_transient(exc):
                raise
            delay = min(_BASE_DELAY_SECONDS * 2 ** (attempt - 1), _MAX_DELAY_SECONDS)
            delay += random.uniform(0, delay * 0.25)  # jitter to de-sync concurrent calls
            logger.warning(
                "transient API error on %s (attempt %d/%d), retrying in %.1fs: %s",
                label, attempt, _MAX_ATTEMPTS, delay, exc.__class__.__name__,
            )
            await asyncio.sleep(delay)


def model_supports_temperature(llm_model: str) -> bool:
    """Whether a requested temperature is actually honored for this model.

    Some models (e.g. claude-opus-4-7) reject the temperature parameter and run at
    their API default regardless of what is requested. Used for reporting which
    judge models honor the configured temperature.
    """
    return llm_model not in _NO_TEMPERATURE_MODELS


async def ainstruct_llm(
    system_prompt: str | None,
    user_prompt: str,
    response_model: Type[BaseModel],
    llm_model: str,
    reasoning_effort: str = "high",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Type[BaseModel]:
    if llm_model not in _NO_TEMPERATURE_MODELS:
        async def _call():
            return await ci.ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                llm_model=llm_model,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raw = await with_transient_retry(_call, label=llm_model)
    else:
        if ci.ASYNC_INSTRUCTOR_CLIENT is None:
            raise ValueError(
                "ASYNC_INSTRUCTOR_CLIENT not set. Call set_async_instructor_client before calling ainstruct_llm."
            )

        async def _call():
            return await ci.ASYNC_INSTRUCTOR_CLIENT.messages.create(
                model=llm_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                response_model=response_model,
            )
        raw = await with_transient_retry(_call, label=llm_model)

    if type(raw) is response_model:
        return raw
    return response_model.model_validate(raw.model_dump())
