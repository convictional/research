from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from geo_analyzer.providers.base import ProviderError
from geo_analyzer.runner.retry import retry_with_backoff


class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self) -> None:
        called = AsyncMock(return_value="ok")
        result = await retry_with_backoff(called, max_attempts=3, backoff_base_s=0.0)
        assert result == "ok"
        assert called.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_provider_error(self) -> None:
        attempts: list[int] = []

        async def flaky() -> str:
            attempts.append(len(attempts))
            if len(attempts) < 3:
                raise ProviderError("transient")
            return "ok"

        with patch("geo_analyzer.runner.retry.asyncio.sleep", new=AsyncMock()):
            result = await retry_with_backoff(flaky, max_attempts=3, backoff_base_s=1.0)
        assert result == "ok"
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self) -> None:
        async def always_fails() -> str:
            raise ProviderError("permanent")

        mock_sleep = AsyncMock()
        with (
            patch("geo_analyzer.runner.retry.asyncio.sleep", new=mock_sleep),
            pytest.raises(ProviderError, match="permanent"),
        ):
            await retry_with_backoff(always_fails, max_attempts=3, backoff_base_s=0.0)

    @pytest.mark.asyncio
    async def test_does_not_retry_unrelated_exceptions(self) -> None:
        async def raises_value_error() -> str:
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await retry_with_backoff(raises_value_error, max_attempts=3, backoff_base_s=0.0)

    @pytest.mark.asyncio
    async def test_backoff_grows_exponentially(self) -> None:
        sleep_calls: list[float] = []

        async def fail_twice_then_ok() -> str:
            if len(sleep_calls) < 2:
                raise ProviderError("transient")
            return "ok"

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch("geo_analyzer.runner.retry.asyncio.sleep", new=fake_sleep):
            await retry_with_backoff(fail_twice_then_ok, max_attempts=4, backoff_base_s=2.0)

        # Backoff: base * 2^(attempt-1) → 2.0, 4.0
        assert sleep_calls == [2.0, 4.0]
