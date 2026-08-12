"""Claude API client for streaming responses with metrics collection."""

import time
from datetime import UTC, datetime

from anthropic import AsyncAnthropic
from anthropic.types import ContentBlockDeltaEvent, MessageStopEvent


async def stream_claude_response_with_metrics(
    api_key: str,
    request_body: dict,
    timeout_seconds: float = 60.0,
) -> tuple[str, dict]:
    """
    Stream response from Claude API with detailed metrics collection.

    Args:
        api_key: Anthropic API key
        request_body: Request body in Anthropic format
        timeout_seconds: Request timeout in seconds

    Returns:
        Tuple of (response_text, metrics_dict) where metrics_dict contains:
        - start_time: datetime when request started
        - end_time: datetime when request completed
        - ttft: time to first token in seconds
        - total_tokens: total output tokens from usage metadata
        - error_type: error classification if request failed
        - error_message: error details if request failed
    """
    metrics: dict = {
        "start_time": datetime.now(UTC),
        "end_time": None,
        "ttft": None,
        "total_tokens": 0,
        "error_type": None,
        "error_message": None,
    }

    response_text = ""
    first_token_received = False

    try:
        client = AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)

        async with client.messages.stream(**request_body) as stream:
            async for event in stream:
                if isinstance(event, ContentBlockDeltaEvent):
                    if hasattr(event.delta, "text"):
                        text = event.delta.text
                        if text:
                            if not first_token_received:
                                metrics["ttft"] = time.time() - metrics["start_time"].timestamp()
                                first_token_received = True
                            response_text += text
                elif isinstance(event, MessageStopEvent):
                    pass

            final_message = await stream.get_final_message()
            if final_message.usage:
                metrics["total_tokens"] = final_message.usage.output_tokens

    except Exception as e:
        import anthropic

        if isinstance(e, anthropic.APITimeoutError):
            metrics["error_type"] = "TimeoutError"
            metrics["error_message"] = str(e)
        elif isinstance(e, anthropic.APIConnectionError):
            metrics["error_type"] = "ConnectionError"
            metrics["error_message"] = str(e)
        elif isinstance(e, anthropic.RateLimitError):
            metrics["error_type"] = "RateLimitError"
            metrics["error_message"] = str(e)
        elif isinstance(e, anthropic.APIStatusError):
            metrics["error_type"] = f"HTTPError_{e.status_code}"
            metrics["error_message"] = f"HTTP {e.status_code}: {e.message}"
        else:
            metrics["error_type"] = f"UnknownError_{type(e).__name__}"
            metrics["error_message"] = str(e)
    finally:
        metrics["end_time"] = datetime.now(UTC)

    return response_text, metrics
