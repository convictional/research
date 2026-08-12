import json
import time
from datetime import UTC, datetime

import dotenv
import httpx
import os
from pathlib import Path

from src.llm.token_counter import count_tokens

dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env.secrets")

ENDPOINT = "aiplatform.googleapis.com"
REGION = "global"
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


async def stream_gpt_oss_response(
    headers: dict,
    payload: dict,
) -> str:
    """
    Stream response from GPT-OSS API.

    Args:
        payload: Request payload
        headers: Request headers

    Returns:
        Full response text
    """
    url = f"https://{ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"

    response_text = ""

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[len("data: ") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content", "")
                        if content:
                            response_text += content
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")

    return response_text


async def stream_gpt_oss_response_with_metrics(
    headers: dict,
    payload: dict,
    timeout_seconds: float = 60.0,
) -> tuple[str, dict]:
    """
    Stream response from GPT-OSS API with detailed metrics collection.

    Args:
        headers: Request headers
        payload: Request payload
        timeout_seconds: Request timeout in seconds

    Returns:
        Tuple of (response_text, metrics_dict) where metrics_dict contains:
        - start_time: datetime when request started
        - end_time: datetime when request completed
        - ttft: time to first token in seconds
        - total_tokens: actual token count using tiktoken
        - error_type: error classification if request failed
        - error_message: error details if request failed
    """
    url = f"https://{ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"

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
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    error_body = await response.aread()
                    metrics["error_type"] = f"HTTPError_{response.status_code}"
                    metrics["error_message"] = f"HTTP {response.status_code}: {error_body.decode('utf-8')}"
                    return response_text, metrics
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[len("data: ") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                if not first_token_received:
                                    metrics["ttft"] = time.time() - metrics["start_time"].timestamp()
                                    first_token_received = True
                                response_text += content
                        except Exception as e:
                            print(f"Error parsing chunk: {e}")
    except httpx.HTTPStatusError as e:
        metrics["error_type"] = f"HTTPError_{e.response.status_code}"
        try:
            error_body = e.response.text
            metrics["error_message"] = f"{str(e)}\nResponse: {error_body}"
        except Exception:
            metrics["error_message"] = str(e)
    except httpx.TimeoutException as e:
        metrics["error_type"] = "TimeoutError"
        metrics["error_message"] = str(e)
    except httpx.ConnectError as e:
        metrics["error_type"] = "ConnectionError"
        metrics["error_message"] = str(e)
    except json.JSONDecodeError as e:
        metrics["error_type"] = "ParseError"
        metrics["error_message"] = str(e)
    except Exception as e:
        metrics["error_type"] = f"UnknownError_{type(e).__name__}"
        metrics["error_message"] = str(e)
    finally:
        metrics["end_time"] = datetime.now(UTC)
        if response_text and metrics["error_type"] is None:
            metrics["total_tokens"] = count_tokens(response_text)

    return response_text, metrics
