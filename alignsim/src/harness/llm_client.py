"""Self-contained Anthropic instructor client for condition 1.

Uses create_with_completion to return both the parsed Pydantic model
and the raw completion (which carries token usage metadata).
"""

from dataclasses import dataclass
from typing import Type

import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel, SecretStr


@dataclass
class LLMResult:
    """Parsed response plus token usage from a single API call."""

    response: BaseModel
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


_CLIENT: instructor.AsyncInstructor | None = None


def init_client(api_key: SecretStr) -> None:
    global _CLIENT
    anthropic = AsyncAnthropic(api_key=api_key.get_secret_value())
    _CLIENT = instructor.from_anthropic(anthropic, mode=instructor.Mode.ANTHROPIC_JSON)


async def instruct(
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[BaseModel],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> LLMResult:
    if _CLIENT is None:
        raise RuntimeError("Call init_client() before instruct()")

    response, completion = await _CLIENT.messages.create_with_completion(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
        response_model=response_model,
    )

    usage = completion.usage
    return LLMResult(
        response=response,
        input_tokens=getattr(usage, "input_tokens", 0),
        output_tokens=getattr(usage, "output_tokens", 0),
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0),
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0),
    )
