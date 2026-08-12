from typing import Type
from pydantic import BaseModel, SecretStr
import instructor
from anthropic import AsyncAnthropic


ASYNC_INSTRUCTOR_CLIENT = None


def set_async_instructor_client(api_key: SecretStr):
    """
    Set the async instructor client.
    This must be called before calling LLM request functions.
    """
    global ASYNC_INSTRUCTOR_CLIENT
    anthropic_client = AsyncAnthropic(api_key=api_key.get_secret_value())
    ASYNC_INSTRUCTOR_CLIENT = instructor.from_anthropic(anthropic_client)


async def ainstruct_llm(
    system_prompt: str,
    messages: list[dict[str, str]],
    response_model: Type[BaseModel] | None,
    llm_model: str,
    tools: list[dict] = [],
    thinking: dict = {"type": "disabled"},
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Type[BaseModel]:
    """
    This function sends a request to the LLM model.
    Note, this function must be called after set_async_instructor_client.
    """
    if not ASYNC_INSTRUCTOR_CLIENT:
        raise ValueError(
            "ASYNC_INSTRUCTOR_CLIENT not set. Call set_async_instructor_client before calling this function."
        )

    response = await ASYNC_INSTRUCTOR_CLIENT.messages.create(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
        response_model=response_model,
        tools=tools,
        thinking=thinking,
    )

    return response
