from typing import Type
from pydantic import BaseModel
import instructor
from anthropic import AsyncAnthropic

from ..settings import settings


def get_async_instructor_client():
    if settings.llm_model.startswith("claude"):
        anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        return instructor.from_anthropic(anthropic_client)

    raise Exception("Unknown llm model")


ASYNC_INSTRUCTOR_CLIENT = get_async_instructor_client()


def get_messages_new(user_prompt):
    return [{"role": "user", "content": user_prompt}]


async def ainstruct_llm(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[BaseModel],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    llm_model: str = settings.llm_model,
) -> Type[BaseModel]:
    messages = get_messages_new(user_prompt)

    response = await ASYNC_INSTRUCTOR_CLIENT.messages.create(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
        response_model=response_model,
    )

    return response
