from typing import Type, Tuple
from anthropic import APIError, InternalServerError
import asyncio
import time

from pydantic import BaseModel

import instructor

from .config.experiment_settings import settings


def is_anthropic_server_error(exception):
    return isinstance(exception, (InternalServerError, APIError)) and getattr(exception, "status_code", 500) >= 500


def get_instructor_client():
    if settings.llm_model.startswith("gpt"):
        from openai import OpenAI

        openai_client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            organization=settings.openai_organization,
        )
        return instructor.from_openai(openai_client)
    elif settings.llm_model.startswith("claude"):
        from anthropic import Anthropic

        anthropic_client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        # TODO: Revert our hard-coding of the mode once bug is resolved: https://github.com/jxnl/instructor/issues/774
        return instructor.from_anthropic(anthropic_client, mode=instructor.mode.Mode.ANTHROPIC_JSON)

    raise Exception("Unknown llm model")


def get_async_instructor_client():
    if settings.llm_model.startswith("gpt"):
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            organization=settings.openai_organization,
        )
        return instructor.from_openai(openai_client)
    elif settings.llm_model.startswith("claude"):
        from anthropic import AsyncAnthropic

        anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        return instructor.from_anthropic(anthropic_client)

    raise Exception("Unknown llm model")


INSTRUCTOR_CLIENT = get_instructor_client()
ASYNC_INSTRUCTOR_CLIENT = get_async_instructor_client()


def get_messages(system_prompt, user_prompt, few_shot):
    if few_shot:
        return [
            {"role": "user", "content": system_prompt},
            *few_shot,
            {"role": "user", "content": user_prompt},
        ]
    return [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def get_num_input_tokens(compeletion):
    if settings.llm_model.startswith("claude"):
        return compeletion.usage.input_tokens
    return compeletion.usage.prompt_tokens


def get_num_output_tokens(compeletion):
    if settings.llm_model.startswith("claude"):
        return compeletion.usage.output_tokens
    return compeletion.usage.completion_tokens


def _retry_once(func):
    try:
        return func()
    except (APIError, InternalServerError):
        # Wait a bit before retry
        time.sleep(4)
        return func()


async def _async_retry_once(func):
    try:
        return await func()
    except (APIError, InternalServerError):
        # Wait a bit before retry
        await asyncio.sleep(4)
        return await func()


def instruct_llm(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[BaseModel],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    few_shot: list[dict] | None = None,
    llm_model: str = settings.llm_model,
) -> Tuple[Type[BaseModel], dict]:
    messages = get_messages(system_prompt, user_prompt, few_shot)
    client = get_instructor_client()

    def execute_completion():
        return client.chat.completions.create_with_completion(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            response_model=response_model,
        )

    response, completion = _retry_once(execute_completion)

    completion_data = {
        "completion": completion,
        "usage": {
            "input_tokens": get_num_input_tokens(completion),
            "output_tokens": get_num_output_tokens(completion),
        },
        "model": completion.model,
    }

    return response, completion_data


async def ainstruct_llm(
    response_model: Type[BaseModel],
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    few_shot: list[dict] | None = None,
    llm_model: str = settings.llm_model,
    messages: list[dict] | None = None,
) -> Tuple[Type[BaseModel], dict]:
    if messages is None:
        messages = get_messages(system_prompt, user_prompt, few_shot)

    async def execute_completion():
        return await ASYNC_INSTRUCTOR_CLIENT.chat.completions.create_with_completion(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            response_model=response_model,
        )

    response, completion = await _async_retry_once(execute_completion)

    completion_data = {
        "completion": completion,
        "usage": {
            "input_tokens": get_num_input_tokens(completion),
            "output_tokens": get_num_output_tokens(completion),
        },
        "model": completion.model,
    }

    return response, completion_data
