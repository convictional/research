from typing import Type
from pydantic import BaseModel, SecretStr
import instructor
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

ASYNC_INSTRUCTOR_CLIENT = None


def set_async_instructor_client(llm_model: str, api_key: SecretStr, openai_organization: str | None = None):
    """
    Set the async instructor client.
    This must be called before calling LLM request functions.
    """
    global ASYNC_INSTRUCTOR_CLIENT
    ASYNC_INSTRUCTOR_CLIENT = _get_async_instructor_client(llm_model, api_key, openai_organization)


def _get_async_instructor_client(llm_model: str, api_key: SecretStr, openai_organization: str | None = None):
    llm_group = _get_llm_model_group(llm_model)
    if llm_group in ["anthropic"]:
        anthropic_client = AsyncAnthropic(api_key=api_key.get_secret_value())
        return instructor.from_anthropic(anthropic_client, mode=instructor.Mode.ANTHROPIC_JSON)
    elif llm_group in ["openai", "openai_o1", "openai_o3_mini"]:
        openai_client = AsyncOpenAI(api_key=api_key.get_secret_value(), organization=openai_organization)
        return instructor.from_openai(openai_client)
    elif llm_group in ["openai_o1_mini"]:
        openai_client = AsyncOpenAI(api_key=api_key.get_secret_value(), organization=openai_organization)
        return instructor.from_openai(openai_client, mode=instructor.Mode.JSON_O1)

    raise Exception(f"Get async instructor client, unknown llm group: {llm_group}")


def _get_llm_model_group(llm_model: str):
    if llm_model.startswith("claude"):
        return "anthropic"
    elif llm_model.startswith("gpt"):
        return "openai"
    elif llm_model.startswith("o1-mini"):
        return "openai_o1_mini"
    elif llm_model.startswith("o1"):
        return "openai_o1"
    elif llm_model.startswith("o3-mini"):
        return "openai_o3_mini"

    raise Exception(f"Get LLM model group, LLM model {llm_model} not supported")


async def ainstruct_llm(
    system_prompt: str | None,
    user_prompt: str,
    response_model: Type[BaseModel],
    llm_model: str,
    reasoning_effort: str = "high",  # used for open ai reasoning models
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

    llm_group = _get_llm_model_group(llm_model)

    if llm_group in ["anthropic"]:
        messages = _get_messages(llm_group, user_prompt)

        response = await ASYNC_INSTRUCTOR_CLIENT.messages.create(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
            response_model=response_model,
        )

        return response
    elif llm_group in ["openai"]:
        messages = _get_messages(llm_group, user_prompt, system_prompt)

        response = await ASYNC_INSTRUCTOR_CLIENT.chat.completions.create(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            response_model=response_model,
        )

        return response
    elif llm_group in ["openai_o1", "openai_o3_mini"]:
        # The API is a bit different for o1 models
        messages = _get_messages(llm_group, user_prompt, system_prompt)

        response = await ASYNC_INSTRUCTOR_CLIENT.chat.completions.create(
            model=llm_model,
            messages=messages,
            response_model=response_model,
            reasoning_effort=reasoning_effort,
        )

        return response
    elif llm_group in ["openai_o1_mini"]:
        # The API is a bit different for o1 models, and o1-mini does not support developer messages
        # Further, o1-mini does not support reasoning_effort
        messages = _get_messages(llm_group, user_prompt)

        response = await ASYNC_INSTRUCTOR_CLIENT.chat.completions.create(
            model=llm_model,
            messages=messages,
            response_model=response_model,
        )

        return response

    raise Exception(f"Instruct LLM, unknown llm group: {llm_group}")


def _get_messages(llm_group: str, user_prompt: str, system_prompt: str = None):
    if llm_group in ["anthropic"]:
        return [{"role": "user", "content": user_prompt}]
    elif llm_group in ["openai", "openai_o1", "openai_o3_mini"]:
        return [{"role": "developer", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    elif llm_group in ["openai_o1_mini"]:
        return [{"role": "user", "content": user_prompt}]

    raise Exception(f"Get messages, unknown llm group: {llm_group}")
