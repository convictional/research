from anthropic import AsyncAnthropic

from ..settings import settings


def get_async_anthropic_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value(), timeout=900)


ASYNC_ANTHROPIC_CLIENT = get_async_anthropic_client()


async def astring_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 8000,
    llm_model: str = settings.llm_model,
) -> str:
    response = await ASYNC_ANTHROPIC_CLIENT.messages.create(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.content[0].text
