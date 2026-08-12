from anthropic import AsyncAnthropic, NOT_GIVEN

from .settings import settings


ASYNC_CLIENT = None


def set_async_client():
    """
    Set the async client.
    This must be called before calling LLM request functions.
    """
    global ASYNC_CLIENT
    ASYNC_CLIENT = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())


async def allm_request(request_body: dict, temperature: float | None = None) -> dict:
    """
    Make a request to the LLM model, given a request body.

    We use a custom function here, rather than what is in common since we are not using instructor
    and are just passing the whole request body (already constructed) to the LLM model.
    """
    if not ASYNC_CLIENT:
        raise ValueError("ASYNC_CLIENT not set. Call set_async_client before calling this function.")

    response = await ASYNC_CLIENT.messages.create(temperature=temperature or NOT_GIVEN, **request_body)
    return response
