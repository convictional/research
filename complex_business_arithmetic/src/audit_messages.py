import time

from .models import AuditorResponse
from .settings import settings
from common.prompt_template_engine import build_prompt
from .instruct_llm import ainstruct_llm


LLM_TEMPERATURE = 0.0


async def audit_messages_output(messages: list[dict]) -> AuditorResponse:
    """
    Audit messages using another LLM call
    """
    print(f"Auditing {len(messages)} messages using an LLM...")

    system_prompt = build_prompt("audit_system.txt.jinja")
    user_prompt = build_prompt("audit_user.txt.jinja", messages=messages)

    messages = [{"role": "user", "content": user_prompt}]

    print("STATUS: Sending request to LLM...")

    request_start_time = time.time()

    response: AuditorResponse = await ainstruct_llm(
        system_prompt=system_prompt,
        messages=messages,
        response_model=AuditorResponse,
        llm_model=settings.llm_model,
        temperature=LLM_TEMPERATURE,
    )

    request_end_time = time.time()

    response.request_duration_s = request_end_time - request_start_time

    print(f"STATUS: Audit took {response.request_duration_s:.2f} seconds")

    return response
