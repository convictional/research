import pandas as pd
import json
import asyncio

from .llm import allm_request, set_async_client
from common.async_helper import limited_task, execute_tasks_with_manual_pbar


def get_request_bodies_and_remove_goals_from_system_prompt(data_df: pd.DataFrame) -> list[dict]:
    """
    Get request bodies from the given data DataFrame and remove goals from the system prompts,
    in those request bodies.

    Return those parsed request bodies without goals.
    """
    # Get request bodies
    print("Getting request bodies...")
    request_bodies: list[str] = list(data_df["request_body"])
    parsed_request_bodies_with_goals: list[dict] = [json.loads(x) for x in request_bodies]

    # Remove goals from system prompts
    parsed_request_bodies_without_goals = _remove_goals_from_system_prompt_of_requests(
        parsed_request_bodies_with_goals
    )

    return parsed_request_bodies_without_goals


def _remove_goals_from_system_prompt_of_requests(request_bodies: list[dict]) -> list[dict]:
    """
    Removes/strips goals from the system prompts of the given request bodies.
    """
    print("Removing goals from system prompts...")
    for request_body in request_bodies:
        system_text = request_body["system"]
        # Find "Goals of the organization:" and "</organization_context>"
        goals_start = system_text.find("Goals of the organization:")
        org_context_end = system_text.find("</organization_context>")

        if goals_start != -1 and org_context_end != -1:
            # Keep everything before "Goals of the organization:" and after "</organization_context>"
            new_system_text = (
                system_text[:goals_start]
                + "</organization_context>"
                + system_text[org_context_end + len("</organization_context>") :]
            )
            request_body["system"] = new_system_text

    return request_bodies


async def make_requests_to_llm_using_request_bodies(
    request_bodies: list[dict],
    temperature: float | None = None,
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> list[dict]:
    """
    Make requests to the LLM model, given a list of request bodies.
    """
    print("Making requests to the LLM model...")
    set_async_client()

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = [
        limited_task(
            allm_request(request_body, temperature=temperature),
            semaphore,
            delay_between_tasks,
        )
        for request_body in request_bodies
    ]

    raw_responses = await execute_tasks_with_manual_pbar(tasks)
    responses: list[dict] = [response.model_dump() for response in raw_responses]

    return responses
