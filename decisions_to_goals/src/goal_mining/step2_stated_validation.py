import asyncio
from pathlib import Path

from common.async_helper import execute_tasks_with_manual_pbar, limited_task
from ..instruct_helper import ainstruct_llm, set_async_instructor_client
from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit
from common.prompt_template_engine import build_prompt

from ..models import ActivityEvent, Decision, StatedGoal, StatedGoalEvidence
from ..settings import logger, settings

CACHE_FILENAME = "step2_evidence.pkl"

NO_STATED_GOALS_NOOP_MESSAGE = (
    "No stated goals: step 2 is a no-op for this condition — "
    "no stated goal validation to run."
)


def _format_events(events: list[ActivityEvent], max_events: int = 100, max_body: int = 250) -> str:
    sample = events[:max_events]
    lines = []
    for e in sample:
        body = e.body[:max_body] if len(e.body) > max_body else e.body
        lines.append(f"[{e.event_id}] ({e.event_type}) {e.title}: {body}")
    return "\n\n".join(lines)


async def _validate_one_goal(
    goal: StatedGoal,
    events_formatted: str,
) -> StatedGoalEvidence:
    system_prompt = build_prompt("step2_stated_validation_system.txt.jinja")
    user_prompt = build_prompt(
        "step2_stated_validation_user.txt.jinja",
        goal=goal,
        events_formatted=events_formatted,
    )

    response: StatedGoalEvidence = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=StatedGoalEvidence,
        llm_model=settings.step2_model,
        temperature=settings.temperature,
        max_tokens=4096,
    )
    # Ensure goal_id is always the canonical ID from input
    return response.model_copy(update={"goal_id": goal.id})


async def run_step2(
    stated_goals: list[StatedGoal],
    activity_events: list[ActivityEvent],
    decisions: list[Decision],
    output_path: Path,
    load_from_cache: bool = True,
) -> tuple[list[StatedGoalEvidence], dict]:
    """Validate each stated goal against activity evidence.

    Returns (evidence_list, step2_notes). When stated_goals == [] (the 'unstated'
    condition), this is a documented no-op: returns empty list immediately without
    any LLM calls.
    """
    cache_path = output_path / CACHE_FILENAME

    if load_from_cache and cache_path.exists():
        log_cache_hit(cache_path)
        evidence = load_pickle_file(cache_path)
        step2_notes: dict = {}
        if not stated_goals:
            step2_notes = {"step2_noop": True, "step2_noop_reason": NO_STATED_GOALS_NOOP_MESSAGE}
        return evidence, step2_notes

    # No stated goals — documented no-op
    if not stated_goals:
        logger.info(NO_STATED_GOALS_NOOP_MESSAGE)
        print(f"  Step 2: {NO_STATED_GOALS_NOOP_MESSAGE}")
        evidence: list[StatedGoalEvidence] = []
        dump_to_pickle_file(evidence, cache_path)
        step2_notes = {"step2_noop": True, "step2_noop_reason": NO_STATED_GOALS_NOOP_MESSAGE}
        return evidence, step2_notes

    print(f"  Step 2: validating {len(stated_goals)} stated goals with {settings.step2_model}...")
    set_async_instructor_client(settings.step2_model, settings.anthropic_api_key)

    events_formatted = _format_events(activity_events)

    semaphore = asyncio.Semaphore(settings.max_concurrency)
    tasks = [
        limited_task(
            _validate_one_goal(goal, events_formatted),
            semaphore,
            settings.delay_between_tasks,
        )
        for goal in stated_goals
    ]

    results = await execute_tasks_with_manual_pbar(tasks)

    dump_to_pickle_file(results, cache_path)
    print(f"  Step 2: validated {len(results)} goals → {cache_path}")
    return results, {}
