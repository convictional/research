import asyncio
from pathlib import Path

from pydantic import BaseModel

from ..instruct_helper import ainstruct_llm, set_async_instructor_client
from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit
from common.prompt_template_engine import build_prompt

from ..models import ActivityEvent, CandidateGoal, Decision
from ..settings import logger, settings

CACHE_FILENAME = "step1_candidates.pkl"


class _Step1Response(BaseModel):
    candidate_goals: list[CandidateGoal]


def _format_events(events: list[ActivityEvent], max_events: int, max_body: int = 300) -> str:
    sample = events[:max_events]
    lines = []
    for e in sample:
        body = e.body[:max_body] if len(e.body) > max_body else e.body
        lines.append(f"[{e.event_id}] ({e.event_type}) {e.title}\n  Body: {body}")
    return "\n\n".join(lines)


def _format_decisions(decisions: list[Decision], max_decisions: int = 40) -> str:
    sample = decisions[:max_decisions]
    lines = []
    for d in sample:
        opts = "; ".join(o["title"] for o in d.options[:3]) or "—"
        goals_text = d.author_stated_goals or "(none stated)"
        lines.append(
            f"[{d.id}] {d.title}\n"
            f"  Author goals: {goals_text[:200]}\n"
            f"  Options: {opts}"
        )
    return "\n\n".join(lines)


async def run_step1(
    activity_events: list[ActivityEvent],
    decisions: list[Decision],
    output_path: Path,
    load_from_cache: bool = True,
) -> list[CandidateGoal]:
    """Extract candidate unstated goals from the activity and decision corpus."""
    cache_path = output_path / CACHE_FILENAME

    if load_from_cache and cache_path.exists():
        log_cache_hit(cache_path)
        return load_pickle_file(cache_path)

    print(f"  Step 1: extracting unstated goals with {settings.step1_model}...")
    set_async_instructor_client(settings.step1_model, settings.anthropic_api_key)

    events_formatted = _format_events(activity_events, settings.max_activity_events_for_extraction)
    decisions_formatted = _format_decisions(decisions)

    system_prompt = build_prompt("step1_unstated_extraction_system.txt.jinja")
    user_prompt = build_prompt(
        "step1_unstated_extraction_user.txt.jinja",
        num_goals=settings.num_unstated_goals_to_extract,
        events_formatted=events_formatted,
        events_count=min(len(activity_events), settings.max_activity_events_for_extraction),
        decisions_formatted=decisions_formatted,
        decisions_count=min(len(decisions), 40),
    )

    response: _Step1Response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=_Step1Response,
        llm_model=settings.step1_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    candidates = response.candidate_goals
    dump_to_pickle_file(candidates, cache_path)
    print(f"  Step 1: extracted {len(candidates)} candidate goals → {cache_path}")
    return candidates
