"""DM mapper — each decision maps to at most one goal.

NOTE: No retrieval pre-filter is applied here. Every decision is analyzed against
the full canonical goal set without keyword/entity/cosine pre-screening.
This is an intentional Phase-1 design constraint to isolate the schema-axis
comparison. The keyword + entity + cosine pre-filter from
linking_tasks_to_goals/approach_10 is a candidate for a Phase-2 follow-up
experiment if DM shows noise from irrelevant goal pairs.
"""
import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from common.async_helper import execute_tasks_with_manual_pbar, limited_task
from ..instruct_helper import ainstruct_llm, set_async_instructor_client
from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit
from common.prompt_template_engine import build_prompt

from ..models import CanonicalGoal, Decision
from ..settings import logger, settings
from .analysis import (
    format_decision_for_prompt,
    format_goals_for_prompt,
    high_confidence_assumptions,
    run_analysis,
    valid_goal_ids,
)
from .schemas import DMEntry, DMMapping, MappingAnalysis

_CACHE_FILENAME = "mapping_dm.pkl"


class _DMJudgementResponse(BaseModel):
    goal_id: str | None
    confidence: Literal["low", "medium", "high"]
    reasoning: str


async def _judge_one(
    decision: Decision,
    goals: list[CanonicalGoal],
    analysis: MappingAnalysis,
) -> DMEntry:
    high_assumptions = high_confidence_assumptions(analysis)

    system_prompt = build_prompt("mapping_judgement_dm_system.jinja")
    user_prompt = build_prompt(
        "mapping_judgement_user.jinja",
        goals_formatted=format_goals_for_prompt(goals),
        goals_count=len(goals),
        decision_formatted=format_decision_for_prompt(decision),
        decision_id=decision.id,
        analysis_text=analysis.analysis,
        high_confidence_assumptions=high_assumptions,
        has_high_assumptions=bool(high_assumptions),
    )

    response: _DMJudgementResponse = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=_DMJudgementResponse,
        llm_model=settings.mapping_judgement_model,
        temperature=settings.temperature,
        max_tokens=2048,
    )

    # Validate the goal ID: drop a hallucinated link (treat as "no goal applies")
    # so DM is held to the same ID-resolvability bar as GM.
    goal_id = response.goal_id
    if goal_id is not None and goal_id not in valid_goal_ids(goals):
        logger.warning(f"DM: dropping unresolvable goal_id for decision {decision.id}: {goal_id}")
        goal_id = None

    return DMEntry(
        decision_id=decision.id,
        goal_id=goal_id,
        confidence=response.confidence,
        reasoning=response.reasoning,
    )


async def run_dm_mapper(
    decisions: list[Decision],
    goals: list[CanonicalGoal],
    condition_name: str,
    output_path: Path,
    load_from_cache: bool = True,
) -> DMMapping:
    """Map each decision to at most one goal (Step B, DM schema)."""
    cache_path = output_path / _CACHE_FILENAME

    if load_from_cache and cache_path.exists():
        log_cache_hit(cache_path)
        return load_pickle_file(cache_path)

    # Step A — shared analysis (loads from cache if available)
    analyses: dict[str, MappingAnalysis] = await run_analysis(
        decisions=decisions,
        goals=goals,
        condition_name=condition_name,
        output_path=output_path,
        load_from_cache=load_from_cache,
    )

    # Step B — DM judgement (cold restart, HIGH assumptions only)
    print(f"  DM judgement: {len(decisions)} decisions [{settings.mapping_judgement_model}]...")
    set_async_instructor_client(settings.mapping_judgement_model, settings.anthropic_api_key)

    semaphore = asyncio.Semaphore(settings.mapping_max_concurrency)
    tasks = [
        limited_task(
            _judge_one(d, goals, analyses[d.id]),
            semaphore,
            settings.delay_between_tasks,
        )
        for d in decisions
        if d.id in analyses
    ]
    entries: list[DMEntry] = list(await execute_tasks_with_manual_pbar(tasks))

    mapping = DMMapping(
        condition_name=condition_name,
        entries=entries,
        model_ids={
            "analysis": settings.mapping_analysis_model,
            "judgement": settings.mapping_judgement_model,
        },
    )

    dump_to_pickle_file(mapping, cache_path)
    linked = sum(1 for e in entries if e.goal_id is not None)
    print(f"  DM done: {len(entries)} entries, {linked} linked, {len(entries)-linked} null → {cache_path}")
    return mapping
