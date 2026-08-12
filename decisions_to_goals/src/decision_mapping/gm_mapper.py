"""GM mapper — decisions and goals form a labeled relationship graph.

NOTE: No retrieval pre-filter is applied here. Every decision is analyzed against
the full canonical goal set without keyword/entity/cosine pre-screening.
This is an intentional Phase-1 design constraint to isolate the schema-axis
comparison. The keyword + entity + cosine pre-filter from
linking_tasks_to_goals/approach_10 is a candidate for a Phase-2 follow-up
experiment if GM shows noise from irrelevant node pairs.
"""
import asyncio
from pathlib import Path

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
from .schemas import GMEdge, GMMapping, MappingAnalysis

_CACHE_FILENAME = "mapping_gm.pkl"


class _GMJudgementResponse(BaseModel):
    edges: list[GMEdge]


async def _judge_one(
    decision: Decision,
    goals: list[CanonicalGoal],
    analysis: MappingAnalysis,
    all_decision_ids: set[str],
    all_goal_ids: set[str],
) -> list[GMEdge]:
    high_assumptions = high_confidence_assumptions(analysis)

    system_prompt = build_prompt("mapping_judgement_gm_system.jinja")
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

    response: _GMJudgementResponse = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=_GMJudgementResponse,
        llm_model=settings.mapping_judgement_model,
        temperature=settings.temperature,
        max_tokens=4096,
    )

    # Validate edges: source_kind and target_kind must match the actual node type
    valid_edges = []
    for edge in response.edges:
        src_ok = (edge.source_kind == "decision" and edge.source_id in all_decision_ids) or \
                 (edge.source_kind == "goal" and edge.source_id in all_goal_ids)
        tgt_ok = (edge.target_kind == "decision" and edge.target_id in all_decision_ids) or \
                 (edge.target_kind == "goal" and edge.target_id in all_goal_ids)
        if src_ok and tgt_ok:
            valid_edges.append(edge)
        else:
            logger.warning(f"GM: dropping edge with unresolvable IDs: {edge.source_id} → {edge.target_id}")

    return valid_edges


async def run_gm_mapper(
    decisions: list[Decision],
    goals: list[CanonicalGoal],
    condition_name: str,
    output_path: Path,
    load_from_cache: bool = True,
) -> GMMapping:
    """Emit labeled edges between decisions and goals (Step B, GM schema)."""
    cache_path = output_path / _CACHE_FILENAME

    if load_from_cache and cache_path.exists():
        log_cache_hit(cache_path)
        return load_pickle_file(cache_path)

    # Step A — shared analysis
    analyses: dict[str, MappingAnalysis] = await run_analysis(
        decisions=decisions,
        goals=goals,
        condition_name=condition_name,
        output_path=output_path,
        load_from_cache=load_from_cache,
    )

    # Build ID sets for edge validation
    all_decision_ids = {d.id for d in decisions}
    all_goal_ids = valid_goal_ids(goals)

    # Step B — GM judgement (cold restart, HIGH assumptions only)
    print(f"  GM judgement: {len(decisions)} decisions [{settings.mapping_judgement_model}]...")
    set_async_instructor_client(settings.mapping_judgement_model, settings.anthropic_api_key)

    semaphore = asyncio.Semaphore(settings.mapping_max_concurrency)
    tasks = [
        limited_task(
            _judge_one(d, goals, analyses[d.id], all_decision_ids, all_goal_ids),
            semaphore,
            settings.delay_between_tasks,
        )
        for d in decisions
        if d.id in analyses
    ]
    per_decision_edges: list[list[GMEdge]] = list(await execute_tasks_with_manual_pbar(tasks))
    all_edges: list[GMEdge] = [e for edges in per_decision_edges for e in edges]

    mapping = GMMapping(
        condition_name=condition_name,
        edges=all_edges,
        model_ids={
            "analysis": settings.mapping_analysis_model,
            "judgement": settings.mapping_judgement_model,
        },
    )

    dump_to_pickle_file(mapping, cache_path)
    print(f"  GM done: {len(all_edges)} edges across {len(decisions)} decisions → {cache_path}")
    return mapping
