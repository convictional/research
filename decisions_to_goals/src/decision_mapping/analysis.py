"""Shared analysis step — identical across all mapping schemas.

All three schemas (DM, DSM, GM) call this module to run Step A. Keeping the analysis
step common ensures that schema-level effects are isolated from analysis-quality effects.
"""
import asyncio
from pathlib import Path

from common.async_helper import execute_tasks_with_manual_pbar, limited_task
from ..instruct_helper import ainstruct_llm, set_async_instructor_client
from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit
from common.prompt_template_engine import build_prompt

from ..models import CanonicalGoal, Decision
from ..settings import logger, settings
from .schemas import MappingAnalysis

_ANALYSIS_CACHE_FILENAME = "mapping_analysis.pkl"


def valid_goal_ids(goals: list[CanonicalGoal]) -> set[str]:
    """Set of canonical goal IDs the LLM is allowed to reference.

    Used by all three mappers to drop hallucinated goal IDs the LLM may emit,
    so every schema is held to the same ID-resolvability bar.
    """
    return {g.id for g in goals}


def format_goals_for_prompt(goals: list[CanonicalGoal]) -> str:
    lines = []
    for g in goals:
        stated = "stated" if g.is_stated else "unstated"
        lines.append(f"[{g.id}] ({stated}) {g.title}\n  {g.description}")
    return "\n\n".join(lines)


def format_decision_for_prompt(decision: Decision) -> str:
    opts = "\n".join(
        f"  - {o.get('title', '')}: {str(o.get('description', ''))[:200]}" for o in decision.options[:5]
    ) or "  (none)"
    crits = "\n".join(
        f"  - {c.get('title', '')}: {str(c.get('description', ''))[:200]}" for c in decision.criteria[:5]
    ) or "  (none)"
    comments = "\n".join(
        f"  - {str(c.get('user', '?'))} ({c.get('created_at', '')}): {str(c.get('text', ''))[:200]}"
        for c in decision.comments[:5]
    ) or "  (none)"
    return (
        f"ID: {decision.id}\n"
        f"Title: {decision.title}\n"
        f"Author's stated goals: {decision.author_stated_goals or '(none stated)'}\n\n"
        f"Options:\n{opts}\n\n"
        f"Criteria:\n{crits}\n\n"
        f"Discussion:\n{comments}"
    )


async def _analyze_one(
    decision: Decision,
    goals: list[CanonicalGoal],
    condition_name: str,
) -> MappingAnalysis:
    system_prompt = build_prompt("mapping_analysis_system.jinja")
    user_prompt = build_prompt(
        "mapping_analysis_user.jinja",
        condition_name=condition_name,
        goals_formatted=format_goals_for_prompt(goals),
        goals_count=len(goals),
        decision_formatted=format_decision_for_prompt(decision),
        decision_id=decision.id,
    )
    response: MappingAnalysis = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=MappingAnalysis,
        llm_model=settings.mapping_analysis_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    return response.model_copy(update={"decision_id": decision.id})


async def run_analysis(
    decisions: list[Decision],
    goals: list[CanonicalGoal],
    condition_name: str,
    output_path: Path,
    load_from_cache: bool = True,
) -> dict[str, MappingAnalysis]:
    """Run Step A (analysis) for all decisions. Results are shared by all three mapping schemas."""
    cache_path = output_path / _ANALYSIS_CACHE_FILENAME

    if load_from_cache and cache_path.exists():
        log_cache_hit(cache_path)
        return load_pickle_file(cache_path)

    print(f"  Analysis: {len(decisions)} decisions × {len(goals)} goals [{settings.mapping_analysis_model}]...")
    set_async_instructor_client(settings.mapping_analysis_model, settings.anthropic_api_key)

    semaphore = asyncio.Semaphore(settings.mapping_max_concurrency)
    tasks = [
        limited_task(
            _analyze_one(d, goals, condition_name),
            semaphore,
            settings.delay_between_tasks,
        )
        for d in decisions
    ]
    results = await execute_tasks_with_manual_pbar(tasks)

    analyses: dict[str, MappingAnalysis] = {a.decision_id: a for a in results}
    dump_to_pickle_file(analyses, cache_path)
    print(f"  Analysis done: {len(analyses)} entries → {cache_path}")
    return analyses


def high_confidence_assumptions(analysis: MappingAnalysis) -> list[str]:
    return [a.text for a in analysis.assumptions if a.confidence == "high"]
