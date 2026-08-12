"""MoE judge ensemble: 3 models × 3 roles = 9 judges per cell.

Critical constraint: cell_id is NEVER passed to any judge prompt. Judges see
only a fixed-length research summary (the obfuscation-layer output) and the
anchored rubric. The raw mapping artifact is never passed to judges directly.
"""
import asyncio
import time
from pathlib import Path

from common.async_helper import execute_tasks_with_manual_pbar, limited_task
from ..instruct_helper import ainstruct_llm, set_async_instructor_client
from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit
from common.prompt_template_engine import build_prompt

from ..settings import CLAUDE_HAIKU, CLAUDE_OPUS, CLAUDE_SONNET, logger, settings
from .rubric import CalibrationResult, JudgeRun, JudgeScore, format_rubric_text

# Schema identity is masked by construction: it appears only in cache-key/filenames,
# never in the summary text or any judge prompt (see decision_mapping/render.py and
# research_summary.py). No acronym blacklist is applied, so coincidental tokens like
# "GM"/"DM" in real org data pass through without aborting a cell.

_JUDGE_MODELS = [CLAUDE_OPUS, CLAUDE_SONNET, CLAUDE_HAIKU]
_JUDGE_ROLES = ["strategy_analyst", "ops_reviewer", "skeptic"]

_ROLE_SYSTEM_PROMPTS = {
    "strategy_analyst": "judge_system_strategy_analyst.jinja",
    "ops_reviewer": "judge_system_ops_reviewer.jinja",
    "skeptic": "judge_system_skeptic.jinja",
}


def cache_filename(schema: str, temperature: float) -> str:
    return f"judge_{schema}_T{temperature:.1f}.pkl"


async def _run_one_judge(
    cid: str,
    rendered_md: str,
    rendered_word_count: int,
    model_id: str,
    role: str,
    temperature: float,
) -> JudgeRun:
    # cell_id is bookkeeping only and never touches the prompt.
    rubric_text = format_rubric_text()
    system_prompt = build_prompt(_ROLE_SYSTEM_PROMPTS[role])
    user_prompt = build_prompt(
        "judge_user.jinja",
        artifact_markdown=rendered_md,
        rubric_text=rubric_text,
    )

    t0 = time.monotonic()
    score: JudgeScore = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=JudgeScore,
        llm_model=model_id,
        temperature=temperature,
        max_tokens=4096,
    )
    duration = time.monotonic() - t0

    # Clamp dimensions to valid range and flag divergent self_reported_overall
    sum_dims = (
        score.coverage + score.fidelity + score.synthesis_quality
        + score.interpretability + score.information_density
    )
    if abs(score.self_reported_overall - sum_dims) > 1:
        logger.warning(
            f"Judge divergence in {cid} / {model_id} / {role}: "
            f"sum_dims={sum_dims}, self_reported={score.self_reported_overall}"
        )

    return JudgeRun(
        cell_id=cid,
        model_id=model_id,
        role=role,
        temperature=temperature,
        score=score,
        duration_seconds=duration,
        rendered_word_count=rendered_word_count,
    )


async def run_cell_judges(
    cell_id: str,
    rendered_md: str,
    rendered_word_count: int,
    schema: str,
    output_path: Path,
    temperature: float = 0.0,
    load_from_cache: bool = True,
) -> list[JudgeRun]:
    """Run all 9 judges (3 models × 3 roles) for one cell.

    cell_id is stored in JudgeRun for bookkeeping but is NEVER passed to any judge prompt.
    """
    cache_path = output_path / cache_filename(schema, temperature)

    if load_from_cache and cache_path.exists():
        log_cache_hit(cache_path)
        return load_pickle_file(cache_path)

    print(f"  Judging cell={cell_id} schema={schema} T={temperature}: 9 judges...")

    # All judges use the same Anthropic client — set it once
    set_async_instructor_client(CLAUDE_OPUS, settings.anthropic_api_key)

    semaphore = asyncio.Semaphore(settings.mapping_max_concurrency)
    tasks = [
        limited_task(
            _run_one_judge(cell_id, rendered_md, rendered_word_count, model, role, temperature),
            semaphore,
            settings.delay_between_tasks,
        )
        for model in _JUDGE_MODELS
        for role in _JUDGE_ROLES
    ]

    runs: list[JudgeRun] = list(await execute_tasks_with_manual_pbar(tasks))

    dump_to_pickle_file(runs, cache_path)
    overalls = [r.score.self_reported_overall for r in runs]
    print(f"  Judged: {len(runs)} runs, overalls={overalls} → {cache_path}")
    return runs
