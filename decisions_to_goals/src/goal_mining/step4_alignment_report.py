from pathlib import Path

from ..instruct_helper import ainstruct_llm, set_async_instructor_client
from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit
from common.prompt_template_engine import build_prompt

from ..models import AlignmentReport, CanonicalGoal
from ..settings import logger, settings

CACHE_FILENAME = "step4_alignment_report.pkl"


def _format_goals(goals: list[CanonicalGoal]) -> str:
    lines = []
    for g in goals:
        stated_tag = "stated" if g.is_stated else "unstated"
        lines.append(f"[{g.id}] ({stated_tag}) {g.title}\n  {g.description}")
    return "\n\n".join(lines)


async def run_step4(
    canonical_goals: list[CanonicalGoal],
    output_path: Path,
    load_from_cache: bool = True,
) -> AlignmentReport:
    """Identify synergies and tensions between canonical goals."""
    cache_path = output_path / CACHE_FILENAME

    if load_from_cache and cache_path.exists():
        log_cache_hit(cache_path)
        return load_pickle_file(cache_path)

    print(f"  Step 4: building alignment report for {len(canonical_goals)} goals with {settings.step4_model}...")
    set_async_instructor_client(settings.step4_model, settings.anthropic_api_key)

    system_prompt = build_prompt("step4_alignment_report_system.txt.jinja")
    user_prompt = build_prompt(
        "step4_alignment_report_user.txt.jinja",
        goals_formatted=_format_goals(canonical_goals),
        goals_count=len(canonical_goals),
    )

    report: AlignmentReport = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=AlignmentReport,
        llm_model=settings.step4_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    dump_to_pickle_file(report, cache_path)
    print(f"  Step 4: {len(report.relations)} relations → {cache_path}")
    return report
