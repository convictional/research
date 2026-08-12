from pathlib import Path

from pydantic import BaseModel

from ..instruct_helper import ainstruct_llm, set_async_instructor_client

from common.io import dump_to_pickle_file, load_pickle_file

from ..cache_log import log_cache_hit
from common.prompt_template_engine import build_prompt

from ..models import AlignmentReport, CanonicalGoal
from ..settings import logger, settings

CACHE_FILENAME = "step5_summary.pkl"


class _SummaryResponse(BaseModel):
    summary_markdown: str


def _format_goals(goals: list[CanonicalGoal]) -> str:
    lines = []
    for g in goals:
        stated_tag = "stated" if g.is_stated else "unstated"
        lines.append(
            f"[{g.id}] ({stated_tag}, support={g.activity_support_score:.2f}) "
            f"{g.title}\n  {g.description}"
        )
    return "\n\n".join(lines)


def _format_alignment_report(report: AlignmentReport | None) -> str:
    if report is None:
        return ""
    lines = [f"Overall: {report.summary}", ""]
    for r in report.relations:
        lines.append(f"- [{r.relation}] {r.goal_a_id} ↔ {r.goal_b_id}: {r.label} (confidence={r.confidence:.2f})")
    return "\n".join(lines)


async def run_step5(
    canonical_goals: list[CanonicalGoal],
    alignment_report: AlignmentReport | None,
    output_path: Path,
    load_from_cache: bool = True,
) -> str:
    """Generate the final summary markdown. Works with or without an alignment report."""
    cache_path = output_path / CACHE_FILENAME

    if load_from_cache and cache_path.exists():
        log_cache_hit(cache_path)
        return load_pickle_file(cache_path)

    has_report = alignment_report is not None
    print(f"  Step 5: generating summary ({len(canonical_goals)} goals, alignment_report={'yes' if has_report else 'no'}) with {settings.step5_model}...")
    set_async_instructor_client(settings.step5_model, settings.anthropic_api_key)

    system_prompt = build_prompt("step5_summary_system.txt.jinja")
    user_prompt = build_prompt(
        "step5_summary_user.txt.jinja",
        goals_formatted=_format_goals(canonical_goals),
        goals_count=len(canonical_goals),
        has_alignment_report=has_report,
        alignment_report_formatted=_format_alignment_report(alignment_report),
    )

    response: _SummaryResponse = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=_SummaryResponse,
        llm_model=settings.step5_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    summary = response.summary_markdown
    dump_to_pickle_file(summary, cache_path)
    print(f"  Step 5: summary generated ({len(summary)} chars) → {cache_path}")
    return summary
