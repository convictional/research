from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt
from src.models import DisagreementAnalysis, DiscoveredRubric, ScoredReport
from src.settings import settings, logger


def find_disagreements(scored_reports: list[ScoredReport], min_gap: int = 2) -> list[ScoredReport]:
    disagreements = [
        r for r in scored_reports
        if abs(r.report.quality_score - r.predicted_score.quality_score) >= min_gap
    ]
    disagreements.sort(
        key=lambda r: abs(r.report.quality_score - r.predicted_score.quality_score),
        reverse=True,
    )
    return disagreements


async def analyze_disagreements(
    scored_reports: list[ScoredReport],
    rubric: DiscoveredRubric,
    min_gap: int = 2,
) -> DisagreementAnalysis | None:
    disagreements = find_disagreements(scored_reports, min_gap)
    if not disagreements:
        logger.info("No significant disagreements found")
        return None

    # Cap at 10 worst cases to keep prompt manageable
    cases = disagreements[:10]
    logger.info(f"Analyzing {len(cases)} disagreement cases (of {len(disagreements)} total, gap >= {min_gap})")

    case_data = []
    for r in cases:
        case_data.append({
            "question": r.report.question,
            "expert_score": r.report.quality_score,
            "predicted_score": r.predicted_score.quality_score,
            "justification": r.predicted_score.overall_justification,
            "research_output": r.report.research_output[:2000],
        })

    system_prompt = build_prompt("disagreement_analysis_system.txt.jinja")
    user_prompt = build_prompt(
        "disagreement_analysis_user.txt.jinja",
        cases=case_data,
        min_gap=min_gap,
        rubric=rubric,
    )

    analysis = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=DisagreementAnalysis,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    _print_analysis(analysis, cases)
    return analysis


def _print_analysis(analysis: DisagreementAnalysis, cases: list[ScoredReport]) -> None:
    print(f"\n{'='*60}")
    print(f"  Disagreement Analysis ({len(cases)} cases)")
    print(f"{'='*60}")

    for insight in analysis.insights:
        case = cases[insight.case_index] if insight.case_index < len(cases) else None
        if case:
            print(f"\n  Case {insight.case_index + 1}: expert={case.report.quality_score}, "
                  f"predicted={case.predicted_score.quality_score}")
        print(f"    Root cause: {insight.root_cause}")
        print(f"    Analysis: {insight.analysis[:200]}")

    if analysis.rubric_changes:
        print(f"\n  Suggested rubric changes:")
        for change in analysis.rubric_changes:
            print(f"    - {change}")

    if analysis.prompt_changes:
        print(f"\n  Suggested prompt changes:")
        for change in analysis.prompt_changes:
            print(f"    - {change}")
