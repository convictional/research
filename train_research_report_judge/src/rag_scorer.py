import asyncio

from common.async_helper import execute_tasks_with_manual_pbar, limited_task
from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt
from src.claim_verifier import verify_claims
from src.content_search import close_search, init_search
from src.format_scorer import score_format
from src.models import (
    ClaimVerificationRollup,
    DimensionScore,
    FormatAssessment,
    PointwiseScore,
    RAGFinalScore,
    RatedReport,
    ScoredReport,
)
from src.pointwise_scorer import _analyze_claims
from src.settings import logger, settings


async def _final_score(
    question: str,
    rollup: ClaimVerificationRollup,
    format_assessment: FormatAssessment,
) -> RAGFinalScore:
    system_prompt = build_prompt("rag_final_scorer_system.txt.jinja")
    user_prompt = build_prompt(
        "rag_final_scorer_user.txt.jinja",
        question=question,
        rollup=rollup,
        format_assessment=format_assessment,
    )

    return await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=RAGFinalScore,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


async def score_report_with_rag(report: RatedReport) -> ScoredReport:
    # Step 1: Extract claims (Sonnet)
    claim_analysis = await _analyze_claims(report)

    # Step 2-3: Verify claims via RAG + Step 4: Score format (parallel)
    rollup, format_assessment = await asyncio.gather(
        verify_claims(claim_analysis.claims, report.question),
        score_format(report),
    )

    # Step 5: Final scoring (Sonnet)
    final = await _final_score(report.question, rollup, format_assessment)

    predicted = PointwiseScore(
        dimension_scores=[
            DimensionScore(
                dimension="claim_verification",
                score=min(3, round(rollup.supported / rollup.total_claims * 3)) if rollup.total_claims > 0 else 0,
                justification=f"{rollup.supported}/{rollup.total_claims} claims supported",
            ),
            DimensionScore(
                dimension="format_quality",
                score=round((format_assessment.structure_score + format_assessment.tone_score + format_assessment.qa_alignment_score) / 3),
                justification=format_assessment.notes,
            ),
        ],
        quality_score=final.quality_score,
        overall_justification=final.justification,
    )

    return ScoredReport(report=report, predicted_score=predicted)


async def score_reports_with_rag(reports: list[RatedReport]) -> list[ScoredReport]:
    logger.info(f"RAG scoring {len(reports)} reports")

    await init_search()

    try:
        semaphore = asyncio.Semaphore(settings.max_concurrency)
        tasks = [
            limited_task(score_report_with_rag(report), semaphore, settings.delay_between_tasks)
            for report in reports
        ]
        results = await execute_tasks_with_manual_pbar(tasks)
        logger.info(f"RAG scoring complete for {len(results)} reports")
        return results
    finally:
        await close_search()
