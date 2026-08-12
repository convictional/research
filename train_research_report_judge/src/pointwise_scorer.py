import asyncio
import random
import statistics
from collections import Counter

from common.async_helper import execute_tasks_with_manual_pbar, limited_task
from common.instruct_llm import ainstruct_llm
from common.prompt_template_engine import build_prompt
from src.models import (
    ClaimAnalysis,
    ClaimVerificationRollup,
    DiscoveredRubric,
    PointwiseScore,
    RatedReport,
    ReportCritique,
    ScoredReport,
)
from src.settings import settings, logger

TARGET_CALIBRATION_SCORES = [0, 1, 1, 2, 3]


def _select_calibration_examples(
    train_reports: list[RatedReport], n: int = settings.calibration_example_count
) -> list[RatedReport]:
    by_score: dict[int, list[RatedReport]] = {}
    for r in train_reports:
        by_score.setdefault(r.quality_score, []).append(r)

    # Sort each bucket by report length (prefer shorter for prompt size management)
    for score in by_score:
        by_score[score].sort(key=lambda r: len(r.research_output))

    target_scores = TARGET_CALIBRATION_SCORES[:n]
    score_counts: dict[int, int] = Counter(target_scores)
    examples: list[RatedReport] = []

    for score, needed in sorted(score_counts.items()):
        available = [r for r in by_score.get(score, []) if r not in examples]
        selected = available[:needed]
        examples.extend(selected)

    # Fill remaining from any available if a bucket was too small
    while len(examples) < n:
        remaining = [r for bucket in by_score.values() for r in bucket if r not in examples]
        if not remaining:
            break
        examples.append(remaining[0])

    random.shuffle(examples)
    return examples


def _select_contrastive_pairs(
    train_reports: list[RatedReport], n_pairs: int = 3
) -> list[tuple[RatedReport, RatedReport]]:
    """Select pairs of length-matched reports from the score 1 vs 2 boundary."""
    score_1 = [r for r in train_reports if r.quality_score == 1]
    score_2 = [r for r in train_reports if r.quality_score == 2]

    if not score_1 or not score_2:
        return []

    score_1.sort(key=lambda r: len(r.research_output))
    score_2.sort(key=lambda r: len(r.research_output))

    pairs: list[tuple[RatedReport, RatedReport]] = []
    used_2: set[str] = set()

    for r1 in score_1:
        len_1 = len(r1.research_output)
        best_match = None
        best_diff = float("inf")
        for r2 in score_2:
            if r2.id in used_2:
                continue
            diff = abs(len(r2.research_output) - len_1)
            if diff < best_diff:
                best_diff = diff
                best_match = r2
        if best_match:
            pairs.append((r1, best_match))
            used_2.add(best_match.id)
        if len(pairs) >= n_pairs:
            break

    return pairs


def _compute_train_distribution(train_reports: list[RatedReport]) -> dict[int, float]:
    counts = Counter(r.quality_score for r in train_reports)
    total = len(train_reports)
    return {score: round(counts.get(score, 0) / total * 100, 1) for score in range(4)}


def _compute_weighted_score(predicted: PointwiseScore, rubric: DiscoveredRubric) -> int:
    weight_map = {dim.name: dim.weight for dim in rubric.dimensions}
    weighted_sum = 0.0
    total_weight = 0.0
    for ds in predicted.dimension_scores:
        w = weight_map.get(ds.dimension, 0.0)
        weighted_sum += ds.score * w
        total_weight += w
    if total_weight > 0:
        return round(weighted_sum / total_weight)
    return predicted.quality_score


async def _critique_report(report: RatedReport) -> ReportCritique:
    system_prompt = build_prompt("critic_system.txt.jinja")
    user_prompt = build_prompt(
        "qa_alignment_gate_user.txt.jinja",
        question=report.question,
        research_output=report.research_output,
    )

    return await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ReportCritique,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


async def _analyze_claims(report: RatedReport) -> ClaimAnalysis:
    system_prompt = build_prompt("claim_analysis_system.txt.jinja")
    user_prompt = build_prompt(
        "claim_analysis_user.txt.jinja",
        question=report.question,
        research_output=report.research_output,
    )

    return await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ClaimAnalysis,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


async def _run_scoring_pass(
    report: RatedReport,
    rubric: DiscoveredRubric,
    calibration_examples: list[RatedReport],
    train_distribution: dict[int, float],
    critique: ReportCritique,
    claim_analysis: ClaimAnalysis | None = None,
    contrastive_pairs: list[tuple[RatedReport, RatedReport]] | None = None,
    rag_verification: ClaimVerificationRollup | None = None,
    temperature: float | None = None,
) -> PointwiseScore:
    """Execute a single scoring pass given pre-computed critique and optional claim analysis."""
    system_prompt = build_prompt(
        "pointwise_scorer_system.txt.jinja",
        rubric=rubric,
        calibration_examples=calibration_examples,
        train_distribution=train_distribution,
        calibration_max_chars=settings.calibration_max_chars,
        contrastive_pairs=contrastive_pairs or [],
    )

    user_kwargs = dict(
        question=report.question,
        research_output=report.research_output,
        critique=critique,
        claim_analysis=claim_analysis,
        rag_verification=rag_verification,
        variant=report.variant if settings.include_metadata else None,
        community_relation=report.community_relation if settings.include_metadata else None,
    )
    user_prompt = build_prompt("pointwise_scorer_user.txt.jinja", **user_kwargs)

    return await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=PointwiseScore,
        llm_model=settings.llm_model,
        temperature=temperature or settings.temperature,
        max_tokens=settings.max_tokens,
    )


async def score_report(
    report: RatedReport,
    rubric: DiscoveredRubric,
    calibration_examples: list[RatedReport],
    train_distribution: dict[int, float],
    contrastive_pairs: list[tuple[RatedReport, RatedReport]] | None = None,
) -> ScoredReport:
    """Score a single report with optional ensemble and claim analysis."""
    # Pass 1: Critique + optional claim analysis in parallel
    tasks_to_gather: list = [_critique_report(report)]
    if settings.claim_analysis_enabled:
        tasks_to_gather.append(_analyze_claims(report))

    gather_results = await asyncio.gather(*tasks_to_gather)
    critique = gather_results[0]
    claim_analysis = gather_results[1] if settings.claim_analysis_enabled else None

    # Optional: RAG verification of extracted claims
    rag_verification = None
    if settings.rag_verification_enabled and claim_analysis:
        from src.claim_verifier import verify_claims

        rag_verification = await verify_claims(claim_analysis.claims, report.question)

    # Pass 2: Ensemble scoring — run N scoring passes, reusing the same critique
    ensemble_n = settings.ensemble_n
    if ensemble_n <= 1:
        predicted = await _run_scoring_pass(
            report, rubric, calibration_examples, train_distribution,
            critique, claim_analysis, contrastive_pairs, rag_verification,
        )
        return ScoredReport(report=report, predicted_score=predicted)

    scoring_tasks = [
        _run_scoring_pass(
            report, rubric, calibration_examples, train_distribution,
            critique, claim_analysis, contrastive_pairs, rag_verification,
            temperature=settings.ensemble_temperature,
        )
        for _ in range(ensemble_n)
    ]
    ensemble_results = await asyncio.gather(*scoring_tasks)

    # Aggregate: median for integer score, mean for continuous
    scores = [r.quality_score for r in ensemble_results]
    median_score = int(statistics.median(scores))
    mean_score = statistics.mean(scores)

    # Use the justification from the result closest to the median
    best_result = min(ensemble_results, key=lambda r: abs(r.quality_score - median_score))

    predicted = PointwiseScore(
        dimension_scores=best_result.dimension_scores,
        quality_score=median_score,
        overall_justification=(
            f"[Ensemble of {ensemble_n}: scores={scores}, median={median_score}, mean={mean_score:.2f}] "
            + best_result.overall_justification
        ),
    )
    return ScoredReport(report=report, predicted_score=predicted, continuous_score=mean_score)


async def score_reports(
    reports: list[RatedReport],
    rubric: DiscoveredRubric,
    train_reports: list[RatedReport],
) -> list[ScoredReport]:
    logger.info(f"Scoring {len(reports)} reports with rubric v{rubric.version}")
    logger.info(f"Ensemble N={settings.ensemble_n}, claim_analysis={settings.claim_analysis_enabled}, "
                f"metadata={settings.include_metadata}, rag_verification={settings.rag_verification_enabled}")

    if settings.rag_verification_enabled:
        from src.content_search import close_search, init_search

        await init_search()

    calibration_examples = _select_calibration_examples(train_reports)
    train_distribution = _compute_train_distribution(train_reports)
    contrastive_pairs = _select_contrastive_pairs(train_reports) if settings.include_metadata else None
    logger.info(f"Using {len(calibration_examples)} calibration examples "
                f"(scores: {[e.quality_score for e in calibration_examples]})")
    logger.info(f"Train distribution: {train_distribution}")

    semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def _score_one(report: RatedReport) -> ScoredReport:
        return await score_report(report, rubric, calibration_examples, train_distribution, contrastive_pairs)

    try:
        tasks = [
            limited_task(_score_one(report), semaphore, settings.delay_between_tasks)
            for report in reports
        ]
        results = await execute_tasks_with_manual_pbar(tasks)
        logger.info(f"Scoring complete for {len(results)} reports")
        return results
    finally:
        if settings.rag_verification_enabled:
            from src.content_search import close_search

            await close_search()


async def score_single(
    question: str,
    research_output: str,
    rubric: DiscoveredRubric,
    calibration_examples: list[RatedReport],
    train_distribution: dict[int, float] | None = None,
) -> PointwiseScore:
    if train_distribution is None:
        train_distribution = {0: 14.0, 1: 39.0, 2: 42.0, 3: 5.0}
    report = RatedReport(
        id="single",
        question=question,
        community_relation="unknown",
        variant="unknown",
        research_output=research_output,
        quality_score=0,
    )
    scored = await score_report(report, rubric, calibration_examples, train_distribution)
    return scored.predicted_score
