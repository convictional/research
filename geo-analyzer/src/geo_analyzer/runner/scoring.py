"""Scoring pipeline. Apply Phase 1 extractors per sample, aggregate per
(prompt, model, subject) per DESIGN §5.3.

Inputs:
  - tasks: every task for a run (failed tasks are skipped automatically).
  - run_id: tagged onto every emitted Score.
  - prompt_targets: prompt_id → list of subject ids the prompt targets.
  - subjects: subject_id → Subject (full catalog map; needed for SoV
    competitor lookups and the anti_brand detection).

Output: list[Score].
"""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from geo_analyzer.runtime import (
    SampleAggregation,
    Score,
    Task,
    TaskStatus,
)
from geo_analyzer.scoring import (
    brand_legacy_conflation,
    mention_presence,
    ordinal_rank,
    share_of_voice,
)
from geo_analyzer.scoring.aggregation import (
    majority_vote,
    mean_of_floats,
    mean_rate,
    median_or_none,
)
from geo_analyzer.types import Subject, SubjectKind

_BRAND_LIKE = {SubjectKind.BRAND, SubjectKind.CATEGORY}


def score_run(
    tasks: list[Task],
    *,
    run_id: str,
    prompt_targets: dict[str, list[str]],
    subjects: dict[str, Subject],
) -> list[Score]:
    """Compute aggregated scores from the run's task results."""
    successful = [t for t in tasks if t.status == TaskStatus.SUCCESS]
    if not successful:
        return []

    # Group tasks by (prompt_id, model_id) — these are the aggregation cohorts.
    cohorts: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for t in successful:
        cohorts[(t.prompt_id, t.model_id)].append(t)

    # Identify the brand and anti_brand subjects for conflation (if any exist).
    anti_brands = [s for s in subjects.values() if s.kind == SubjectKind.ANTI_BRAND]
    brands = [s for s in subjects.values() if s.kind == SubjectKind.BRAND]

    out: list[Score] = []

    for (prompt_id, model_id), cohort in cohorts.items():
        target_ids = prompt_targets.get(prompt_id, [])
        n = len(cohort)
        is_grounded_multi = n > 1

        # --- per-target subject metrics ---
        for sid in target_ids:
            subj = subjects.get(sid)
            if subj is None:
                continue
            if subj.kind not in _BRAND_LIKE:
                continue

            presence_samples = [mention_presence(t.text, subj).present for t in cohort]
            rank_samples = [ordinal_rank(t.text, subj).rank for t in cohort]
            sov_samples = [share_of_voice(t.text, subj, subjects).value for t in cohort]

            out.append(
                _score(
                    run_id,
                    prompt_id,
                    model_id,
                    sid,
                    metric="mention_presence",
                    value=majority_vote(presence_samples) if is_grounded_multi else presence_samples[0],
                    aggregation="majority_vote" if is_grounded_multi else "single",
                )
            )
            if is_grounded_multi:
                out.append(
                    _score(
                        run_id,
                        prompt_id,
                        model_id,
                        sid,
                        metric="mention_presence_rate",
                        value=mean_rate(presence_samples),
                        aggregation="mean",
                    )
                )

            out.append(
                _score(
                    run_id,
                    prompt_id,
                    model_id,
                    sid,
                    metric="ordinal_rank",
                    value=median_or_none(rank_samples) if is_grounded_multi else rank_samples[0],
                    aggregation="median" if is_grounded_multi else "single",
                )
            )

            out.append(
                _score(
                    run_id,
                    prompt_id,
                    model_id,
                    sid,
                    metric="share_of_voice",
                    value=mean_of_floats(sov_samples) if is_grounded_multi else sov_samples[0],
                    aggregation="mean" if is_grounded_multi else "single",
                )
            )

        # --- conflation: brand x anti_brand co-occurrence ---
        # Each anti_brand declares its parent via `legacy_of`. We pair only with
        # that one brand — pairing with every brand in the catalog (e.g. Notion,
        # Slack) would emit meaningless False rows that dilute the rate.
        # Anti-brands with legacy_of=None are skipped.
        for anti in anti_brands:
            if anti.legacy_of is None:
                continue
            brand = next((b for b in brands if b.id == anti.legacy_of), None)
            if brand is None:
                continue  # catalog loader should reject this case, but be defensive.
            conflation_samples = [brand_legacy_conflation(t.text, brand, anti).fired for t in cohort]
            out.append(
                _score(
                    run_id,
                    prompt_id,
                    model_id,
                    anti.id,
                    metric="brand_legacy_conflation",
                    value=majority_vote(conflation_samples) if is_grounded_multi else conflation_samples[0],
                    aggregation="majority_vote" if is_grounded_multi else "single",
                )
            )
            if is_grounded_multi:
                out.append(
                    _score(
                        run_id,
                        prompt_id,
                        model_id,
                        anti.id,
                        metric="brand_legacy_conflation_rate",
                        value=mean_rate(conflation_samples),
                        aggregation="mean",
                    )
                )

    return out


def _score(
    run_id: str,
    prompt_id: str,
    model_id: str,
    subject_id: str,
    *,
    metric: str,
    value: bool | int | float | None,
    aggregation: str,
) -> Score:
    return Score(
        run_id=run_id,
        prompt_id=prompt_id,
        model_id=model_id,
        subject_id=subject_id,
        metric=metric,
        value=value,
        scoring_method="deterministic",
        sample_aggregation=cast(SampleAggregation, aggregation),
    )
