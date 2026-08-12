"""Trimmed mean aggregation across the 9-judge ensemble."""
import statistics
from typing import Literal

from .rubric import CellAggregate, JudgeRun

_DIMENSIONS = ["coverage", "fidelity", "synthesis_quality", "interpretability", "information_density"]


def _trimmed_mean(values: list[float]) -> float:
    """Drop the single highest and lowest value, average the rest."""
    if len(values) <= 2:
        return statistics.mean(values) if values else 0.0
    sorted_vals = sorted(values)
    trimmed = sorted_vals[1:-1]  # drop min and max
    return statistics.mean(trimmed)


def aggregate(
    judge_runs: list[JudgeRun],
    condition_name: Literal["unstated", "stated", "mixed"],
    schema_name: Literal["dm", "dsm", "gm"],
) -> CellAggregate:
    """Compute aggregated statistics from 9 judge runs."""
    if not judge_runs:
        raise ValueError("Cannot aggregate zero judge runs")

    overalls = [r.score.self_reported_overall for r in judge_runs]
    trimmed = _trimmed_mean([float(o) for o in overalls])

    # Per-dimension mean
    per_dim: dict[str, float] = {}
    for dim in _DIMENSIONS:
        scores = [float(getattr(r.score, dim)) for r in judge_runs]
        per_dim[dim] = round(statistics.mean(scores), 2)

    # Inter-judge variance of self_reported_overall
    variance = round(statistics.variance([float(o) for o in overalls]), 2) if len(overalls) > 1 else 0.0

    # Model decomposition: mean overall per model_id
    model_scores: dict[str, list[float]] = {}
    for r in judge_runs:
        model_scores.setdefault(r.model_id, []).append(float(r.score.self_reported_overall))
    model_decomp = {mid: round(statistics.mean(vals), 2) for mid, vals in model_scores.items()}

    # Role decomposition: mean overall per role
    role_scores: dict[str, list[float]] = {}
    for r in judge_runs:
        role_scores.setdefault(r.role, []).append(float(r.score.self_reported_overall))
    role_decomp = {role: round(statistics.mean(vals), 2) for role, vals in role_scores.items()}

    return CellAggregate(
        cell_id=judge_runs[0].cell_id,
        condition_name=condition_name,
        schema_name=schema_name,
        judge_runs=judge_runs,
        trimmed_mean_overall=round(trimmed, 2),
        per_dimension_mean=per_dim,
        inter_judge_variance=variance,
        model_decomposition=model_decomp,
        role_decomposition=role_decomp,
    )


def flag_divergent_runs(runs: list[JudgeRun], threshold: int = 1) -> list[JudgeRun]:
    """Return judge runs where self_reported_overall diverges from sum_dims by > threshold."""
    flagged = []
    for r in runs:
        s = r.score
        sum_dims = s.coverage + s.fidelity + s.synthesis_quality + s.interpretability + s.information_density
        if abs(s.self_reported_overall - sum_dims) > threshold:
            flagged.append(r)
    return flagged
