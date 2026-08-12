from collections import Counter

from scipy.stats import spearmanr

from src.models import EvaluationResult, ScoredReport
from src.settings import settings, logger


def evaluate(scored_reports: list[ScoredReport], split: str = "dev") -> EvaluationResult:
    expert_scores = [r.report.quality_score for r in scored_reports]
    predicted_scores = [r.predicted_score.quality_score for r in scored_reports]
    n = len(scored_reports)

    # Spearman rank correlation (integer scores)
    if len(set(expert_scores)) > 1 and len(set(predicted_scores)) > 1:
        spearman_corr, _ = spearmanr(expert_scores, predicted_scores)
    else:
        spearman_corr = 0.0

    # Spearman on continuous ensemble mean scores when available
    continuous_scores = [r.continuous_score for r in scored_reports]
    has_continuous = all(s is not None for s in continuous_scores)
    spearman_continuous = None
    if has_continuous:
        continuous_vals = [s for s in continuous_scores]
        if len(set(expert_scores)) > 1 and len(set(continuous_vals)) > 1:
            spearman_continuous, _ = spearmanr(expert_scores, continuous_vals)
        else:
            spearman_continuous = 0.0

    # Mean Absolute Error
    mae = sum(abs(e - p) for e, p in zip(expert_scores, predicted_scores)) / n

    # Exact match rate
    exact_matches = sum(1 for e, p in zip(expert_scores, predicted_scores) if e == p)
    exact_match_rate = exact_matches / n

    # Adjacent match rate (within +/- 1)
    adjacent_matches = sum(1 for e, p in zip(expert_scores, predicted_scores) if abs(e - p) <= 1)
    adjacent_match_rate = adjacent_matches / n

    # Per-score-level accuracy
    per_score: dict[str, float] = {}
    score_counts = Counter(expert_scores)
    for score in sorted(score_counts.keys()):
        correct = sum(
            1 for e, p in zip(expert_scores, predicted_scores) if e == score and p == score
        )
        per_score[str(score)] = correct / score_counts[score] if score_counts[score] > 0 else 0.0

    result = EvaluationResult(
        spearman_correlation=round(spearman_corr, 4),
        spearman_continuous=round(spearman_continuous, 4) if spearman_continuous is not None else None,
        mae=round(mae, 4),
        exact_match_rate=round(exact_match_rate, 4),
        adjacent_match_rate=round(adjacent_match_rate, 4),
        per_score_accuracy=per_score,
        n_samples=n,
        split=split,
    )

    _print_results(result, expert_scores, predicted_scores)
    return result


def save_results(result: EvaluationResult, rubric_version: int | str) -> None:
    path = settings.results_path / f"eval_{result.split}_v{rubric_version}.json"
    path.write_text(result.model_dump_json(indent=2))
    logger.info(f"Saved evaluation results to {path}")


def meets_targets(result: EvaluationResult) -> bool:
    # Use continuous Spearman if available (more sensitive from ensemble)
    spearman = result.spearman_continuous if result.spearman_continuous is not None else result.spearman_correlation
    return (
        spearman >= settings.target_spearman
        and result.mae <= settings.target_mae
        and result.adjacent_match_rate >= settings.target_adjacent_match
    )


def _print_results(
    result: EvaluationResult,
    expert_scores: list[int],
    predicted_scores: list[int],
) -> None:
    print(f"\n{'='*60}")
    print(f"  Evaluation Results ({result.split}, n={result.n_samples})")
    print(f"{'='*60}")

    target_met = lambda val, target, higher_better=True: (
        "PASS" if (val >= target if higher_better else val <= target) else "MISS"
    )

    print(f"\n  Spearman (integer):   {result.spearman_correlation:.4f}  "
          f"[{target_met(result.spearman_correlation, settings.target_spearman)}  target >= {settings.target_spearman}]")
    if result.spearman_continuous is not None:
        print(f"  Spearman (continuous):{result.spearman_continuous:.4f}  "
              f"[{target_met(result.spearman_continuous, settings.target_spearman)}  target >= {settings.target_spearman}]")
    print(f"  MAE:                  {result.mae:.4f}  "
          f"[{target_met(result.mae, settings.target_mae, higher_better=False)}  target <= {settings.target_mae}]")
    print(f"  Exact match rate:     {result.exact_match_rate:.4f}")
    print(f"  Adjacent match rate:  {result.adjacent_match_rate:.4f}  "
          f"[{target_met(result.adjacent_match_rate, settings.target_adjacent_match)}  target >= {settings.target_adjacent_match}]")

    print(f"\n  Per-score accuracy:")
    for score, acc in sorted(result.per_score_accuracy.items()):
        print(f"    Score {score}: {acc:.2%}")

    # Confusion-style summary
    print(f"\n  Score distribution (expert -> predicted):")
    for score in range(4):
        expert_count = sum(1 for s in expert_scores if s == score)
        predicted_count = sum(1 for s in predicted_scores if s == score)
        print(f"    Score {score}: expert={expert_count:>3}, predicted={predicted_count:>3}")
