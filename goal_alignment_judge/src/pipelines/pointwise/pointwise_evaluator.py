from collections import Counter

from scipy import stats

from .pointwise_models import (
    HumanAction,
    PointwiseAccuracyResult,
    PointwiseClassMetrics,
    PointwiseEvaluationResult,
    ScoredPointwiseExample,
)
from ...settings import settings

# Three-class ground truth mapping
GROUND_TRUTH_CLASS = {
    HumanAction.PINNED: "pinned",
    HumanAction.NEUTRAL: "neutral",
    HumanAction.DELETED: "deleted",
    HumanAction.SYNTHETIC_NEGATIVE: "deleted",
}

CLASSES = ["pinned", "neutral", "deleted"]

ACTION_RANK = {
    HumanAction.PINNED: 3,
    HumanAction.NEUTRAL: 2,
    HumanAction.DELETED: 1,
    HumanAction.SYNTHETIC_NEGATIVE: 0,
}


def _ground_truth(action: HumanAction) -> str:
    return GROUND_TRUTH_CLASS[action]


def _predicted_class(sp: ScoredPointwiseExample, pinned_threshold: float = 0.4, deleted_threshold: float = 0.15) -> str:
    """Map alignment score to class using thresholds."""
    score = sp.judgment.alignment_score
    if score >= pinned_threshold:
        return "pinned"
    if score >= deleted_threshold:
        return "neutral"
    return "deleted"


def compute_three_class_accuracy(
    scored: list[ScoredPointwiseExample],
    pinned_threshold: float = 0.4,
    deleted_threshold: float = 0.15,
) -> PointwiseAccuracyResult:
    n = len(scored)

    # Build confusion matrix
    confusion: dict[str, dict[str, int]] = {gt: {pred: 0 for pred in CLASSES} for gt in CLASSES}
    for sp in scored:
        gt = _ground_truth(sp.example.human_action)
        pred = _predicted_class(sp, pinned_threshold=pinned_threshold, deleted_threshold=deleted_threshold)
        confusion[gt][pred] += 1

    n_correct = sum(confusion[c][c] for c in CLASSES)

    # Per-class precision, recall, F1
    per_class: dict[str, PointwiseClassMetrics] = {}
    for cls in CLASSES:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in CLASSES if other != cls)
        fn = sum(confusion[cls][other] for other in CLASSES if other != cls)
        support = sum(confusion[cls].values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = PointwiseClassMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            support=support,
        )

    # Macro F1 (unweighted average)
    class_f1s = [per_class[c].f1 for c in CLASSES if per_class[c].support > 0]
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0

    # Weighted F1 (weighted by support)
    total_support = sum(per_class[c].support for c in CLASSES)
    weighted_f1 = (
        sum(per_class[c].f1 * per_class[c].support for c in CLASSES) / total_support
        if total_support > 0 else 0.0
    )

    # Critical errors: pinned predicted as deleted, or deleted predicted as pinned
    critical = confusion["pinned"]["deleted"] + confusion["deleted"]["pinned"]

    return PointwiseAccuracyResult(
        accuracy=round(n_correct / n, 4) if n > 0 else 0.0,
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class=per_class,
        n_total=n,
        n_correct=n_correct,
        confusion=confusion,
        critical_errors=critical,
    )


def compute_score_correlation(scored: list[ScoredPointwiseExample]) -> float:
    human_ranks = [ACTION_RANK[sp.example.human_action] for sp in scored]
    pred_scores = [sp.judgment.alignment_score for sp in scored]

    if len(set(human_ranks)) < 2 or len(set(pred_scores)) < 2:
        return 0.0

    corr, _ = stats.spearmanr(human_ranks, pred_scores)
    return round(float(corr), 4)


def compute_baselines(scored: list[ScoredPointwiseExample]) -> dict[str, float]:
    n = len(scored)
    if n == 0:
        return {}

    gt_counts = Counter(_ground_truth(sp.example.human_action) for sp in scored)
    majority_count = gt_counts.most_common(1)[0][1]

    return {
        "random": round(1 / 3, 4),
        "majority_class": round(majority_count / n, 4),
        "always_neutral": round(gt_counts.get("neutral", 0) / n, 4),
    }


def compute_per_goal_accuracy(
    scored: list[ScoredPointwiseExample],
    pinned_threshold: float = 0.4,
    deleted_threshold: float = 0.15,
) -> dict[str, float]:
    by_goal: dict[str, list[ScoredPointwiseExample]] = {}
    for sp in scored:
        by_goal.setdefault(sp.example.goal_id, []).append(sp)

    result = {}
    for goal_id, items in sorted(by_goal.items()):
        correct = sum(
            1 for sp in items
            if _predicted_class(sp, pinned_threshold=pinned_threshold, deleted_threshold=deleted_threshold)
            == _ground_truth(sp.example.human_action)
        )
        result[goal_id] = round(correct / len(items), 4)

    return result


def find_optimal_thresholds(
    scored: list[ScoredPointwiseExample],
    pinned_range: tuple[float, float] = (0.25, 0.70),
    deleted_range: tuple[float, float] = (0.05, 0.35),
    step: float = 0.025,
) -> tuple[float, float, float]:
    """Grid search over (pinned_threshold, deleted_threshold) to maximize macro F1.

    Returns (best_pinned_threshold, best_deleted_threshold, best_macro_f1).
    """
    best_f1 = -1.0
    best_pt = 0.4
    best_dt = 0.15

    pt = pinned_range[0]
    while pt <= pinned_range[1]:
        dt = deleted_range[0]
        while dt <= deleted_range[1]:
            if dt >= pt:
                dt += step
                continue
            result = compute_three_class_accuracy(scored, pinned_threshold=pt, deleted_threshold=dt)
            if result.macro_f1 > best_f1:
                best_f1 = result.macro_f1
                best_pt = pt
                best_dt = dt
            dt += step
        pt += step

    return round(best_pt, 4), round(best_dt, 4), round(best_f1, 4)


def evaluate_pointwise(
    scored: list[ScoredPointwiseExample],
    split: str = "dev",
    pinned_threshold: float = 0.4,
    deleted_threshold: float = 0.15,
) -> PointwiseEvaluationResult:
    accuracy = compute_three_class_accuracy(scored, pinned_threshold=pinned_threshold, deleted_threshold=deleted_threshold)
    correlation = compute_score_correlation(scored)
    baselines = compute_baselines(scored)
    per_goal = compute_per_goal_accuracy(scored, pinned_threshold=pinned_threshold, deleted_threshold=deleted_threshold)

    result = PointwiseEvaluationResult(
        accuracy=accuracy,
        score_correlation=correlation,
        baselines=baselines,
        per_goal_accuracy=per_goal,
        n_samples=len(scored),
        split=split,
    )

    _print_results(result, scored)
    return result


def save_pointwise_results(result: PointwiseEvaluationResult, rubric_version: int | str, prefix: str | None = None) -> None:
    filename = f"{prefix}_pointwise_eval_{result.split}_v{rubric_version}.json" if prefix else f"pointwise_eval_{result.split}_v{rubric_version}.json"
    path = settings.results_path / filename
    path.write_text(result.model_dump_json(indent=2))


def meets_pointwise_targets(result: PointwiseEvaluationResult) -> bool:
    return result.accuracy.macro_f1 >= settings.target_pointwise_f1


def _print_results(result: PointwiseEvaluationResult, scored: list[ScoredPointwiseExample]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Pointwise Evaluation ({result.split}, n={result.n_samples})")
    print(f"{'=' * 60}")

    acc = result.accuracy
    target_met = lambda val, target: "PASS" if val >= target else "MISS"

    print(f"\n  Accuracy:         {acc.accuracy:.4f}")
    print(f"  Macro F1:         {acc.macro_f1:.4f}  "
          f"[{target_met(acc.macro_f1, settings.target_pointwise_f1)}  "
          f"target >= {settings.target_pointwise_f1}]")
    print(f"  Weighted F1:      {acc.weighted_f1:.4f}")
    print(f"  Critical errors:  {acc.critical_errors} (pinned↔deleted)")
    print(f"  Score correlation: {result.score_correlation:.4f} (Spearman)")

    # Per-class metrics
    print(f"\n  Per-class metrics:")
    print(f"    {'class':>12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for cls in CLASSES:
        m = acc.per_class.get(cls)
        if m:
            print(f"    {cls:>12} {m.precision:>10.4f} {m.recall:>10.4f} {m.f1:>10.4f} {m.support:>10}")

    print(f"\n  Baselines:")
    for name, val in result.baselines.items():
        print(f"    {name:>20}: {val:.4f}")

    # Confusion matrix
    print(f"\n  Confusion (ground truth \\ predicted):")
    header = f"    {'':>12}" + "".join(f"{c:>12}" for c in CLASSES)
    print(header)
    for gt in CLASSES:
        row = f"    {gt:>12}"
        for pred in CLASSES:
            row += f" {acc.confusion.get(gt, {}).get(pred, 0):>11}"
        print(row)

    # Signal strength by action
    print(f"\n  Predicted class by action:")
    for action in HumanAction:
        items = [sp for sp in scored if sp.example.human_action == action]
        if not items:
            continue
        pred_dist = Counter(_predicted_class(sp) for sp in items)
        print(f"    {action.value:>20}: {dict(pred_dist)}")

    # Score distribution by action
    print(f"\n  Alignment score by action:")
    for action in HumanAction:
        items = [sp for sp in scored if sp.example.human_action == action]
        if not items:
            continue
        scores = [sp.judgment.alignment_score for sp in items]
        print(f"    {action.value:>20}: mean={sum(scores)/len(scores):.3f}, "
              f"min={min(scores):.3f}, max={max(scores):.3f}")

    # Worst goals
    worst = sorted(result.per_goal_accuracy.items(), key=lambda x: x[1])[:5]
    if worst:
        print(f"\n  Worst 5 goals by accuracy:")
        for gid, acc_val in worst:
            n_items = sum(1 for sp in scored if sp.example.goal_id == gid)
            print(f"    {gid[:12]}...: {acc_val:.4f} (n={n_items})")
