import dspy

# Adjacent-class partial credit pairs and their scores.
# Critical errors (pinned<->deleted) get 0.0.
_ADJACENT_CREDIT = {
    ("pinned", "neutral"): 0.25,
    ("neutral", "pinned"): 0.25,
    ("neutral", "deleted"): 0.25,
    ("deleted", "neutral"): 0.25,
}


def _predicted_class_from_score(score: float) -> str:
    """Same thresholds as pointwise_evaluator._predicted_class."""
    if score >= 0.4:
        return "pinned"
    if score >= 0.15:
        return "neutral"
    return "deleted"


def _safe_score(prediction: dspy.Prediction) -> float:
    try:
        score = float(prediction.alignment_score)
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError, AttributeError):
        return 0.0


def macro_f1_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Per-example metric for MIPROv2.

    Returns 1.0 for correct class, 0.25 for adjacent-class errors,
    0.0 for critical errors (pinned<->deleted) or malformed output.
    """
    score = _safe_score(prediction)
    pred_class = _predicted_class_from_score(score)
    gt_class = example.ground_truth_class

    if pred_class == gt_class:
        return 1.0
    return _ADJACENT_CREDIT.get((gt_class, pred_class), 0.0)


def gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
):
    """GEPA metric that returns score + textual feedback for reflective optimization."""
    from gepa.adapters.dspy_adapter.dspy_adapter import ScoreWithFeedback

    score_val = _safe_score(pred)
    pred_class = _predicted_class_from_score(score_val)
    gt_class = gold.ground_truth_class

    if pred_class == gt_class:
        return ScoreWithFeedback(
            score=1.0,
            feedback=f"Correct: predicted {pred_class} (score={score_val:.2f}) matches ground truth.",
        )

    numeric_score = _ADJACENT_CREDIT.get((gt_class, pred_class), 0.0)

    # Build structured feedback for GEPA's reflective optimization
    direction = "over" if score_val > 0.4 and gt_class != "pinned" else "under"
    is_critical = (gt_class == "pinned" and pred_class == "deleted") or (
        gt_class == "deleted" and pred_class == "pinned"
    )

    feedback_parts = [
        f"WRONG: predicted {pred_class} (score={score_val:.2f}) but ground truth is {gt_class}.",
    ]

    if is_critical:
        feedback_parts.append(
            "CRITICAL ERROR: This is a pinned<->deleted swap — the worst kind of mistake. "
            "The scorer completely misjudged the content's relationship to the goal."
        )

    if direction == "over":
        feedback_parts.append(
            f"The scorer {direction}-estimated alignment. The content may be topically related "
            f"but lacks the directness or actionability that would make it truly aligned."
        )
    else:
        feedback_parts.append(
            f"The scorer {direction}-estimated alignment. The content may appear tangential "
            f"but actually represents meaningful progress toward the goal."
        )

    feedback_parts.append(
        f"Goal: '{gold.goal_title}'. Content type: {gold.content_type}, title: '{gold.content_title[:80]}'."
    )

    return ScoreWithFeedback(
        score=numeric_score,
        feedback=" ".join(feedback_parts),
    )
