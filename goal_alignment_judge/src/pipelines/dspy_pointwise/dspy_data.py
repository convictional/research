import dspy

from ..pointwise.pointwise_models import (
    HumanAction,
    PointwiseExample,
    PointwiseJudgment,
    ScoredPointwiseExample,
)

GROUND_TRUTH_CLASS = {
    HumanAction.PINNED: "pinned",
    HumanAction.NEUTRAL: "neutral",
    HumanAction.DELETED: "deleted",
    HumanAction.SYNTHETIC_NEGATIVE: "deleted",
}

CONTENT_BODY_MAX_CHARS = 3000


def pointwise_to_dspy(example: PointwiseExample) -> dspy.Example:
    """Convert PointwiseExample to dspy.Example with inputs and labels."""
    gt_class = GROUND_TRUTH_CLASS[example.human_action]

    return dspy.Example(
        goal_title=example.goal_title,
        goal_description=example.goal_description,
        content_type=example.content_type,
        content_title=example.content_title,
        content_body=example.content_body[:CONTENT_BODY_MAX_CHARS],
        human_action=example.human_action.value,
        ground_truth_class=gt_class,
    ).with_inputs("goal_title", "goal_description", "content_type", "content_title", "content_body")


def prediction_to_judgment(prediction: dspy.Prediction) -> PointwiseJudgment:
    """Convert dspy.Prediction to PointwiseJudgment with defensive parsing."""
    try:
        score = float(prediction.alignment_score)
        score = max(0.0, min(1.0, score))
    except (ValueError, TypeError, AttributeError):
        score = 0.0

    try:
        is_aligned = bool(prediction.is_aligned)
    except (ValueError, TypeError, AttributeError):
        is_aligned = score >= 0.3

    signal_raw = str(getattr(prediction, "signal", "none")).lower().strip()
    signal = signal_raw if signal_raw in ("strong", "medium", "weak") else None

    reasoning = str(getattr(prediction, "reasoning", ""))

    return PointwiseJudgment(
        is_aligned=is_aligned,
        signal=signal,
        alignment_score=round(score, 4),
        description=f"DSPy prediction (signal={signal})",
        reasoning=reasoning,
    )


def prediction_to_scored(example: PointwiseExample, prediction: dspy.Prediction) -> ScoredPointwiseExample:
    """Convert PointwiseExample + dspy.Prediction into ScoredPointwiseExample."""
    judgment = prediction_to_judgment(prediction)
    return ScoredPointwiseExample(example=example, judgment=judgment)
