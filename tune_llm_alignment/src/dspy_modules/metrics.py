"""Metric functions for evaluating priority predictions."""

import dspy

from .signatures import JudgeSignature


def alignment_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Score prediction alignment using LLM judge.

    Evaluates how well predicted team priorities match actual team work.
    The judge evaluates on:
    - Correctness (recall): Are ground truth items in predictions?
    - Completeness: What % of ground truth items are captured?
    - Ordering: Are ground truth items ranked highly?
    - Context usage: Were predictions derived from context?

    Args:
        example: DSPy Example with ground_truth (list of team priorities) and context_summary
        prediction: DSPy Prediction with priority_1 through priority_5
        trace: Optional trace for debugging (not used)

    Returns:
        Float score between 0 and 1 (normalized from 0-100 scale)
    """
    # Format predicted priorities
    predicted_str = "\n".join([
        f"1. {prediction.priority_1}",
        f"2. {prediction.priority_2}",
        f"3. {prediction.priority_3}",
        f"4. {prediction.priority_4}",
        f"5. {prediction.priority_5}",
    ])

    # Format ground truth - for team data, this is multiple people's priorities
    ground_truth_str = "\n".join([f"- {item}" for item in example.ground_truth])

    # Use judge signature to evaluate
    judge = dspy.Predict(JudgeSignature)

    try:
        result = judge(
            predicted_priorities=predicted_str,
            ground_truth=ground_truth_str,
            context_summary=example.context_summary,
        )

        # Calculate overall score (same formula as OPRO judge)
        overall_score = (
            result.correctness_score
            + result.completeness_score
            + result.ordering_score
            + result.context_usage_score
        ) * 2.5  # Scale to 0-100

        # Normalize to 0-1 for DSPy
        return overall_score / 100.0

    except Exception as e:
        print(f"Judge error: {e}")
        return 0.0


def team_recall_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Simpler metric: what fraction of team priorities were predicted?

    Uses LLM to check semantic overlap between predictions and ground truth.
    More lenient than exact match - looks for topical alignment.

    Args:
        example: DSPy Example with ground_truth (list of team priorities)
        prediction: DSPy Prediction with priority_1 through priority_5
        trace: Optional trace for debugging

    Returns:
        Float score between 0 and 1 representing recall
    """

    class RecallJudge(dspy.Signature):
        """Count how many ground truth items are covered by predictions."""

        predictions: str = dspy.InputField(desc="5 predicted priorities")
        ground_truth: str = dspy.InputField(desc="Actual team priorities")

        covered_count: int = dspy.OutputField(
            desc="Number of ground truth items that are semantically covered by predictions"
        )
        total_count: int = dspy.OutputField(desc="Total number of ground truth items")
        reasoning: str = dspy.OutputField(desc="Brief explanation of which items matched")

    # Format predictions
    predicted_str = "\n".join([
        f"1. {prediction.priority_1}",
        f"2. {prediction.priority_2}",
        f"3. {prediction.priority_3}",
        f"4. {prediction.priority_4}",
        f"5. {prediction.priority_5}",
    ])

    ground_truth_str = "\n".join([f"- {item}" for item in example.ground_truth])

    judge = dspy.Predict(RecallJudge)

    try:
        result = judge(predictions=predicted_str, ground_truth=ground_truth_str)
        recall = result.covered_count / max(result.total_count, 1)
        return min(recall, 1.0)  # Cap at 1.0
    except Exception as e:
        print(f"Recall judge error: {e}")
        return 0.0
