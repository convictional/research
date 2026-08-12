"""DSPy signatures for priority prediction and evaluation."""

import dspy


class PrioritySignature(dspy.Signature):
    """Predict a user's top work priorities from historical context.

    Given emails, meetings, tasks, and discussions from before the target date,
    predict the 5 most likely priorities the user will focus on, ranked by
    importance. Focus on evidence of actual work being performed, not just
    planned items or discussions.
    """

    context: str = dspy.InputField(
        desc="Historical content including emails, meetings, tasks, and discussions"
    )
    target_date: str = dspy.InputField(desc="The date to predict priorities for (YYYY-MM-DD)")

    reasoning: str = dspy.OutputField(
        desc="Analysis of the context: what patterns suggest priority? What evidence of active work exists?"
    )
    priority_1: str = dspy.OutputField(desc="Highest priority - the main focus for the day")
    priority_2: str = dspy.OutputField(desc="Second priority")
    priority_3: str = dspy.OutputField(desc="Third priority")
    priority_4: str = dspy.OutputField(desc="Fourth priority")
    priority_5: str = dspy.OutputField(desc="Fifth priority")


class JudgeSignature(dspy.Signature):
    """Evaluate how well predicted priorities align with what was actually worked on.

    Score the prediction on recall (are ground truth items in predictions?),
    completeness (what % captured?), ordering (are they ranked high?), and
    context usage (were predictions derived from context?).
    """

    predicted_priorities: str = dspy.InputField(
        desc="The 5 predicted priorities, numbered 1-5"
    )
    ground_truth: str = dspy.InputField(
        desc="What was actually worked on (from standup)"
    )
    context_summary: str = dspy.InputField(
        desc="Summary of available context (counts of emails, meetings, tasks, discussions)"
    )

    correctness_score: int = dspy.OutputField(
        desc="0-10: Are ground truth items included in predictions? 10=ALL items present"
    )
    completeness_score: int = dspy.OutputField(
        desc="0-10: What percentage of ground truth items are captured? 10=every item"
    )
    ordering_score: int = dspy.OutputField(
        desc="0-10: Are ground truth items ranked highly (top 1-3)? 10=all in top positions"
    )
    context_usage_score: int = dspy.OutputField(
        desc="0-10: Were predictions clearly derived from context? 10=clear derivation"
    )
    reasoning: str = dspy.OutputField(
        desc="Detailed explanation of the scores, specific issues, and suggestions"
    )
