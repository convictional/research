"""DSPy module for priority prediction."""

import dspy

from .signatures import PrioritySignature


class PriorityPredictor(dspy.Module):
    """Predicts daily work priorities from historical context.

    Uses Chain of Thought prompting to reason through the context
    before generating priority predictions.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(PrioritySignature)

    def forward(self, context: str, target_date: str) -> dspy.Prediction:
        """Generate priority predictions.

        Args:
            context: Formatted historical context (emails, meetings, tasks, discussions)
            target_date: Date to predict priorities for (YYYY-MM-DD format)

        Returns:
            DSPy Prediction with reasoning and 5 ranked priorities
        """
        return self.predict(context=context, target_date=target_date)


class PriorityPredictorSimple(dspy.Module):
    """Simpler predictor without Chain of Thought.

    Use this for faster iteration or if CoT doesn't help.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PrioritySignature)

    def forward(self, context: str, target_date: str) -> dspy.Prediction:
        return self.predict(context=context, target_date=target_date)
