import dspy

from .dspy_signatures import GoalAlignmentSignature


class GoalAlignmentScorer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(GoalAlignmentSignature)

    def forward(
        self,
        goal_title: str,
        goal_description: str,
        content_type: str,
        content_title: str,
        content_body: str,
    ) -> dspy.Prediction:
        return self.assess(
            goal_title=goal_title,
            goal_description=goal_description,
            content_type=content_type,
            content_title=content_title,
            content_body=content_body,
        )
