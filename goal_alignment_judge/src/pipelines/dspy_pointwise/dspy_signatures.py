import dspy


class GoalAlignmentSignature(dspy.Signature):
    """Assess whether a piece of content is aligned with an organizational goal.

    Evaluate the content against the goal, considering how directly it advances
    the goal, whether it represents actionable progress or is merely tangential,
    and the strength of the connection. Most content surfaced is already topically
    relevant — the key judgment is whether it is specific, actionable, and directly
    advances the goal versus being loosely related.

    Score from 0.0 (not aligned) to 1.0 (strongly aligned):
    - 0.6-1.0: Strong alignment — content directly advances the goal with specific, actionable information
    - 0.4-0.59: Medium alignment — content is relevant and useful but not directly actionable
    - 0.15-0.39: Weak alignment — content is tangentially related but unlikely to drive progress
    - 0.0-0.14: Not aligned — content has no meaningful connection to the goal
    """

    goal_title: str = dspy.InputField(desc="Title of the organizational goal")
    goal_description: str = dspy.InputField(desc="Description of the goal")
    content_type: str = dspy.InputField(desc="Type of content (e.g., post, meeting, email)")
    content_title: str = dspy.InputField(desc="Title of the content item")
    content_body: str = dspy.InputField(desc="Body text of the content item")

    reasoning: str = dspy.OutputField(desc="Step-by-step analysis of how the content relates to the goal")
    is_aligned: bool = dspy.OutputField(desc="Whether the content is aligned with the goal")
    signal: str = dspy.OutputField(desc="Signal strength: 'strong', 'medium', 'weak', or 'none'")
    alignment_score: float = dspy.OutputField(desc="Alignment score from 0.0 (not aligned) to 1.0 (strongly aligned)")
