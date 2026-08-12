"""
Alignment Judge using Gemini Pro.

This LLM evaluates how well predictions align with ground truth,
focusing on recall (did it include what was worked on?) and ranking
(are the most important items ranked highest?).
"""

import instructor
from datetime import datetime

from .models import Context, Prediction, StandupEntry, JudgeScore
from .config import config


class AlignmentJudge:
    """
    Evaluate prediction alignment using Gemini Pro with Instructor.

    This is LLM 2 in the OPRO architecture - provides the reward signal.

    Key metrics:
    - Recall: What % of ground truth items appear in predictions?
    - Ranking: Are ground truth items ranked high (top positions)?
    - Precision: Are predicted items actually relevant?
    """

    def __init__(self, api_key: str):
        model = config.get("models.judge_model", "gemini-2.0-pro")
        # Patch Gemini client with Instructor for structured responses
        self.client = instructor.from_provider(
            f"google/{model}",
            api_key=api_key,
            async_client=True,
        )

    async def evaluate(
        self,
        prediction: Prediction,
        ground_truth: StandupEntry,
        context: Context,
    ) -> JudgeScore:
        """
        Evaluate how well the prediction aligns with ground truth.

        Args:
            prediction: Generated priority predictions (typically 5 items)
            ground_truth: What was actually worked on (from standup)
            context: Historical context that was available

        Returns:
            JudgeScore with detailed evaluation focused on recall and ranking
        """
        # Format prediction and ground truth
        pred_text = self._format_prediction(prediction)
        truth_text = self._format_ground_truth(ground_truth)
        context_summary = self._summarize_context(context)

        # Build evaluation prompt
        prompt = f"""You are an expert evaluator assessing priority predictions.

PREDICTED PRIORITIES (what the LLM recommended):
{pred_text}

GROUND TRUTH (what was actually worked on):
{truth_text}

AVAILABLE CONTEXT:
{context_summary}

Evaluate the prediction on these criteria (each 0-10):

1. **Correctness** (Recall): Are the ground truth items included in the predictions?
   - 10: ALL ground truth items are in the predictions
   - 7-9: Most ground truth items are included
   - 4-6: Some ground truth items are included
   - 0-3: Few or no ground truth items are included

   Focus: Did the LLM predict what was actually worked on?

2. **Completeness** (Coverage): How many of the ground truth priorities are captured?
   - 10: Every single ground truth item appears in predictions
   - 5: About half of ground truth items appear
   - 0: None of the ground truth items appear

3. **Ordering** (Ranking Quality): Are ground truth items ranked highly?
   - 10: All ground truth items are in top positions (ranks 1-3)
   - 7-9: Ground truth items are mostly high-ranked
   - 4-6: Ground truth items are scattered in rankings
   - 0-3: Ground truth items are ranked low or missing

   Example: If ground truth has 2 items and they're ranked #1 and #2 in predictions → score 10
   Example: If ground truth has 2 items and they're ranked #3 and #5 in predictions → score 6

4. **Context Usage**: Did the model use available context effectively?
   - 10: Predictions clearly derived from relevant context
   - 7-9: Good use of context
   - 4-6: Some context usage
   - 0-3: Poor context usage or hallucinated items

Calculate overall_score as: (correctness + completeness + ordering + context_usage) * 2.5

IMPORTANT: The LLM always predicts 5 items, but ground truth may have fewer.
Focus on whether ground truth items appear in predictions and are ranked high,
NOT on whether all 5 predictions match ground truth exactly.

Identify specific issues and provide actionable suggestions.
"""

        # Get structured evaluation
        response = await self.client.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=JudgeScore,
            temperature=config.get("generation.temperature_judge", 0.3),
            max_tokens=config.get("generation.max_tokens_judge", 2048),
        )

        # Add timestamp
        response.timestamp = datetime.utcnow()

        return response

    def _format_prediction(self, prediction: Prediction) -> str:
        """Format prediction for evaluation."""
        lines = ["Reasoning:", prediction.reasoning, "", "Predicted Priorities:"]
        for priority in prediction.priorities:
            lines.append(f"{priority.rank}. {priority.description}")
        return "\n".join(lines)

    def _format_ground_truth(self, ground_truth: StandupEntry) -> str:
        """Format ground truth for evaluation."""
        lines = ["What was actually worked on:"]
        for priority in ground_truth.priorities:
            lines.append(f"- {priority.description}")

        if ground_truth.context_signals:
            lines.append("\nAdditional Context from Standup:")
            lines.append(ground_truth.context_signals)

        return "\n".join(lines)

    def _summarize_context(self, context: Context) -> str:
        """Summarize available context."""
        return f"""- {len(context.emails)} emails
- {len(context.meetings)} meetings/calendar events
- {len(context.tasks)} tasks/issues
- {len(context.discussions)} discussions/comments
Total: {context.total_items} items available before target date"""
