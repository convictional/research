"""
Priority Generator using Gemini Flash.

This is the LLM being optimized - it takes an instruction and context,
then generates priority predictions.
"""

import instructor
from datetime import datetime
from pydantic import BaseModel, Field

from .models import Context, Prediction, Priority
from .config import config


class PriorityResponse(BaseModel):
    """Structured response from the Generator."""

    reasoning: str = Field(
        ...,
        description="Your thinking process - what patterns did you notice? What seems urgent? What dependencies exist?",
    )
    priority_1: str = Field(..., description="Highest priority task")
    priority_2: str = Field(..., description="Second priority task")
    priority_3: str = Field(..., description="Third priority task")
    priority_4: str = Field(..., description="Fourth priority task")
    priority_5: str = Field(..., description="Fifth priority task")


class PriorityGenerator:
    """
    Generate priority predictions using Gemini Flash with Instructor.

    This is LLM 1 in the OPRO architecture - the model being optimized.
    """

    def __init__(self, api_key: str):
        model = config.get("models.generator_model", "gemini-2.0-flash")
        # Patch Gemini client with Instructor for structured responses
        self.client = instructor.from_provider(
            f"google/{model}",
            api_key=api_key,
            async_client=True,
        )

    async def generate(
        self,
        instruction: str,
        context: Context,
    ) -> Prediction:
        """
        Generate priority predictions given instruction and context.

        Args:
            instruction: The optimized instruction/prompt to use
            context: Historical context (emails, meetings, tasks, discussions)

        Returns:
            Prediction with ranked priorities and reasoning
        """
        # Format context for the prompt
        context_text = self._format_context(context)

        # Build the full prompt
        prompt = f"""{instruction}

TARGET DATE: {context.target_date.strftime('%Y-%m-%d (%A)')}

AVAILABLE CONTEXT:

{context_text}

Based on the context above, predict the top 5 work priorities for {context.target_date.strftime('%B %d, %Y')}.
"""

        # Generate with Gemini Flash + Instructor
        response = await self.client.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=PriorityResponse,
            temperature=config.get("generation.temperature_generator", 0.7),
            max_tokens=config.get("generation.max_tokens_generator", 1024),
        )

        # Convert to our Prediction model
        priorities = [
            Priority(description=response.priority_1, rank=1),
            Priority(description=response.priority_2, rank=2),
            Priority(description=response.priority_3, rank=3),
            Priority(description=response.priority_4, rank=4),
            Priority(description=response.priority_5, rank=5),
        ]

        return Prediction(
            priorities=priorities,
            reasoning=response.reasoning,
            instruction_used=instruction,
            model=config.get("models.generator_model", "gemini-2.0-flash"),
            timestamp=datetime.utcnow(),
        )

    def _format_context(self, context: Context) -> str:
        """Format context items into readable text."""
        sections = []

        if context.emails:
            sections.append("EMAILS:")
            for item in context.emails[:5]:  # Limit to top 5
                sections.append(
                    f"- [{item.created_at.strftime('%b %d')}] {item.title}\n  {item.content[:150]}..."
                )
            sections.append("")

        if context.meetings:
            sections.append("MEETINGS/CALENDAR:")
            for item in context.meetings[:5]:
                sections.append(
                    f"- [{item.created_at.strftime('%b %d')}] {item.title}\n  {item.content[:150]}..."
                )
            sections.append("")

        if context.tasks:
            sections.append("TASKS/ISSUES:")
            for item in context.tasks[:10]:  # More tasks since they're common
                sections.append(
                    f"- [{item.created_at.strftime('%b %d')}] {item.title}\n  {item.content[:150]}..."
                )
            sections.append("")

        if context.discussions:
            sections.append("DISCUSSIONS/COMMENTS:")
            for item in context.discussions[:10]:
                sections.append(
                    f"- [{item.created_at.strftime('%b %d')}] {item.title}\n  {item.content[:150]}..."
                )
            sections.append("")

        if not sections:
            return "No relevant context found."

        return "\n".join(sections)
