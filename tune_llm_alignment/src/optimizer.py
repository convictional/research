"""
OPRO Optimizer using Gemini Pro.

This LLM analyzes the optimization trajectory and judge feedback,
then proposes improved instructions for the Generator.
"""

from google import genai
from google.genai import types
from typing import List, Tuple
from pydantic import BaseModel, Field

from .config import config


class InstructionProposal(BaseModel):
    """Proposed instruction improvement from the Optimizer."""

    instruction: str = Field(
        ...,
        description="The new instruction that should improve Generator performance",
    )
    rationale: str = Field(
        ...,
        description="Why this instruction should perform better than previous ones",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this proposal (0-1)",
    )


class OPROOptimizer:
    """
    Meta-agent that proposes instruction improvements using OPRO.

    This is LLM 3 in the OPRO architecture - analyzes failures and
    generates better prompts for the Generator.
    """

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = config.get("models.optimizer_model", "gemini-2.0-pro")

    async def generate_candidates(
        self,
        trajectory: List[Tuple[str, float]],
        exemplars: List[dict],
        n: int = 8,
    ) -> List[str]:
        """
        Generate n candidate instructions based on optimization trajectory.

        Args:
            trajectory: List of (instruction, score) pairs sorted by score ascending
            exemplars: Example tasks showing how instructions are used
            n: Number of candidates to generate

        Returns:
            List of candidate instructions
        """
        # Build OPRO meta-prompt
        meta_prompt = self._build_meta_prompt(trajectory, exemplars)

        # Generate multiple candidates
        candidates = []
        for i in range(n):
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=meta_prompt,
                config=types.GenerateContentConfig(
                    temperature=config.get("generation.temperature_optimizer", 1.0),
                    max_output_tokens=config.get("generation.max_tokens_optimizer", 512),
                ),
            )

            # Extract instruction from response
            instruction = self._extract_instruction(response.text)
            if instruction and len(instruction) > 20:  # Basic validation
                candidates.append(instruction)

        return candidates

    def _build_meta_prompt(
        self,
        trajectory: List[Tuple[str, float]],
        exemplars: List[dict],
    ) -> str:
        """
        Build OPRO meta-prompt showing trajectory and task exemplars.

        Following the OPRO pattern from Google DeepMind research.
        """
        # Sort trajectory by score (ascending) and keep best 20
        sorted_traj = sorted(trajectory, key=lambda x: x[1])
        if len(sorted_traj) > 20:
            sorted_traj = sorted_traj[-20:]

        best_score = sorted_traj[-1][1] if sorted_traj else 0.0

        prompt = f"""# OPTIMIZATION OBJECTIVE
Your goal is to generate a new instruction for an AI model that predicts daily work priorities.
The instruction should help the model better align with what a person actually works on.

# OPTIMIZATION TRAJECTORY
Below are previously generated instructions with their alignment scores (0-100).
Higher scores indicate better alignment with actual work priorities.

"""

        # Add trajectory
        for instruction, score in sorted_traj:
            prompt += f"instruction: {instruction}\nscore: {score:.1f}\n\n"

        prompt += f"""# TASK EXEMPLARS
Here are examples showing how the instruction will be used by the Generator model.
The <INSTRUCTION> will be provided to the model along with historical context.

"""

        # Add 3 exemplars
        for i, ex in enumerate(exemplars[:3], 1):
            prompt += f"""Example {i}:
Target Date: {ex.get('date', 'Unknown')}
Context Available: {ex.get('context_summary', 'emails, meetings, tasks, discussions')}
Ground Truth: {ex.get('ground_truth_preview', 'User worked on specific tasks')}

"""

        prompt += f"""# GENERATION INSTRUCTIONS
Based on the optimization trajectory above, generate ONE new instruction that is likely to
achieve a score higher than {best_score:.1f}.

Key insights from the trajectory:
- High-scoring instructions focus on recall and ranking of actual work items
- The model should prioritize finding items in context that match what was worked on
- Ranking matters: predicted items should be ordered by importance/urgency

The instruction should:
- Be clear and concise (2-4 sentences)
- Build on patterns from high-scoring instructions
- Avoid patterns from low-scoring instructions
- Help the model identify and rank actual priorities from context
- Focus on recall (including what was worked on) and ranking quality

Output only the new instruction, no additional explanation:

New instruction:"""

        return prompt

    def _extract_instruction(self, response_text: str) -> str:
        """Extract instruction from optimizer response."""
        # Remove common prefixes/labels
        text = response_text.strip()

        # Remove "New instruction:" prefix if present
        if text.lower().startswith("new instruction:"):
            text = text[len("new instruction:") :].strip()

        # Remove quotes if wrapped
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        return text.strip()
