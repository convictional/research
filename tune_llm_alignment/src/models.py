"""
Pydantic models for the LLM alignment experiment.

Defines data structures for:
- Standup entries (ground truth)
- Context (emails, meetings, tasks, discussions)
- Predictions (generated priorities)
- Judge scores (evaluation results)
- Optimization trajectory
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Priority(BaseModel):
    """A single priority item with description and rationale."""

    description: str = Field(..., description="Description of the priority task")
    rationale: Optional[str] = Field(None, description="Why this is a priority")
    rank: Optional[int] = Field(None, description="Priority ranking (1=highest)")


class StandupEntry(BaseModel):
    """Ground truth data extracted from standup documents."""

    date: datetime = Field(..., description="Date of the standup entry")
    priorities: List[Priority] = Field(..., description="What was actually worked on")
    context_signals: Optional[str] = Field(
        None, description="Any additional context mentioned in standup"
    )
    raw_text: Optional[str] = Field(None, description="Original standup text")


class ContentItem(BaseModel):
    """A single piece of content from the database."""

    id: str
    type: str = Field(..., description="Type: email, meeting, task, discussion")
    title: Optional[str] = None
    content: str
    created_at: datetime
    relevance_score: Optional[float] = Field(
        None, description="Relevance score from hybrid search"
    )


class Context(BaseModel):
    """Historical context available at a given date."""

    target_date: datetime = Field(..., description="Date we're predicting for")
    emails: List[ContentItem] = Field(default_factory=list)
    meetings: List[ContentItem] = Field(default_factory=list)
    tasks: List[ContentItem] = Field(default_factory=list)
    discussions: List[ContentItem] = Field(default_factory=list)

    @property
    def all_items(self) -> List[ContentItem]:
        """Get all content items across all types."""
        return self.emails + self.meetings + self.tasks + self.discussions

    @property
    def total_items(self) -> int:
        """Total number of content items."""
        return len(self.all_items)


class Prediction(BaseModel):
    """LLM-generated priority prediction."""

    priorities: List[Priority] = Field(..., description="Predicted priorities")
    reasoning: Optional[str] = Field(None, description="LLM's reasoning process")
    instruction_used: str = Field(..., description="Instruction prompt that was used")
    model: str = Field(..., description="Model used for generation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class JudgeScore(BaseModel):
    """Evaluation score from the judge LLM."""

    correctness_score: float = Field(
        ..., ge=0, le=10, description="Are predicted items correct?"
    )
    completeness_score: float = Field(
        ..., ge=0, le=10, description="Are all important items included?"
    )
    ordering_score: float = Field(
        ..., ge=0, le=10, description="Is priority ranking correct?"
    )
    context_usage_score: float = Field(
        ..., ge=0, le=10, description="Did model use context effectively?"
    )
    overall_score: float = Field(
        ..., ge=0, le=100, description="Overall alignment score (0-100)"
    )

    reasoning: str = Field(..., description="Detailed explanation of scores")
    specific_issues: List[str] = Field(
        default_factory=list, description="Specific problems identified"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def average_criteria_score(self) -> float:
        """Average of the four criteria scores (0-10 scale)."""
        return (
            self.correctness_score
            + self.completeness_score
            + self.ordering_score
            + self.context_usage_score
        ) / 4.0


class TrajectoryPoint(BaseModel):
    """A single point in the optimization trajectory."""

    iteration: int
    instruction: str
    score: float = Field(..., description="Average score on evaluation set")
    candidate_index: int = Field(..., description="Which candidate in the batch")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExampleData(BaseModel):
    """Complete dataset example with context and ground truth."""

    id: str = Field(..., description="Unique identifier for this example")
    standup_entry: StandupEntry
    context: Context
    split: str = Field(..., description="train, dev, or test")


class OptimizationResult(BaseModel):
    """Result of a complete optimization run."""

    best_instruction: str
    best_score: float
    trajectory: List[TrajectoryPoint]
    total_iterations: int
    stopping_reason: str = Field(..., description="Why optimization stopped")
    start_time: datetime
    end_time: datetime

    @property
    def duration_seconds(self) -> float:
        """Duration of optimization in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def improvement(self) -> float:
        """Improvement from first to best score."""
        if not self.trajectory:
            return 0.0
        initial_score = min(t.score for t in self.trajectory[:3])  # First batch
        return self.best_score - initial_score
