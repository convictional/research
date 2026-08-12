from enum import Enum

from pydantic import BaseModel, Field


class HumanAction(str, Enum):
    PINNED = "pinned"
    DELETED = "deleted"
    NEUTRAL = "neutral"
    SYNTHETIC_NEGATIVE = "synthetic_negative"


class PointwiseExample(BaseModel):
    goal_id: str
    goal_title: str
    goal_description: str
    content_id: str
    content_type: str
    content_title: str
    content_body: str
    human_action: HumanAction
    original_signal: str = ""
    original_score: float = 0.0
    similarity_score: float | None = None


class PointwiseJudgment(BaseModel):
    is_aligned: bool = Field(..., description="Whether the content is aligned with the goal")
    signal: str | None = Field(None, description="Signal strength: strong, medium, weak, or None if not aligned")
    alignment_score: float = Field(..., ge=0.0, le=1.0, description="Alignment score 0-1")
    description: str = Field(..., description="Brief description of the alignment")
    reasoning: str = Field(..., description="Detailed reasoning for the judgment")


class ScoredPointwiseExample(BaseModel):
    example: PointwiseExample
    judgment: PointwiseJudgment


# --- Rubric models ---


class PointwiseDimension(BaseModel):
    dimension_name: str = Field(..., description="Name of the scoring dimension")
    description: str = Field(..., description="What this dimension captures")
    aligned_signals: list[str] = Field(..., description="Signals that content IS aligned on this dimension")
    not_aligned_signals: list[str] = Field(..., description="Signals that content is NOT aligned")
    weight_hint: str = Field(default="medium", description="Relative importance: high, medium, low")


class PointwiseRubric(BaseModel):
    dimensions: list[PointwiseDimension]
    version: int = 1
    notes: str = ""


class PointwiseBatchAnalysis(BaseModel):
    patterns: list[PointwiseDimension] = Field(
        ..., description="Pointwise dimensions identified from this batch"
    )
    observations: str = Field(..., description="General observations about pin/delete/neutral patterns")


class SynthesizedPointwiseRubric(BaseModel):
    dimensions: list[PointwiseDimension] = Field(..., description="Consolidated 3-6 pointwise dimensions")
    notes: str = Field(..., description="Overall notes about the scoring approach")


class RefinedPointwiseRubric(BaseModel):
    dimensions: list[PointwiseDimension] = Field(..., description="Updated pointwise dimensions")
    change_summary: str = Field(..., description="Summary of what changed and why")


# --- Evaluation models ---


class PointwiseClassMetrics(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


class PointwiseAccuracyResult(BaseModel):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, PointwiseClassMetrics] = Field(default_factory=dict)
    n_total: int
    n_correct: int
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    critical_errors: int = 0


class PointwiseEvaluationResult(BaseModel):
    accuracy: PointwiseAccuracyResult
    score_correlation: float = 0.0
    baselines: dict[str, float] = Field(default_factory=dict)
    per_goal_accuracy: dict[str, float] = Field(default_factory=dict)
    n_samples: int = 0
    split: str = ""


# --- Disagreement models ---


class PointwiseDisagreementInsight(BaseModel):
    case_index: int = Field(..., description="Index of the disagreement case")
    analysis: str = Field(..., description="Why the scorer likely got this wrong")
    root_cause: str = Field(
        ..., description="Category: rubric_gap, threshold_calibration, content_ambiguity, or signal_mismatch"
    )


class PointwiseDisagreementAnalysis(BaseModel):
    insights: list[PointwiseDisagreementInsight] = Field(
        ..., description="Analysis of each disagreement case"
    )
    rubric_changes: list[str] = Field(..., description="Specific rubric changes to improve scoring")
    prompt_changes: list[str] = Field(..., description="Specific prompt changes to improve scoring")
