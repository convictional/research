"""Pydantic models for structured response testing."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    """Confidence levels for model responses."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SimpleResponse(BaseModel):
    """Simple structured response - baseline complexity."""

    final_response: str = Field(description="The main response to the user's question")
    confidence: ConfidenceLevel = Field(description="Confidence level in this response")


class ReasonedResponse(BaseModel):
    """Medium complexity - adds reasoning chain."""

    final_response: str = Field(description="The main response to the user's question")
    reasoning: str = Field(description="Step-by-step reasoning that led to this response")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class ComplexResponse(BaseModel):
    """High complexity - full production-like structure."""

    final_response: str = Field(min_length=10, description="The main response to the user's question")
    reasoning: str = Field(min_length=20, description="Detailed step-by-step reasoning process")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    learnings: list[str] = Field(
        min_length=1, max_length=5, description="Key insights or learnings from this analysis"
    )
    caveats: list[str] | None = Field(default=None, description="Optional limitations or caveats to consider")

    @field_validator("learnings")
    @classmethod
    def validate_learnings(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one learning is required")
        if len(v) > 5:
            raise ValueError("Maximum 5 learnings allowed")
        return v
