"""Synthetic prompts for structured response testing."""

from dataclasses import dataclass


@dataclass
class StructuredPrompt:
    """Test prompt with expected schema."""

    prompt: str
    model_class_name: str
    description: str


STRUCTURED_PROMPTS = [
    StructuredPrompt(
        prompt="What is the capital of France?",
        model_class_name="SimpleResponse",
        description="Basic factual question",
    ),
    StructuredPrompt(
        prompt="Should I invest in cryptocurrency?",
        model_class_name="SimpleResponse",
        description="Opinion question with confidence",
    ),
    StructuredPrompt(
        prompt="A train leaves Station A at 60 mph. Another train leaves Station B (100 miles away) at 40 mph toward Station A. When do they meet?",
        model_class_name="ReasonedResponse",
        description="Math problem requiring reasoning",
    ),
    StructuredPrompt(
        prompt="Why did the Roman Empire fall?",
        model_class_name="ReasonedResponse",
        description="Historical analysis with reasoning",
    ),
    StructuredPrompt(
        prompt="Is Python or JavaScript better for web development?",
        model_class_name="ReasonedResponse",
        description="Comparative analysis requiring reasoning",
    ),
    StructuredPrompt(
        prompt="Analyze the trade-offs between microservices and monolithic architecture for a startup with 5 engineers.",
        model_class_name="ComplexResponse",
        description="Multi-faceted technical decision",
    ),
    StructuredPrompt(
        prompt="What are the ethical implications of using AI for hiring decisions?",
        model_class_name="ComplexResponse",
        description="Complex ethical analysis",
    ),
    StructuredPrompt(
        prompt="How would you design a caching strategy for a social media feed with 10 million daily active users?",
        model_class_name="ComplexResponse",
        description="System design with multiple considerations",
    ),
]
