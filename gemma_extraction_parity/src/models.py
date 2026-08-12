from pydantic import BaseModel, Field


class GeneratedResearchQueryReview(BaseModel):
    learnings: list[str] = Field(..., description="List of learnings")
    follow_up_questions: list[str] = Field(
        ...,
        description="List of follow-up questions to research the topic further, max of 3.",
    )
    follow_up_title: str = Field(
        ..., description="A one or two word user-facing title for the follow-up questions, shown as progress in the UI"
    )


class SharedLearningPair(BaseModel):
    a_index: int = Field(description="1-based index from list A")
    b_index: int = Field(description="1-based index from list B")
    rationale: str = Field(description="Brief explanation of why these convey the same core fact")


class MatchResult(BaseModel):
    pairs: list[SharedLearningPair] = Field(description="Pairs of learnings that convey the same core fact")


class ParityAnalysis(BaseModel):
    shared: list[SharedLearningPair]
    a_only: list[str]
    b_only: list[str]
    a_deduped: list[str]
    b_deduped: list[str]
    duplicate_warnings: list[str]


class ContentRecord(BaseModel):
    id: str
    title: str
    author: str | None
    source_url: str
    created_at: str
    index_content: str


class ExtractionInput(BaseModel):
    query_id: str
    prompt_id: str
    topic: str
    directions: str
    max_learnings: int
    results: list[ContentRecord]
