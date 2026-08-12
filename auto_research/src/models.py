from pydantic import BaseModel, Field


class PaperRelevance(BaseModel):
    paper_id: str
    is_relevant: bool
    relevance_reason: str
    relevance_score: int = Field(ge=1, le=10)


class FilteredPapers(BaseModel):
    papers: list[PaperRelevance]
