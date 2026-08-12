from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field


class ContentRecord(BaseModel):
    id: UUID
    title: str
    index_content: str
    content_type: str
    category: str
    source: str
    source_id: str
    source_url: str
    author: str | None = None
    preview_content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    token_embeddings: list[list[float]] | None = None
    created_at: datetime
    updated_at: datetime
    organization_id: UUID

    class Config:
        arbitrary_types_allowed = True

    def get_embedding_array(self) -> np.ndarray | None:
        if self.embedding is None:
            return None
        return np.array(self.embedding, dtype=np.float32)

    def get_token_embeddings_array(self) -> np.ndarray | None:
        if self.token_embeddings is None:
            return None
        return np.array(self.token_embeddings, dtype=np.float32)


class SearchQuery(BaseModel):
    text: str
    top_k: int = 10
    content_types: list[str] | None = None
    organization_id: UUID | None = None


class SearchResult(BaseModel):
    content_id: UUID
    score: float
    rank: int
    title: str
    preview: str
    content_type: str
