import numpy as np
from uuid import UUID

from src.data.extractor import ContentExtractor
from src.models.content import SearchQuery, SearchResult


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(dot_product / norm_product) if norm_product > 0 else 0.0


class OpenAIEmbeddingSearch:
    def __init__(self, extractor: ContentExtractor, embedding_model):
        self.extractor = extractor
        self.embedding_model = embedding_model

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        query_embedding = await self._get_query_embedding(query.text)

        content_records = await self.extractor.get_content_with_token_embeddings()

        if query.content_types:
            content_records = [r for r in content_records if r.content_type in query.content_types]

        if query.organization_id:
            content_records = [r for r in content_records if r.organization_id == query.organization_id]

        content_records = [r for r in content_records if r.embedding is not None]

        scores: list[tuple[UUID, float, str, str, str]] = []
        for record in content_records:
            doc_embedding = np.array(record.embedding)
            score = cosine_similarity(query_embedding, doc_embedding)
            preview = record.index_content[:500]
            scores.append((record.id, score, record.title, preview, record.content_type))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[: query.top_k]

        results = []
        for rank, (content_id, score, title, preview, content_type) in enumerate(top_results, 1):
            results.append(
                SearchResult(
                    content_id=content_id,
                    score=score,
                    rank=rank,
                    title=title,
                    preview=preview,
                    content_type=content_type,
                )
            )

        return results

    async def _get_query_embedding(self, text: str) -> np.ndarray:
        """Get query embedding using the same embedding model as content."""
        embedding = await self.embedding_model.embed(text)
        return np.array(embedding)
