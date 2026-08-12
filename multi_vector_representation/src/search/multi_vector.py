import torch
from uuid import UUID

from src.embedders.colbert import ColBERTEmbedder
from src.search.maxsim import maxsim_score
from src.data.extractor import ContentExtractor
from src.models.content import SearchQuery, SearchResult


class ColBERTLocalSearch:
    def __init__(self, embedder: ColBERTEmbedder, extractor: ContentExtractor):
        self.embedder = embedder
        self.extractor = extractor

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        query_embeddings = self.embedder.embed_single(query.text)

        content_records = await self.extractor.get_content_with_token_embeddings()

        if query.content_types:
            content_records = [r for r in content_records if r.content_type in query.content_types]

        if query.organization_id:
            content_records = [r for r in content_records if r.organization_id == query.organization_id]

        scores: list[tuple[UUID, float, str, str, str]] = []
        for record in content_records:
            doc_embeddings = torch.tensor(record.token_embeddings)
            score = maxsim_score(query_embeddings, doc_embeddings)
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
