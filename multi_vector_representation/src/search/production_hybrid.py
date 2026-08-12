import asyncpg
import numpy as np
from uuid import UUID

from src.data.extractor import ContentExtractor
from src.models.content import SearchQuery, SearchResult


class ProductionHybridSearch:
    def __init__(self, extractor: ContentExtractor, embedding_model):
        self.extractor = extractor
        self.embedding_model = embedding_model

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        query_embedding = await self._get_query_embedding(query.text)
        embedding_str = "[" + ", ".join(str(value) for value in query_embedding) + "]"

        text_query = " OR ".join(query.text.split())

        sql = """
            WITH vector_matches AS (
                SELECT id, (1 - (embedding <=> $1) / 2) AS similarity
                FROM content
                WHERE token_embeddings IS NOT NULL
            ),
            ranked_results AS (
                SELECT
                    c.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.category
                        ORDER BY (
                            vm.similarity * 0.7 +
                            ts_rank(c.text_search, websearch_to_tsquery('english', $2)) * 0.3
                        ) DESC
                    ) AS rn
                FROM vector_matches vm
                JOIN content c ON c.id = vm.id
            )
            SELECT *
            FROM ranked_results
            WHERE rn <= $3
            ORDER BY (
                (1 - (embedding <=> $1) / 2) * 0.7 +
                ts_rank(text_search, websearch_to_tsquery('english', $2)) * 0.3
            ) DESC
            LIMIT $4;
        """

        if not self.extractor.pool:
            raise RuntimeError("Database pool not initialized")

        async with self.extractor.pool.acquire() as conn:
            rows = await conn.fetch(sql, embedding_str, text_query, 10, query.top_k)

        results = []
        for rank, row in enumerate(rows, 1):
            preview = str(row["index_content"])[:500]
            results.append(
                SearchResult(
                    content_id=row["id"],
                    score=0.0,
                    rank=rank,
                    title=row["title"],
                    preview=preview,
                    content_type=row["content_type"],
                )
            )

        return results

    async def _get_query_embedding(self, text: str) -> list[float]:
        embedding = await self.embedding_model.embed(text)
        return embedding if isinstance(embedding, list) else list(embedding)
