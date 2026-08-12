import asyncio
from urllib.parse import urlparse

import asyncpg
from openai import AsyncOpenAI

from common.embeddings import aembed_query
from src.models import SearchResult
from src.settings import logger, settings

_openai_client: AsyncOpenAI | None = None
_pool: asyncpg.Pool | None = None


def _parse_postgres_url(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "user": parsed.username or "decide",
        "password": parsed.password or "",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/"),
    }


async def init_search() -> None:
    global _openai_client, _pool
    db_params = _parse_postgres_url(settings.postgres_url)
    _pool = await asyncpg.create_pool(**db_params, min_size=1, max_size=5)
    _openai_client = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        organization=settings.openai_organization,
    )
    logger.info("Initialized content search (DB + OpenAI embeddings)")


async def close_search() -> None:
    global _openai_client, _pool
    if _pool:
        await _pool.close()
        _pool = None
    _openai_client = None


async def hybrid_search(query: str, limit: int | None = None) -> list[SearchResult]:
    if _openai_client is None or _pool is None:
        raise RuntimeError("Call init_search() before hybrid_search()")

    limit = limit or settings.content_search_limit

    embedding = await aembed_query(
        _openai_client, query, settings.embedding_model, settings.embedding_dimension
    )

    sql = f"""
    WITH vector_matches AS (
        SELECT id, title, index_content,
               (1 - (embedding <=> $1::vector) / 2) AS similarity
        FROM {settings.content_table}
        WHERE organization_id = $2
        ORDER BY embedding <=> $1::vector
        LIMIT 500
    ),
    text_matches AS (
        SELECT id, title, index_content,
               ts_rank(text_search, websearch_to_tsquery('english', $3)) AS text_rank
        FROM {settings.content_table}
        WHERE organization_id = $2
          AND text_search @@ websearch_to_tsquery('english', $3)
        ORDER BY text_rank DESC
        LIMIT 300
    ),
    scored AS (
        SELECT id, title, index_content,
               MAX(similarity) AS similarity,
               MAX(text_rank) AS text_rank
        FROM (
            SELECT id, title, index_content, similarity, 0::float AS text_rank FROM vector_matches
            UNION ALL
            SELECT id, title, index_content, 0::float AS similarity, text_rank FROM text_matches
        ) combined
        GROUP BY id, title, index_content
    )
    SELECT id, title, index_content,
           (COALESCE(similarity, 0) * 0.7 + COALESCE(text_rank, 0) * 0.3) AS score
    FROM scored
    ORDER BY score DESC
    LIMIT $4;
    """

    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
    async with _pool.acquire() as conn:
        rows = await conn.fetch(sql, embedding_str, settings.organization_id, query, limit)

    return [
        SearchResult(
            id=str(row["id"]),
            title=row["title"] or "",
            content=row["index_content"] or "",
            score=float(row["score"]),
        )
        for row in rows
    ]
