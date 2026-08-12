import asyncpg
from dataclasses import dataclass


@dataclass
class ContentRecord:
    id: str
    index_content: str
    openai_embedding: list[float]


def parse_pgvector(value: str | list[float] | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip("[]")
        return [float(x) for x in value.split(",")]
    return list(value)


async def fetch_content_with_embeddings(
    connection_string: str,
    limit: int = 1000,
) -> list[ContentRecord]:
    """Fetch content records that have non-zero OpenAI embeddings and non-empty index_content."""
    conn = await asyncpg.connect(connection_string)
    try:
        rows = await conn.fetch(
            """
            SELECT id, index_content, embedding
            FROM content
            WHERE index_content IS NOT NULL
              AND index_content != ''
              AND embedding IS NOT NULL
              AND embedding != $1::vector
            ORDER BY id
            LIMIT $2
            """,
            "[" + ",".join(["0.0"] * 1536) + "]",
            limit,
        )
        records = []
        for row in rows:
            embedding = parse_pgvector(row["embedding"])
            if embedding is None:
                continue
            records.append(
                ContentRecord(
                    id=str(row["id"]),
                    index_content=row["index_content"],
                    openai_embedding=embedding,
                )
            )
        return records
    finally:
        await conn.close()


async def fetch_search_queries(connection_string: str) -> list[str]:
    """Fetch distinct search queries from the researchquery table."""
    conn = await asyncpg.connect(connection_string)
    try:
        rows = await conn.fetch(
            """
            SELECT DISTINCT content_search->>'query' AS query
            FROM researchquery
            WHERE content_search->>'query' IS NOT NULL
              AND content_search->>'query' != ''
            ORDER BY query
            """
        )
        return [row["query"] for row in rows]
    finally:
        await conn.close()
