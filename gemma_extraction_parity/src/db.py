import logging

import asyncpg

from src.models import ContentRecord
from src.settings import settings

logger = logging.getLogger(__name__)


async def get_connection() -> asyncpg.Connection:
    conn = await asyncpg.connect(settings.postgres_url)
    count = await conn.fetchval("SELECT COUNT(*) FROM content")
    logger.info(f"Connected to database. Content count: {count}")
    if count == 0:
        await conn.close()
        raise RuntimeError("Content table is empty — run `make db_seed` to populate the dev database.")
    return conn


async def get_organization_id(conn: asyncpg.Connection) -> str:
    row = await conn.fetchrow("SELECT id FROM organization LIMIT 1")
    if not row:
        raise RuntimeError("No organizations found in the database.")
    org_id = str(row["id"])
    logger.info(f"Using organization: {org_id}")
    return org_id


async def search_content(
    conn: asyncpg.Connection,
    query: str,
    organization_id: str,
    limit: int = 10,
) -> list[ContentRecord]:
    # Production joins terms with OR for broader recall (ContentResearchQuery)
    or_query = " OR ".join(query.split())
    rows = await conn.fetch(
        """
        SELECT id, title, author, source_url, created_at,
               LEFT(index_content, $3) as index_content
        FROM content
        WHERE organization_id = $1
          AND text_search @@ websearch_to_tsquery('english', $2)
        ORDER BY ts_rank(text_search, websearch_to_tsquery('english', $2)) DESC
        LIMIT $4
        """,
        organization_id,
        or_query,
        settings.max_tokens_per_result,
        limit,
    )

    results = [
        ContentRecord(
            id=str(row["id"]),
            title=row["title"] or "",
            author=row["author"],
            source_url=row["source_url"] or "",
            created_at=str(row["created_at"]),
            index_content=row["index_content"] or "",
        )
        for row in rows
    ]
    logger.info(f"Search for '{query[:60]}...' returned {len(results)} results")
    return results
