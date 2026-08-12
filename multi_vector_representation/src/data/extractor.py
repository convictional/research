import json
import asyncpg
from uuid import UUID

from src.models.content import ContentRecord


def parse_pgvector(value) -> list[float] | None:
    """Parse pgvector value which may be returned as string or native type."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip("[]")
        return [float(x) for x in value.split(",")]
    return list(value)


class ContentExtractor:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.connection_string, min_size=1, max_size=10)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def get_all_content(self, limit: int | None = None) -> list[ContentRecord]:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        query = """
            SELECT
                id, title, index_content, content_type, category, source, source_id,
                source_url, author, preview_content, metadata, embedding, token_embeddings,
                created_at, updated_at, organization_id
            FROM content
            ORDER BY created_at DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [self._row_to_record(row) for row in rows]

    async def get_content_by_id(self, content_id: UUID) -> ContentRecord | None:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        query = """
            SELECT
                id, title, index_content, content_type, category, source, source_id,
                source_url, author, preview_content, metadata, embedding, token_embeddings,
                created_at, updated_at, organization_id
            FROM content
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, content_id)
            return self._row_to_record(row) if row else None

    async def get_content_with_token_embeddings(self, limit: int | None = None) -> list[ContentRecord]:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        query = """
            SELECT
                id, title, index_content, content_type, category, source, source_id,
                source_url, author, preview_content, metadata, embedding, token_embeddings,
                created_at, updated_at, organization_id
            FROM content
            WHERE token_embeddings IS NOT NULL
            ORDER BY created_at DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [self._row_to_record(row) for row in rows]

    async def get_content_ids_with_token_embeddings(self) -> list[UUID]:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        query = "SELECT id FROM content WHERE token_embeddings IS NOT NULL ORDER BY created_at DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [row["id"] for row in rows]

    async def get_content_by_ids(self, ids: list[UUID]) -> list[ContentRecord]:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        query = """
            SELECT
                id, title, index_content, content_type, category, source, source_id,
                source_url, author, preview_content, metadata, embedding, token_embeddings,
                created_at, updated_at, organization_id
            FROM content
            WHERE id = ANY($1)
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, ids)
            return [self._row_to_record(row) for row in rows]

    async def get_content_without_token_embeddings(self, limit: int | None = None) -> list[ContentRecord]:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        query = """
            SELECT
                id, title, index_content, content_type, category, source, source_id,
                source_url, author, preview_content, metadata, embedding, token_embeddings,
                created_at, updated_at, organization_id
            FROM content
            WHERE token_embeddings IS NULL
            ORDER BY created_at DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: asyncpg.Record) -> ContentRecord:
        data = dict(row)

        if data.get("embedding"):
            data["embedding"] = parse_pgvector(data["embedding"])

        if data.get("token_embeddings"):
            if isinstance(data["token_embeddings"], str):
                data["token_embeddings"] = None
            else:
                data["token_embeddings"] = [parse_pgvector(vec) for vec in data["token_embeddings"]]

        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])

        return ContentRecord(**data)
