"""Database client for querying the seed database."""

import asyncpg
from datetime import datetime
from typing import List, Optional
from .models import ContentItem
from .config import config


class DatabaseClient:
    """Client for querying the content table in the seed database."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Establish connection pool to the database."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=config.get("database.db_host", "localhost"),
                port=config.get("database.db_port", 5432),
                user=config.get("database.db_user", "postgres"),
                database=config.get("database.db_name", "local_research_db"),
                min_size=1,
                max_size=5,
            )

    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def query_content_before_date(
        self,
        before_date: datetime,
        content_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[ContentItem]:
        """
        Query content items created before a given date.

        Args:
            before_date: Only return content created before this date
            content_types: Optional list of content types to filter by (include only these)
            exclude_types: Optional list of content types to exclude (e.g., google_doc)
            limit: Maximum number of items to return

        Returns:
            List of ContentItem objects
        """
        if not self.pool:
            await self.connect()

        # Build query
        query = """
            SELECT id, content_type, title, index_content, created_at, author, metadata
            FROM content
            WHERE created_at < $1
        """

        params = [before_date]

        if content_types:
            query += f" AND content_type = ANY(${len(params) + 1})"
            params.append(content_types)

        if exclude_types:
            query += f" AND content_type != ALL(${len(params) + 1})"
            params.append(exclude_types)

        query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)

        # Execute query
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Convert to ContentItem models
        items = []
        for row in rows:
            items.append(
                ContentItem(
                    id=str(row["id"]),
                    type=row["content_type"],
                    title=row["title"],
                    content=row["index_content"],
                    created_at=row["created_at"],
                )
            )

        return items

    async def query_content_by_user(
        self,
        before_date: datetime,
        user_id: str,
        content_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[ContentItem]:
        """
        Query content items accessible to a specific user before a date.

        Args:
            before_date: Only return content created before this date
            user_id: User ID to filter by (checks allowed_user_ids or author)
            content_types: Optional list of content types to filter by (include only these)
            exclude_types: Optional list of content types to exclude (e.g., google_doc)
            limit: Maximum number of items to return

        Returns:
            List of ContentItem objects

        Note:
            We exclude google_doc by default because documents are mutable and we can't
            reconstruct their historical state at a specific date. This prevents temporal
            leakage in the experiment.
        """
        if not self.pool:
            await self.connect()

        # Build query - check if user_id is in allowed_user_ids or is the author
        query = """
            SELECT id, content_type, title, index_content, created_at, author, metadata
            FROM content
            WHERE created_at < $1
            AND (
                allowed_user_ids @> $2::jsonb
                OR author = $3
            )
        """

        params = [before_date, f'["{user_id}"]', user_id]

        if content_types:
            query += f" AND content_type = ANY(${len(params) + 1})"
            params.append(content_types)

        if exclude_types:
            query += f" AND content_type != ALL(${len(params) + 1})"
            params.append(exclude_types)

        query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)

        # Execute query
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Convert to ContentItem models
        items = []
        for row in rows:
            items.append(
                ContentItem(
                    id=str(row["id"]),
                    type=row["content_type"],
                    title=row["title"],
                    content=row["index_content"],
                    created_at=row["created_at"],
                )
            )

        return items


# Global database client instance
db_client = DatabaseClient()
