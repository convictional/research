"""
Context retriever using hybrid search (semantic + text-based).

Recreates production hybrid search with temporal filtering for the experiment.
"""

import asyncpg
from datetime import datetime
from typing import List, Optional
from openai import AsyncOpenAI

from .models import Context, ContentItem
from .config import config


class ContextRetriever:
    """
    Retrieve relevant historical context using hybrid search.

    Combines:
    - Vector similarity search (70% weight) using OpenAI embeddings
    - Full-text search (30% weight) using PostgreSQL tsvector
    - Temporal filtering (only content before target date)
    - Content type filtering (excludes mutable content like google_doc)
    """

    def __init__(self, openai_api_key: str):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
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

    async def get_context_for_date(
        self,
        target_date: datetime,
        previous_work: Optional[str] = None,
        top_k: int = 20,
    ) -> Context:
        """
        Retrieve relevant context for a given date using hybrid search.

        Strategy:
        1. Use previous day's work to query for semantically similar content
        2. Include all content created in the last 24 hours (recent activity)
        3. Combine and deduplicate

        Args:
            target_date: Date we're predicting for (only get content before this)
            previous_work: What was worked on the day before (used as query)
            top_k: Number of semantic search results to return

        Returns:
            Context object with categorized content items
        """
        if not self.pool:
            await self.connect()

        # Get excluded content types from config
        exclude_types = config.get("context.exclude_content_types", ["google_doc"])

        items = []

        # Part 1: Recent activity (last 24 hours)
        # This captures all recent work without filtering
        from datetime import timedelta

        recent_cutoff = target_date - timedelta(days=1)
        recent_items = await self._get_recent_content(
            start_date=recent_cutoff,
            end_date=target_date,
            exclude_types=exclude_types,
        )
        items.extend(recent_items)

        # Part 2: Semantic search based on previous day's work
        if previous_work:
            # Use previous work as the query
            query = previous_work
        else:
            # Fallback to generic query
            query = "What should I prioritize today?"

        semantic_items = await self._semantic_search(
            query=query,
            before_date=target_date,
            exclude_types=exclude_types,
            top_k=top_k,
        )
        items.extend(semantic_items)

        # Deduplicate by ID (prefer recent items if duplicates)
        seen_ids = set()
        unique_items = []
        for item in items:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                unique_items.append(item)

        # Categorize items by type
        emails = []
        meetings = []
        tasks = []
        discussions = []
        other = []

        for item in unique_items:
            if "email" in item.type.lower():
                emails.append(item)
            elif "meeting" in item.type.lower() or "calendar" in item.type.lower():
                meetings.append(item)
            elif "task" in item.type.lower() or "issue" in item.type.lower():
                tasks.append(item)
            elif "discussion" in item.type.lower() or "comment" in item.type.lower():
                discussions.append(item)
            else:
                other.append(item)

        # Create Context object
        return Context(
            target_date=target_date,
            emails=emails,
            meetings=meetings,
            tasks=tasks + other,  # Combine tasks and other into tasks
            discussions=discussions,
        )

    async def _get_recent_content(
        self,
        start_date: datetime,
        end_date: datetime,
        exclude_types: List[str],
    ) -> List[ContentItem]:
        """Get all content created in a date range (e.g., last 24 hours)."""
        sql = """
            SELECT id, content_type, title, index_content, created_at, author
            FROM content
            WHERE created_at >= $1
            AND created_at < $2
            AND content_type != ALL($3)
            ORDER BY created_at DESC;
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, start_date, end_date, exclude_types)

        items = []
        for row in rows:
            items.append(
                ContentItem(
                    id=str(row["id"]),
                    type=row["content_type"],
                    title=row["title"] or "",
                    content=row["index_content"] or "",
                    created_at=row["created_at"],
                    relevance_score=1.0,  # Recent items get max relevance
                )
            )

        return items

    async def _semantic_search(
        self,
        query: str,
        before_date: datetime,
        exclude_types: List[str],
        top_k: int,
    ) -> List[ContentItem]:
        """Perform hybrid semantic + text search."""
        # Get query embedding from OpenAI
        query_embedding = await self._get_query_embedding(query)
        embedding_str = "[" + ", ".join(str(v) for v in query_embedding) + "]"

        # Prepare text search query (OR all words)
        text_query = " OR ".join(query.split())

        # Build hybrid search SQL with temporal filtering
        sql = """
            WITH vector_matches AS (
                SELECT id, (1 - (embedding <=> $1::vector) / 2) AS similarity
                FROM content
                WHERE created_at < $2
                AND content_type != ALL($3)
            )
            SELECT
                c.id,
                c.content_type,
                c.title,
                c.index_content,
                c.created_at,
                c.author,
                (
                    vm.similarity * 0.7 +
                    ts_rank(c.text_search, websearch_to_tsquery('english', $4)) * 0.3
                ) AS hybrid_score
            FROM vector_matches vm
            JOIN content c ON c.id = vm.id
            ORDER BY hybrid_score DESC
            LIMIT $5;
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                sql,
                embedding_str,
                before_date,
                exclude_types,
                text_query,
                top_k,
            )

        items = []
        for row in rows:
            items.append(
                ContentItem(
                    id=str(row["id"]),
                    type=row["content_type"],
                    title=row["title"] or "",
                    content=row["index_content"] or "",
                    created_at=row["created_at"],
                    relevance_score=float(row["hybrid_score"]),
                )
            )

        return items

    async def _get_query_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for query text."""
        response = await self.openai_client.embeddings.create(
            model="text-embedding-3-small",  # Fast and cheap
            input=text,
        )
        return response.data[0].embedding
