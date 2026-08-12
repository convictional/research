"""
Database access helpers using asyncpg + Pydantic.

Implements the hybrid approach: raw SQL for performance, Pydantic for type safety.
"""

import asyncpg
from typing import Any

from .models import (
    Adapter,
    AdapterCreate,
    Content,
    Episode,
    EpisodeCreate,
    EvalQuery,
    EvalQueryCreate,
    EvalResult,
    EvalResultCreate,
    TrainingPair,
    TrainingPairCreate,
)


# Database connection configuration
DB_CONFIG = {
    "user": "adammccabe",
    "password": "",
    "database": "lora_cassettes",
    "host": "127.0.0.1",
    "port": 5432,
}


async def get_connection() -> asyncpg.Connection:
    """Create a database connection."""
    return await asyncpg.connect(**DB_CONFIG)


# =============================================================================
# EPISODES
# =============================================================================


async def create_episode(episode: EpisodeCreate) -> Episode:
    """Create a new training episode."""
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO episodes (
                episode_num, start_date, end_date, corpus_snapshot_date,
                num_new_chunks, num_updated_chunks, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            episode.episode_num,
            episode.start_date,
            episode.end_date,
            episode.corpus_snapshot_date,
            episode.num_new_chunks,
            episode.num_updated_chunks,
            episode.status,
        )
        return Episode(**dict(row))
    finally:
        await conn.close()


async def get_episode(episode_num: int) -> Episode | None:
    """Get episode by episode number."""
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM episodes WHERE episode_num = $1",
            episode_num,
        )
        return Episode(**dict(row)) if row else None
    finally:
        await conn.close()


async def get_latest_episode() -> Episode | None:
    """Get the most recent episode."""
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM episodes ORDER BY episode_num DESC LIMIT 1"
        )
        return Episode(**dict(row)) if row else None
    finally:
        await conn.close()


# =============================================================================
# TRAINING PAIRS
# =============================================================================


async def create_training_pairs_bulk(pairs: list[TrainingPairCreate]) -> int:
    """
    Bulk insert training pairs.

    Uses COPY for maximum performance with large datasets.
    Returns number of pairs inserted.
    """
    if not pairs:
        return 0

    conn = await get_connection()
    try:
        # Prepare data for COPY
        records = [
            (
                pair.episode_id,
                pair.anchor_content_id,
                pair.positive_content_id,
                pair.negative_content_id,
                pair.pair_type,
                pair.mining_method,
                pair.source_family,
                pair.anchor_text,
                pair.positive_text,
                pair.negative_text,
                pair.is_in_replay_buffer,
            )
            for pair in pairs
        ]

        # Use COPY for bulk insert
        result = await conn.copy_records_to_table(
            "training_pairs",
            records=records,
            columns=[
                "episode_id",
                "anchor_content_id",
                "positive_content_id",
                "negative_content_id",
                "pair_type",
                "mining_method",
                "source_family",
                "anchor_text",
                "positive_text",
                "negative_text",
                "is_in_replay_buffer",
            ],
        )

        return len(pairs)
    finally:
        await conn.close()


async def get_training_pairs_for_episode(
    episode_id: int, limit: int | None = None
) -> list[TrainingPair]:
    """Get training pairs for an episode."""
    conn = await get_connection()
    try:
        if limit is not None:
            query = "SELECT * FROM training_pairs WHERE episode_id = $1 LIMIT $2"
            rows = await conn.fetch(query, episode_id, limit)
        else:
            query = "SELECT * FROM training_pairs WHERE episode_id = $1"
            rows = await conn.fetch(query, episode_id)

        return [TrainingPair(**dict(row)) for row in rows]
    finally:
        await conn.close()


async def get_replay_buffer_pairs(limit: int | None = None) -> list[TrainingPair]:
    """Get pairs marked for replay buffer."""
    conn = await get_connection()
    try:
        if limit is not None:
            query = "SELECT * FROM training_pairs WHERE is_in_replay_buffer = TRUE LIMIT $1"
            rows = await conn.fetch(query, limit)
        else:
            query = "SELECT * FROM training_pairs WHERE is_in_replay_buffer = TRUE"
            rows = await conn.fetch(query)

        return [TrainingPair(**dict(row)) for row in rows]
    finally:
        await conn.close()


# =============================================================================
# ADAPTERS
# =============================================================================


async def create_adapter(adapter: AdapterCreate) -> Adapter:
    """Create a new adapter registry entry."""
    import json

    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO adapters (
                adapter_id, base_model, episode_id, sources, objective,
                train_start_date, train_end_date, replay_pct, hnsw_index_id,
                lora_config, training_config, metrics, stability_delta,
                status, storage_path, created_by
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            RETURNING *
            """,
            adapter.adapter_id,
            adapter.base_model,
            adapter.episode_id,
            json.dumps(adapter.sources),  # Convert list to JSON string
            adapter.objective,
            adapter.train_start_date,
            adapter.train_end_date,
            adapter.replay_pct,
            adapter.hnsw_index_id,
            json.dumps(adapter.lora_config),  # Convert dict to JSON string
            json.dumps(adapter.training_config),  # Convert dict to JSON string
            json.dumps(adapter.metrics),  # Convert dict to JSON string
            adapter.stability_delta,
            adapter.status,
            adapter.storage_path,
            adapter.created_by,
        )
        return Adapter(**dict(row))
    finally:
        await conn.close()


async def get_adapter_by_id(adapter_id: str) -> Adapter | None:
    """Get adapter by its semantic version ID."""
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM adapters WHERE adapter_id = $1",
            adapter_id,
        )
        return Adapter(**dict(row)) if row else None
    finally:
        await conn.close()


async def get_active_adapters() -> list[Adapter]:
    """Get all active adapters."""
    conn = await get_connection()
    try:
        rows = await conn.fetch(
            "SELECT * FROM adapters WHERE status = 'active' ORDER BY created_at DESC"
        )
        return [Adapter(**dict(row)) for row in rows]
    finally:
        await conn.close()


# =============================================================================
# CONTENT
# =============================================================================


async def get_content_by_id(content_id: str) -> Content | None:
    """Get content by ID."""
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM content WHERE id = $1",
            content_id,
        )
        return Content(**dict(row)) if row else None
    finally:
        await conn.close()


async def get_content_by_source(
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[Content]:
    """Get content by source type."""
    conn = await get_connection()
    try:
        if limit is not None:
            query = "SELECT * FROM content WHERE source = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3"
            rows = await conn.fetch(query, source, limit, offset)
        else:
            query = "SELECT * FROM content WHERE source = $1 ORDER BY created_at DESC OFFSET $2"
            rows = await conn.fetch(query, source, offset)

        return [Content(**dict(row)) for row in rows]
    finally:
        await conn.close()


async def get_github_threads() -> list[dict[str, Any]]:
    """
    Get GitHub issue/PR threads grouped by issue number.

    Returns list of dicts with issue_number, item_count, and content_ids.
    """
    conn = await get_connection()
    try:
        rows = await conn.fetch(
            """
            SELECT
                (metadata->>'issue_number')::int as issue_number,
                COUNT(*) as item_count,
                ARRAY_AGG(id ORDER BY created_at) as content_ids,
                MIN(title) as sample_title
            FROM content
            WHERE source = 'github'
              AND metadata->>'issue_number' IS NOT NULL
              AND sharing = 'organization'
            GROUP BY (metadata->>'issue_number')::int
            HAVING COUNT(*) > 1
            ORDER BY item_count DESC
            """
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


# =============================================================================
# EVAL QUERIES
# =============================================================================


async def create_eval_query(query: EvalQueryCreate) -> EvalQuery:
    """Create a new evaluation query."""
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO eval_queries (
                query_text, query_type, difficulty, expected_sources,
                ground_truth_content_ids, tags, is_in_stability_set
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            query.query_text,
            query.query_type,
            query.difficulty,
            query.expected_sources,
            query.ground_truth_content_ids,
            query.tags,
            query.is_in_stability_set,
        )
        return EvalQuery(**dict(row))
    finally:
        await conn.close()


async def get_stability_queries() -> list[EvalQuery]:
    """Get all queries in the stability set."""
    conn = await get_connection()
    try:
        rows = await conn.fetch(
            "SELECT * FROM eval_queries WHERE is_in_stability_set = TRUE"
        )
        return [EvalQuery(**dict(row)) for row in rows]
    finally:
        await conn.close()
