"""
GitHub-specific pair mining strategies.

Based on PLAN.md section 5:
- Same thread pairs: items within the same GitHub issue/PR
- Parent↔reply relationships (issue ↔ comment, comment ↔ comment reply)
- Consecutive items within a thread

Mining methods:
- Heuristic: rule-based positive pair selection
- In-batch negatives: other items in the batch serve as negatives
"""

import asyncpg
from uuid import UUID

from ..data.models import Content, TrainingPairCreate


async def mine_same_thread_pairs(
    conn: asyncpg.Connection,
    episode_id: int,
    min_thread_size: int = 2,
    max_pairs_per_thread: int | None = None,
) -> list[TrainingPairCreate]:
    """
    Mine positive pairs from GitHub threads.

    Strategy:
    1. Get all threads with 2+ items
    2. For each thread, create pairs between consecutive items
    3. anchor=item[i], positive=item[i+1] (temporal ordering)

    Args:
        conn: Database connection
        episode_id: Episode ID for these pairs
        min_thread_size: Minimum items in thread to mine from
        max_pairs_per_thread: Limit pairs per thread (None = unlimited)

    Returns:
        List of training pair objects (without negatives - those come from in-batch sampling)
    """
    # Get all GitHub threads with issue numbers
    threads = await conn.fetch(
        """
        SELECT
            (metadata->>'issue_number')::int as issue_number,
            ARRAY_AGG(id ORDER BY created_at) as content_ids
        FROM content
        WHERE source = 'github'
          AND metadata->>'issue_number' IS NOT NULL
          AND sharing = 'organization'
        GROUP BY (metadata->>'issue_number')::int
        HAVING COUNT(*) >= $1
        ORDER BY COUNT(*) DESC
        """,
        min_thread_size,
    )

    print(f"Found {len(threads)} GitHub threads with {min_thread_size}+ items")

    pairs = []

    for thread in threads:
        issue_num = thread["issue_number"]
        content_ids = thread["content_ids"]

        # Fetch content for this thread
        content_rows = await conn.fetch(
            """
            SELECT id, title, index_content, content_type
            FROM content
            WHERE id = ANY($1::uuid[])
            ORDER BY created_at
            """,
            content_ids,
        )

        # Create consecutive pairs
        max_pairs = max_pairs_per_thread or len(content_rows)
        pair_count = 0

        for i in range(len(content_rows) - 1):
            if pair_count >= max_pairs:
                break

            anchor = content_rows[i]
            positive = content_rows[i + 1]

            pairs.append(
                TrainingPairCreate(
                    episode_id=episode_id,
                    anchor_content_id=UUID(str(anchor["id"])),
                    positive_content_id=UUID(str(positive["id"])),
                    negative_content_id=None,  # Will use in-batch negatives during training
                    pair_type="same_thread",
                    mining_method="heuristic",
                    source_family="github",
                    anchor_text=anchor["index_content"],
                    positive_text=positive["index_content"],
                    negative_text=None,
                )
            )

            pair_count += 1

    print(f"Mined {len(pairs)} same_thread pairs from {len(threads)} threads")
    return pairs


async def mine_parent_reply_pairs(
    conn: asyncpg.Connection,
    episode_id: int,
) -> list[TrainingPairCreate]:
    """
    Mine parent-reply pairs from GitHub.

    Strategy:
    - Issue ↔ first comment
    - Each comment ↔ related issue/PR context

    This is more sophisticated than same_thread as it captures semantic relationships
    rather than just temporal adjacency.

    Note: For MVP, we're focusing on same_thread pairs. This is a TODO for later.
    """
    # TODO: Implement parent-reply mining
    # Would need to parse comment threading structure
    return []


async def mine_cross_thread_hard_negatives(
    conn: asyncpg.Connection,
    anchor_pairs: list[TrainingPairCreate],
    num_negatives_per_pair: int = 1,
) -> list[TrainingPairCreate]:
    """
    Mine hard negatives from different threads using BM25 or dense similarity.

    Strategy:
    1. For each anchor, find similar content from DIFFERENT threads
    2. Use these as hard negatives (topically similar but not actually related)

    This is an advanced technique from PLAN.md section 5 (BM25 hard negatives, ANCE mining).

    Note: For MVP, we'll use in-batch negatives. This is a TODO for Episode 1+.
    """
    # TODO: Implement hard negative mining
    # Would need to build BM25 index or use existing dense embeddings
    return anchor_pairs
