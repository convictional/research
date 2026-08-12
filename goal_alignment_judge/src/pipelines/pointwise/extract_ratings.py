from pathlib import Path

import pandas as pd

from common.postgres import query_local_postgres

from ...settings import DATABASE, logger

RATINGS_TABLE = "goalalignment"

EXTRACT_RATINGS_QUERY = """
SELECT
    ga.goal_id::text,
    g.title AS goal_title,
    CASE WHEN g.parent_id IS NULL THEN 'true' ELSE 'false' END AS is_top_level,
    c.source_id AS content_source_url,
    c.sharing,
    ga.deleted_at,
    ga.id::text,
    ga.created_at,
    ga.updated_at,
    ga.signal,
    ga.alignment_score,
    ga.description,
    ga.content_id::text,
    ga.goal_id::text AS "goal_id_dup",
    ga.organization_id::text,
    ga.pinned_by_id::text,
    ga.content_indexed_at,
    ga.created_by_id::text,
    0 AS pair_count
FROM {table} ga
JOIN goal g ON g.id = ga.goal_id
JOIN content c ON c.id = ga.content_id
ORDER BY ga.goal_id, ga.created_at
"""


async def extract_ratings(
    output_path: Path,
    database: str = DATABASE,
    table: str = RATINGS_TABLE,
) -> Path:
    """Extract goal alignment ratings from the local dev DB as a CSV.

    Produces a CSV matching the format of goal_alignments_rated.csv, with columns:
    goal_id, goal_title, is_top_level, content_source_url, sharing, deleted_at, id,
    created_at, updated_at, signal, alignment_score, description, content_id, goal_id,
    organization_id, pinned_by_id, content_indexed_at, created_by_id, pair_count
    """
    query = EXTRACT_RATINGS_QUERY.format(table=table)
    rows = await query_local_postgres(query, logger, database=database)

    if not rows:
        logger.warning("No ratings found in database")
        return output_path

    # Rename goal_id_dup back to goal_id for CSV format compatibility (duplicate column)
    df = pd.DataFrame(rows)
    df = df.rename(columns={"goal_id_dup": "goal_id"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Extracted {len(df)} ratings to {output_path}")

    # Summary
    pinned = df["pinned_by_id"].notna() & (df["pinned_by_id"] != "")
    deleted = df["deleted_at"].notna() & (df["deleted_at"] != "")
    n_pinned = pinned.sum()
    n_deleted = deleted.sum()
    n_neutral = len(df) - n_pinned - n_deleted
    n_goals = df.iloc[:, 0].nunique()  # First goal_id column
    logger.info(f"  {n_goals} goals, {n_pinned} pinned, {n_deleted} deleted, {n_neutral} neutral")

    return output_path
