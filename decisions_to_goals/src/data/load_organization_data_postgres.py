"""Postgres-based data loader for the decisions_to_goals experiment.

This is the data ingress: it queries the local postgres DB populated via
`make research_load` in app/web (see settings.postgres_dsn).

Schema mapping (legacy BigQuery export schema → postgres):
  `decision`  → postgres `post` + `content` (index_content has goals/criteria/options)
  `task`      → postgres `post` (discussions as activity events)
  `insight`   → postgres `postcomment`
  `goal`      → postgres `goal`
"""

import re
import uuid
from datetime import datetime

import asyncpg

from ..models import ActivityEvent, Decision, StatedGoal
from ..settings import logger, settings


def _extract_goals(index_content: str) -> str | None:
    """Extract the 'Goals:' line from old-format decision content."""
    for line in index_content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Goals: "):
            return stripped[len("Goals: "):]
    return None


def _extract_section(index_content: str, section_name: str) -> list[dict]:
    """Parse Title/Description blocks from a named section in old-format content."""
    # Find the section header
    pattern = re.compile(rf"^{section_name}:\s*$", re.MULTILINE)
    match = pattern.search(index_content)
    if not match:
        return []

    # Find the next major section (Options:, Criteria:, or end)
    rest = index_content[match.end():]
    # Stop at the next all-caps-style section header or end-of-content marker
    stop = re.search(r"^(Options|Criteria|This decision has been finalized):", rest, re.MULTILINE)
    block = rest[: stop.start()] if stop else rest

    items = []
    # Each item starts with "Title: ..." followed optionally by "Description: ..."
    title_positions = [m.start() for m in re.finditer(r"^Title: ", block, re.MULTILINE)]
    for i, pos in enumerate(title_positions):
        end = title_positions[i + 1] if i + 1 < len(title_positions) else len(block)
        chunk = block[pos:end].strip()
        title_line = chunk.split("\n")[0]
        title = title_line[len("Title: "):].strip()
        desc_match = re.search(r"^Description: (.+?)(?:\nEvaluations:|\n\n|$)", chunk, re.DOTALL | re.MULTILINE)
        description = desc_match.group(1).strip() if desc_match else ""
        if title:
            items.append({"title": title, "description": description})
    return items


def _uuid_from_source_id(source_id: str) -> str:
    """Extract UUID string from 'gid://decide/Post/<uuid>'."""
    return source_id.split("/")[-1]


async def _fetch_decisions(
    conn: asyncpg.Connection,
    org_id: str,
    cutoff_date: datetime,
) -> list[Decision]:
    """Return structured decisions from the content table.

    Old-format decisions ("Title: I'm considering ...") live only in the
    content search index — there is no corresponding row in post because the
    old decision table was migrated away. These entries carry Goals, Options,
    and Criteria text that is the primary signal for the experiment.

    New-format posts that have a matching content entry are also included as
    decisions with no explicit goals text.
    """
    # Old-format decisions: in content but not in post
    orphan_rows = await conn.fetch(
        """
        SELECT
            c.source_id,
            c.title,
            c.created_at,
            COALESCE(c.author, 'Unknown') AS creator_name,
            c.index_content
        FROM content c
        WHERE c.organization_id = $1
          AND c.content_type = 'post'
          AND c.sharing = 'organization'
          AND c.index_content LIKE 'Title: I''m considering%'
          AND c.created_at <= $2
          AND NOT EXISTS (
              SELECT 1 FROM post p
              WHERE 'gid://decide/Post/' || p.id::text = c.source_id
          )
        ORDER BY c.created_at DESC
        """,
        uuid.UUID(org_id),
        cutoff_date,
    )
    logger.info(f"  Fetched {len(orphan_rows)} old-format decisions from content (no post record).")

    # New-format posts with content entries (for completeness)
    post_rows = await conn.fetch(
        """
        SELECT
            p.id AS post_id,
            p.title,
            p.created_at,
            COALESCE(u.name, 'Unknown') AS creator_name,
            COALESCE(c.index_content, p.title) AS index_content
        FROM post p
        LEFT JOIN "user" u ON u.id = p.creator_id
        LEFT JOIN content c ON
            c.source_id = 'gid://decide/Post/' || p.id::text
            AND c.content_type = 'post'
        WHERE p.organization_id = $1
          AND p.deleted_at IS NULL
          AND p.sharing = 'organization'
          AND p.created_at <= $2
        ORDER BY p.created_at DESC
        """,
        uuid.UUID(org_id),
        cutoff_date,
    )
    logger.info(f"  Fetched {len(post_rows)} new-format posts as decisions.")

    # Fetch postcomments for new-format posts (orphan posts have no FK-linked comments)
    post_uuids = [r["post_id"] for r in post_rows]
    comment_rows = await conn.fetch(
        """
        SELECT
            pc.post_id AS post_id,
            COALESCE(pc.content, '') AS content,
            pc.created_at,
            COALESCE(u.name, 'Unknown') AS user_name
        FROM postcomment pc
        LEFT JOIN "user" u ON u.id = pc.user_id
        WHERE pc.post_id = ANY($1)
          AND pc.deleted_at IS NULL
        ORDER BY pc.created_at ASC
        """,
        post_uuids,
    ) if post_uuids else []

    comments_by_post: dict[uuid.UUID, list[dict]] = {}
    for c in comment_rows:
        comments_by_post.setdefault(c["post_id"], []).append(
            {
                "user": c["user_name"],
                "created_at": str(c["created_at"]),
                "text": c["content"],
            }
        )

    decisions: list[Decision] = []

    # Old-format decisions from orphaned content entries
    for row in orphan_rows:
        index_content: str = row["index_content"]
        dec_id = _uuid_from_source_id(row["source_id"])
        decisions.append(
            Decision(
                id=dec_id,
                title=row["title"],
                description=index_content,
                author_stated_goals=_extract_goals(index_content),
                options=_extract_section(index_content, "Options"),
                criteria=_extract_section(index_content, "Criteria"),
                comments=[],  # orphan posts have no FK-linked comments
                created_at=row["created_at"],
            )
        )

    # New-format posts
    for row in post_rows:
        index_content = row["index_content"]
        row_id: uuid.UUID = row["post_id"]
        decisions.append(
            Decision(
                id=str(row_id),
                title=row["title"],
                description=index_content,
                author_stated_goals=None,
                options=[],
                criteria=[],
                comments=comments_by_post.get(row_id, []),
                created_at=row["created_at"],
            )
        )

    decisions.sort(key=lambda d: d.created_at, reverse=True)
    return decisions


async def _fetch_activity_events(
    conn: asyncpg.Connection,
    org_id: str,
    cutoff_date: datetime,
) -> list[ActivityEvent]:
    org_uuid = uuid.UUID(org_id)
    post_rows = await conn.fetch(
        """
        SELECT
            p.id AS id,
            p.title,
            p.created_at,
            COALESCE(u.name, 'Unknown') AS creator_name,
            COALESCE(c.index_content, p.title) AS body
        FROM post p
        LEFT JOIN "user" u ON u.id = p.creator_id
        LEFT JOIN content c ON
            c.source_id = 'gid://decide/Post/' || p.id::text
            AND c.content_type = 'post'
        WHERE p.organization_id = $1
          AND p.deleted_at IS NULL
          AND p.sharing = 'organization'
          AND p.created_at <= $2
        ORDER BY p.created_at DESC
        """,
        org_uuid,
        cutoff_date,
    )
    logger.info(f"  Fetched {len(post_rows)} posts as activity events.")

    post_uuids = [r["id"] for r in post_rows]
    comment_rows = await conn.fetch(
        """
        SELECT
            pc.id AS id,
            pc.post_id AS post_id,
            COALESCE(pc.content, '') AS content,
            pc.created_at,
            COALESCE(u.name, 'Unknown') AS user_name
        FROM postcomment pc
        LEFT JOIN "user" u ON u.id = pc.user_id
        WHERE pc.post_id = ANY($1)
          AND pc.deleted_at IS NULL
          AND pc.created_at <= $2
        ORDER BY pc.created_at DESC
        """,
        post_uuids,
        cutoff_date,
    ) if post_uuids else []
    logger.info(f"  Fetched {len(comment_rows)} post comments as activity events.")

    events: list[ActivityEvent] = []
    for r in post_rows:
        events.append(
            ActivityEvent(
                event_id=str(r["id"]),
                event_type="discussion",
                title=r["title"] or "(untitled post)",
                body=r["body"],
                created_at=r["created_at"],
                author=r["creator_name"],
            )
        )
    for c in comment_rows:
        content: str = c["content"]
        events.append(
            ActivityEvent(
                event_id=str(c["id"]),
                event_type="comment",
                title=content[:80] if content else "(empty comment)",
                body=content,
                created_at=c["created_at"],
                author=c["user_name"],
                parent_ids=[str(c["post_id"])],
            )
        )

    events.sort(key=lambda e: e.created_at, reverse=True)
    return events


async def _fetch_stated_goals(
    conn: asyncpg.Connection,
    org_id: str,
    cutoff_date: datetime,
) -> list[StatedGoal]:
    rows = await conn.fetch(
        """
        SELECT id AS id, title
        FROM goal
        WHERE organization_id = $1
          AND sharing = 'organization'
          AND deleted_at IS NULL
          AND created_at <= $2
        """,
        uuid.UUID(org_id),
        cutoff_date,
    )
    logger.info(f"  Fetched {len(rows)} stated goals from postgres.")
    return [
        StatedGoal(
            id=str(r["id"]),
            title=r["title"] or "(untitled goal)",
            description=r["title"] or "(untitled goal)",
            source="convictional_seed",
        )
        for r in rows
    ]


async def load_from_postgres() -> tuple[list[ActivityEvent], list[Decision], list[StatedGoal]]:
    """Load org data from local postgres. Requires settings.postgres_dsn."""
    dsn = settings.postgres_dsn
    if not dsn:
        raise RuntimeError(
            "postgres_dsn is not set. Add POSTGRES_DSN=postgresql://user@host/dbname to your .env file."
        )

    logger.info(f"Connecting to postgres at {dsn!r}...")
    conn = await asyncpg.connect(dsn)
    try:
        org_id = settings.org_id
        cutoff = datetime.fromisoformat(settings.dataset_cutoff_date)

        print("Fetching decisions from postgres...")
        decisions = await _fetch_decisions(conn, org_id, cutoff)

        print("Fetching activity events from postgres...")
        activity_events = await _fetch_activity_events(conn, org_id, cutoff)

        print("Fetching stated goals from postgres...")
        stated_goals = await _fetch_stated_goals(conn, org_id, cutoff)
    finally:
        await conn.close()

    return activity_events, decisions, stated_goals
