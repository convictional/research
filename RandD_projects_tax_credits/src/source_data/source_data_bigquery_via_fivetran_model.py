from datetime import datetime, timezone
from typing import TypedDict
import pandas as pd

from ..utils.bigquery import query_bq
from ..utils.tokens import split_content_into_chunks_by_tokens
from ..settings import settings
from ..utils.io import load_pickle_file, dump_to_pickle_file
from ..models import GitHubSourceContent
from .clean_content import clean_content


BIGQUERY_QUERY = """
-- 1 row per issue comment,
-- with issue metadata attached to each row
WITH issue_comments AS (
    SELECT
        i.id AS issue_id,
        i.number AS issue_number,
        i.title AS issue_title,
        i.body AS issue_body,
        ic.body AS comment_body,
        i.created_at AS issue_created_at,
        i.closed_at as issue_closed_at,
        ic.created_at AS comment_created_at,
        i.user_id AS issue_user_id,
        ic.user_id AS comment_user_id,
        r.full_name AS repository_full_name
    FROM `${GCP_PROJECT}.github.issue` i
    LEFT JOIN `${GCP_PROJECT}.github.issue_comment` ic
        ON i.id = ic.issue_id
    LEFT JOIN `${GCP_PROJECT}.github.repository` r
        ON i.repository_id = r.id
    WHERE
        i.pull_request IS FALSE
        AND i.body IS NOT NULL
        AND ic.body IS NOT NULL
        AND i.created_at >= '2024-01-01'
        AND i.created_at < '2025-01-01'
        AND ic.created_at >= '2024-01-01'
        AND ic.created_at < '2025-01-01'
        AND r.full_name in (
            "convictional/decide",
            "convictional/data"
        )
),

-- 1 row per issue or issue comment
-- there will be duplicates, e.g. for each comment there will be a row for a given issue
issues_and_comments_flattened as (
    -- issues
    SELECT
        issue_id,
        issue_number,
        issue_title,
        repository_full_name,
        issue_body AS content,
        issue_created_at AS content_created_at,
        issue_user_id AS user_id
    FROM issue_comments

    UNION ALL

    -- issue comments
    SELECT
        issue_id,
        issue_number,
        issue_title,
        repository_full_name,
        comment_body AS content,
        comment_created_at AS content_created_at,
        comment_user_id AS user_id
    FROM issue_comments
    WHERE comment_body IS NOT NULL
),

-- 1 row per issue or issue comment,
-- no duplication of issues here
issues_and_comments_flattened_deduplicated as (
    SELECT DISTINCT
        issue_id,
        issue_number,
        issue_title,
        repository_full_name,
        content,
        content_created_at,
        user_id
    FROM issues_and_comments_flattened
),

-- 1 row per issue
issues_timestamps as (
    SELECT DISTINCT
        issue_id,
        MIN(issue_created_at) AS issue_created_at,
        MAX(comment_created_at) as last_comment_at,
        MAX(issue_closed_at) AS issue_closed_at,
    FROM issue_comments
    GROUP BY issue_id
),

-- 1 row per issue or issue comment
content_with_user_names AS (
    SELECT
        icfd.issue_id,
        icfd.issue_number,
        icfd.issue_title,
        icfd.repository_full_name,
        icfd.content,
        icfd.content_created_at,
        icfd.user_id,
        u.name AS user_name,
        u.login AS user_login,
        coalesce(u.name, u.login, 'Unknown user') AS username,
        it.issue_created_at,
        it.last_comment_at,
        it.issue_closed_at
    FROM issues_and_comments_flattened_deduplicated icfd
    LEFT JOIN `${GCP_PROJECT}.github.user` u
        ON icfd.user_id = u.id
    LEFT JOIN issues_timestamps it
        ON icfd.issue_id = it.issue_id
),

-- 1 row per issue
aggregated_content as (
    SELECT
        issue_id,
        issue_number,
        issue_title,
        issue_created_at,
        last_comment_at,
        issue_closed_at,
        repository_full_name,
        STRING_AGG(
            CONCAT(username, ': ', content),
            '\\n'
            ORDER BY
                CASE WHEN content_created_at = issue_created_at THEN 0 ELSE 1 END,
                content_created_at ASC
            LIMIT 1000
        ) AS combined_content
    FROM content_with_user_names
    GROUP BY
        issue_id, issue_number, issue_title, issue_created_at, last_comment_at, issue_closed_at, repository_full_name
)

select *
from aggregated_content
order by issue_id
"""
MAX_TOKENS_PER_CHUNK = 40000
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S%z"
CONTENT_DATA_OUTPUT_PATH = settings.output_path / "source_content_data" / "content_data.pkl"


class SourceContentDict(TypedDict):
    issue_id: int
    issue_number: int
    issue_title: str
    repository_full_name: str
    combined_content: str
    issue_created_at: pd.Timestamp
    last_comment_at: pd.Timestamp
    issue_closed_at: pd.Timestamp


class ChunkedSourceContentDict(SourceContentDict):
    chunk_index: int
    text_chunk: str


def get_source_data_from_bigquery() -> list[SourceContentDict]:
    """
    Get source data from BigQuery.
    """
    print("Querying BigQuery for source data and cleaning content...")
    results_df = query_bq(BIGQUERY_QUERY)
    results = results_df.to_dict(orient="records")

    # Clean content strings
    for r in results:
        r["combined_content"] = clean_content(str(r["combined_content"]))

    # return results
    return [SourceContentDict(**r) for r in results]


def chunk_source_content(content: list[SourceContentDict]) -> list[ChunkedSourceContentDict]:
    chunked_content = split_content_into_chunks_by_tokens(content, "combined_content", MAX_TOKENS_PER_CHUNK)

    return [ChunkedSourceContentDict(**c) for c in chunked_content]


def convert_to_content_objects(chunked_source_content: list[dict]) -> list[GitHubSourceContent]:
    """
    Convert chunked source content data to content objects.
    """
    print("Converting to content objects...")

    data = [
        GitHubSourceContent(
            content_id=f"{c['repository_full_name']}/issues/{c['issue_number']}",
            title=c["issue_title"],
            content=c["combined_content"],
            created_at=datetime.strptime(str(c["issue_created_at"]), DATETIME_FORMAT).replace(tzinfo=timezone.utc),
            last_comment_at=datetime.strptime(str(c["last_comment_at"]), DATETIME_FORMAT).replace(tzinfo=timezone.utc),
            closed_at=handle_string_timestamp(str(c["issue_closed_at"])),
            chunk_index=c["chunk_index"],
            text_chunk=f"{c['issue_title']} {c['text_chunk']}",
            url=f"https://github.com/{c['repository_full_name']}/issues/{c['issue_number']}",
        )
        for c in chunked_source_content
    ]

    return data


def handle_string_timestamp(ts: str) -> datetime | None:
    """
    Handle the string timestamp.
    When the timestamp is null from BigQuery, the string representation is "NaT".
    """
    if ts != "NaT":
        return datetime.strptime(str(ts), DATETIME_FORMAT).replace(tzinfo=timezone.utc)
    else:
        return None


def get_content_data_from_bigquery_via_fivetran_model(load_from_cache: bool = False) -> list[GitHubSourceContent]:
    """
    The source data we are using is GitHub data in BigQuery, that is ingested via the Fivetran connector.
    This function gets content data from BigQuery, split content into chunks, and augment the data.
    """
    print("Getting source content data...")

    if load_from_cache:
        print("Loading content data from cache...")
        content_data = load_pickle_file(CONTENT_DATA_OUTPUT_PATH)
    else:
        # Get source data from BigQuery
        print("Getting GitHub source data from BigQuery...")
        source_content: list[SourceContentDict] = get_source_data_from_bigquery()
        print(f"Got {len(source_content)} source content data items from BigQuery.")

        # Split content into chunks
        print("Splitting content into chunks by tokens...")
        chunked_source_content: list[ChunkedSourceContentDict] = chunk_source_content(source_content)
        print(f"Got {len(chunked_source_content)} chunked source content data items.")

        # Convert content to content objects
        print("Converting content to content objects...")
        content_data: list[GitHubSourceContent] = convert_to_content_objects(chunked_source_content)
        print(f"Converted to {len(content_data)} content data objects.")

        # dump to pickle file
        dump_to_pickle_file(content_data, CONTENT_DATA_OUTPUT_PATH)

    print(f"Number of content items loaded: {len(content_data)}")
    print(f"Type of content items is {type(content_data[0])}")

    return content_data
