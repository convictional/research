from datetime import datetime, timezone
from typing import TypedDict
import pandas as pd

from ..utils.bigquery import query_bq
from ..utils.tokens import split_content_into_chunks_by_tokens
from ..settings import settings
from ..utils.io import load_pickle_file, dump_to_pickle_file
from ..models import SourceContentBase, SourceContent
from .clean_content import clean_content
from .queries import GITHUB_ISSUES_QUERY, GITHUB_PULL_REQUESTS_QUERY, APP_TASKS_QUERY


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S%z"
MAX_TOKENS_PER_CHUNK = 40000
GITHUB_ISSUES_SOURCE_CONTENT_OUTPUT_PATH = settings.output_path / "source_content_data" / "github_issues_source_content_data.pkl"
GITHUB_PULL_REQUESTS_SOURCE_CONTENT_OUTPUT_PATH = settings.output_path / "source_content_data" / "github_pull_requests_source_content_data.pkl"
APP_TASKS_SOURCE_CONTENT_OUTPUT_PATH = settings.output_path / "source_content_data" / "app_tasks_source_content_data.pkl"


class SourceContentDictGitHubIssue(TypedDict):
    issue_id: int
    issue_number: int
    issue_title: str
    repository_full_name: str
    combined_content: str
    issue_created_at: pd.Timestamp
    last_comment_at: pd.Timestamp
    issue_closed_at: pd.Timestamp


class SourceContentDictGitHubPullRequest(TypedDict):
    issue_id: int
    issue_number: int
    issue_title: str
    repository_full_name: str
    combined_content: str
    issue_created_at: pd.Timestamp
    last_comment_at: pd.Timestamp
    issue_closed_at: pd.Timestamp


class SourceContentDictAppTask(TypedDict):
    task_id: int
    task_title: str
    combined_content: str
    task_created_at: pd.Timestamp
    last_comment_at: pd.Timestamp
    task_closed_at: pd.Timestamp


def execute_bigquery_query(query: str) -> pd.DataFrame:
    """
    Given a query string, return the results from BigQuery as a pandas DataFrame.
    """
    print("Querying BigQuery...")
    results_df = query_bq(query)
    results = results_df.to_dict(orient="records")

    # Clean content strings
    for r in results:
        r["combined_content"] = clean_content(str(r["combined_content"]))

    print(f"Retrieved {len(results)} rows from BigQuery.")

    return results


def fetch_and_cache_github_issues_data():
    """
    Fetch GitHub issues data from BigQuery via Fivetran model and cache it locally.
    """
    print("Fetching GitHub issues data from BigQuery...\n==============================")
    raw_data = execute_bigquery_query(GITHUB_ISSUES_QUERY)

    # Convert raw data to typed dicts
    source_content_dict: list[SourceContentDictGitHubIssue] = [SourceContentDictGitHubIssue(**r) for r in raw_data]
    print(f"Processed {len(source_content_dict)} GitHub issues from raw data.")

    # Convert to SourceContentBase objects
    source_content_base: list[SourceContentBase] = [
        SourceContentBase(
            content_id=f"{c['repository_full_name']}/issues/{c['issue_number']}",
            title=c["issue_title"],
            content=c["combined_content"],
            created_at=datetime.strptime(str(c["issue_created_at"]), DATETIME_FORMAT).replace(tzinfo=timezone.utc),
            last_comment_at=datetime.strptime(str(c["last_comment_at"]), DATETIME_FORMAT).replace(tzinfo=timezone.utc),
            closed_at=handle_string_timestamp(str(c["issue_closed_at"])),
            url=f"https://github.com/{c['repository_full_name']}/issues/{c['issue_number']}",
        )
        for c in source_content_dict
    ]
    print(f"Converted to {len(source_content_base)} SourceContentBase objects.")

    # Chunk content
    source_content: list[SourceContent] = chunk_source_content_base_content(source_content_base, "github_issue")
    print(f"Chunked into {len(source_content)} SourceContent objects.")

    # Dump to pickle file
    dump_to_pickle_file(source_content, GITHUB_ISSUES_SOURCE_CONTENT_OUTPUT_PATH)

    return source_content


def fetch_and_cache_github_pull_requests_data():
    """
    Fetch GitHub pull requests data from BigQuery via Fivetran model and cache it locally.
    """
    print("Fetching GitHub pull requests data from BigQuery...\n==============================")
    raw_data = execute_bigquery_query(GITHUB_PULL_REQUESTS_QUERY)

    # Convert raw data to typed dicts
    source_content_dict: list[SourceContentDictGitHubPullRequest] = [SourceContentDictGitHubPullRequest(**r) for r in raw_data]
    print(f"Processed {len(source_content_dict)} GitHub pull requests from raw data.")

    # Convert to SourceContentBase objects
    source_content_base: list[SourceContentBase] = [
        SourceContentBase(
            content_id=f"{c['repository_full_name']}/pull/{c['issue_number']}",
            title=c["issue_title"],
            content=c["combined_content"],
            created_at=datetime.strptime(str(c["issue_created_at"]), DATETIME_FORMAT).replace(tzinfo=timezone.utc),
            last_comment_at=datetime.strptime(str(c["last_comment_at"]), DATETIME_FORMAT).replace(tzinfo=timezone.utc),
            closed_at=handle_string_timestamp(str(c["issue_closed_at"])),
            url=f"https://github.com/{c['repository_full_name']}/pull/{c['issue_number']}",
        )
        for c in source_content_dict
    ]
    print(f"Converted to {len(source_content_base)} SourceContentBase objects.")

    # Chunk content
    source_content: list[SourceContent] = chunk_source_content_base_content(source_content_base, "github_pull_request")
    print(f"Chunked into {len(source_content)} SourceContent objects.")

    # Dump to pickle file
    dump_to_pickle_file(source_content, GITHUB_PULL_REQUESTS_SOURCE_CONTENT_OUTPUT_PATH)

    return source_content


def fetch_and_cache_app_task_data():
    """
    Fetch app Task data from BigQuery via Fivetran model and cache it locally.
    NOTE: HASHING FOR CONTENT COLUMNS SHOULD BE TURNED OFF IN FIVETRAN FOR THIS
    ELSE, THE CONTENT WILL BE HASHED AND NOT USEFUL FOR ANALYSIS.
    """
    print("Fetching app Task data from BigQuery...\n==============================")
    raw_data = execute_bigquery_query(APP_TASKS_QUERY)

    # Convert raw data to typed dicts
    source_content_dict: list[SourceContentDictAppTask] = [SourceContentDictAppTask(**r) for r in raw_data]
    print(f"Processed {len(source_content_dict)} app tasks from raw data.")

    # Convert to SourceContentBase objects
    source_content_base: list[SourceContentBase] = [
        SourceContentBase(
            content_id=f"tasks/{c['task_id']}",
            title=c["task_title"],
            content=c["combined_content"],
            created_at=datetime.strptime(str(c["task_created_at"]), DATETIME_FORMAT).replace(tzinfo=timezone.utc),
            last_comment_at=handle_string_timestamp(str(c["last_comment_at"])),
            closed_at=handle_string_timestamp(str(c["task_closed_at"])),
            url=f"https://app.example.com/tasks/{c['task_id']}",
        )
        for c in source_content_dict
    ]
    print(f"Converted to {len(source_content_base)} SourceContentBase objects.")

    # Chunk content
    source_content: list[SourceContent] = chunk_source_content_base_content(source_content_base, "app_task")
    print(f"Chunked into {len(source_content)} SourceContent objects.")

    # Dump to pickle file
    dump_to_pickle_file(source_content, APP_TASKS_SOURCE_CONTENT_OUTPUT_PATH)

    return source_content


def handle_string_timestamp(ts: str) -> datetime | None:
    """
    Handle the string timestamp.
    When the timestamp is null from BigQuery, the string representation is "NaT".
    """
    if ts != "NaT":
        return datetime.strptime(str(ts), DATETIME_FORMAT).replace(tzinfo=timezone.utc)
    else:
        return None


def chunk_source_content_base_content(content: list[SourceContentBase], source_content_type: str) -> list[SourceContent]:
    """
    Convert source content base into SourceContent with text chunks.
    """
    print(f"Splitting {len(content)} source content base items into chunks by {MAX_TOKENS_PER_CHUNK} tokens...")

    chunked_content: list[SourceContent] = []

    for c in content:
        chunks: list[SourceContent] = split_content_into_chunks_by_tokens(c, MAX_TOKENS_PER_CHUNK, source_content_type)
        chunked_content.extend(chunks)

    return chunked_content


def get_content_data_from_bigquery(load_from_cache: bool = False) -> list[SourceContent]:
    """
    The source data we are using is GitHub and Task data in BigQuery, that is ingested via the Fivetran connector.
    This function gets content data from BigQuery, split content into chunks, and augments the data.
    """
    print("Getting source content data...")

    if load_from_cache:
        print("Loading content data from cache...")
        github_issues_source_content = load_pickle_file(GITHUB_ISSUES_SOURCE_CONTENT_OUTPUT_PATH)
        github_pull_requests_source_content = load_pickle_file(GITHUB_PULL_REQUESTS_SOURCE_CONTENT_OUTPUT_PATH)
        task_source_content = load_pickle_file(APP_TASKS_SOURCE_CONTENT_OUTPUT_PATH)
    else:
        # Get GitHub issues data from BigQuery
        github_issues_source_content: list[SourceContent] = fetch_and_cache_github_issues_data()

        # Get GitHub pull requests data from BigQuery
        github_pull_requests_source_content: list[SourceContent] = fetch_and_cache_github_pull_requests_data()

        # Get app Task data from BigQuery
        task_source_content: list[SourceContent] = fetch_and_cache_app_task_data()

    # Combine all source content data
    source_content: list[SourceContent] = github_issues_source_content + github_pull_requests_source_content + task_source_content

    # Print summary
    print(f"GitHub issues source content items: {len(github_issues_source_content)}")
    print(f"Type of GitHub issues source content items: {type(github_issues_source_content[0])}")
    print(f"GitHub issues source content items type: {github_issues_source_content[0].type}")
    print(f"GitHub pull requests source content items: {len(github_pull_requests_source_content)}")
    print(f"Type of GitHub pull requests source content items: {type(github_pull_requests_source_content[0])}")
    print(f"GitHub pull requests source content items type: {github_pull_requests_source_content[0].type}")
    print(f"App tasks source content items: {len(task_source_content)}")
    print(f"Type of App tasks source content items: {type(task_source_content[0])}")
    print(f"App tasks source content items type: {task_source_content[0].type}")
    print(f"Total source content items: {len(source_content)}")

    return source_content
