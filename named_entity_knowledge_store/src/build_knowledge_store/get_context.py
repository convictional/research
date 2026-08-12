import re
import pandas as pd

from typing import List
from google.cloud.bigquery import Client, QueryJobConfig

from ..settings import settings, logger


def query_bq(query: str) -> pd.DataFrame:
    # Note: this requres you to have setup application-default auth for gcloud, use `make auth` to do this!
    client = Client(settings.gcp_project)
    job_config = QueryJobConfig(use_query_cache=False)
    query_job = client.query(query, job_config=job_config)
    return query_job.to_dataframe()


def augment_content_chunks(content_chunks: list[dict]) -> list[dict]:
    for chunk in content_chunks:
        chunk["text_chunk"] = f"{chunk['title']} {chunk['content']}"
        # Convert Timestamp to datetime object
        if isinstance(chunk["created_at"], pd.Timestamp):
            chunk["created_at"] = chunk["created_at"].to_pydatetime()
        if isinstance(chunk["updated_at"], pd.Timestamp):
            chunk["updated_at"] = chunk["updated_at"].to_pydatetime()
    return content_chunks


def clean_content(content: str) -> str:
    """
    Clean the content by removing or replacing special characters.
    """
    # Remove non-ASCII characters
    cleaned_content = re.sub(r"[^\x00-\x7F]+", " ", content)
    return cleaned_content


def get_source_content_from_bq(
    sources_filter: List[str] | None = None,
    org_id: str = settings.organization_id,
    content_date_filter: str = "2024-01-01",
) -> List[dict]:
    """
    This function queries our bigquery instance for content data for the data sources specified in the sources list.
    If no sources are specified, it will query all sources.
    """
    logger.info(f"Getting content from BigQuery for sources: {sources_filter if sources_filter else 'all'}...")

    query = f"""
    SELECT
        id as content_id,
        title,
        source_type as source,
        created_at,
        updated_at,
        TO_JSON_STRING(STRUCT(
                    metadata,
                    category as content_category,
                    content
        )) AS content
    FROM
        {settings.gcp_project}.cloudsql_decide_public.content
    WHERE
        NOT _fivetran_deleted
        AND organization_id = '{org_id}'
        AND created_at >= TIMESTAMP '{content_date_filter}'
        AND NOT content_type in ('github_comment')
    ORDER BY
        created_at DESC
    """
    if sources_filter:
        query += "WHERE source IN ({sources})"
        query = query.format(sources=", ".join([f"'{source}'" for source in sources_filter]))
    results_df = query_bq(query)
    results = results_df.to_dict(orient="records")

    # Clean content chunks
    for chunk in results:
        chunk["content"] = clean_content(str(chunk["content"]))

    return results
