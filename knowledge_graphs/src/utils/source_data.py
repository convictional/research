import re
import pandas as pd
import asyncpg

from typing import List
from google.cloud.bigquery import Client, QueryJobConfig

from ..config.experiment_settings import settings


async def query_local_postgres_db(query: str, *args) -> List[dict]:
    """
    This function executes a query on the local postgres database and returns the results as a list of dictionaries.
    """
    try:
        conn = await asyncpg.connect(
            user=settings.local_postgres_user,
            password=settings.local_postgres_password,
            database=settings.local_postgres_dbname,
            host=settings.local_postgres_host,
            port=settings.local_postgres_port,
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

    try:
        records = await conn.fetch(query, *args)
        return [dict(row) for row in records]
    except Exception as e:
        print(f"An unexpected error during query execution: {e}")
        raise e
    finally:
        await conn.close()
        print("Database connection is closed.")


def query_bq(query: str) -> pd.DataFrame:
    # Note: this requres you to have setup application-default auth for gcloud, use `make auth` to do this!
    client = Client(settings.gcp_project)
    job_config = QueryJobConfig(use_query_cache=False)
    query_job = client.query(query, job_config=job_config)
    return query_job.to_dataframe()


def get_source_content_from_bq(
    sources_filter: List[str] | None = None,
    org_id: str = settings.organization_id,
    content_date_filter: str = "2024-01-01",
) -> List[dict]:
    """
    This function queries local postgres database for content data for the data sources specified in the sources list.
    If no sources are specified, it will query all sources.
    """
    print(f"Getting content from BigQuery for sources: {sources_filter if sources_filter else 'all'}...")

    query = f"""
    WITH decision_data AS (
        SELECT
            decision.id as decision_id,
            decision.title,
            decision.goals,
            outcome.summary as outcome_summary,
            outcome.explanation as outcome_explanation,
            decision.created_at,
            decision.updated_at,
            decider.name as decider_name,
            creator.name as creator_name,
            MAX(CASE WHEN option.selected_at IS NOT NULL THEN TRUE ELSE FALSE END) as is_decided
        FROM
            {settings.gcp_project}.cloudsql_decide_public.decision
        JOIN
            {settings.gcp_project}.cloudsql_decide_public.user creator on creator.id = decision.creator_id
        JOIN
            {settings.gcp_project}.cloudsql_decide_public.user decider on decider.id = decision.decider_id
        LEFT JOIN
            {settings.gcp_project}.cloudsql_decide_public.option ON option.decision_id = decision.id
        LEFT JOIN
            {settings.gcp_project}.cloudsql_decide_public.outcome ON outcome.id = decision.outcome_id
        WHERE
            decision.sharing <> 'private'
            AND decision.organization_id = '{org_id}'
            AND decision.created_at >= TIMESTAMP '{content_date_filter}'
            AND decision._fivetran_deleted = false
        GROUP BY
            decision.id,
            decision.title,
            decision.goals,
            decision.created_at,
            decision.updated_at,
            decider.name,
            creator.name,
            outcome.summary,
            outcome.explanation
        HAVING is_decided
    ),
    criteria AS (
        SELECT
            decision_id,
            ARRAY_AGG(STRUCT(
                title AS criterion_title,
                description AS criterion_description,
                source AS criterion_source
            )) AS criteria
        FROM {settings.gcp_project}.cloudsql_decide_public.criterion
        WHERE _fivetran_deleted = false
        GROUP BY decision_id
    ),
    collaborators AS (
        SELECT
            co.decision_id,
            ARRAY_AGG(STRUCT(
                u.name as names
            )) as collaborator
        FROM
            {settings.gcp_project}.cloudsql_decide_public.collaborator co
        JOIN
            {settings.gcp_project}.cloudsql_decide_public.user u ON u.id = co.user_id
        GROUP BY co.decision_id
    ),
    insights AS (
        SELECT
            i.decision_id,
            ARRAY_AGG(STRUCT(
                i.id as insight_id,
                i.title as insight_title,
                i.description as insight_description,
                i.created_at as insight_created_at,
                i.updated_at as insight_updated_at,
                i.citations,
                i.position,
                i.source as insight_source,
                i.subtitle,
                creator.name as creator_name,
                assigned.name as assignee_name,
                completor.name as completor_name
                )) as insights
        FROM
            {settings.gcp_project}.cloudsql_decide_public.insight i
        LEFT JOIN
            {settings.gcp_project}.cloudsql_decide_public.user creator on creator.id = i.creator_id
        LEFT JOIN
            {settings.gcp_project}.cloudsql_decide_public.user assigned on assigned.id = i.assignee_id
        LEFT JOIN
            {settings.gcp_project}.cloudsql_decide_public.user completor on completor.id = i.completer_id
        WHERE
            i._fivetran_deleted = false
        GROUP BY i.decision_id
    ),
    content AS (
    SELECT
        dd.decision_id as content_id,
        "app_decision" as source,
        dd.title,
        dd.created_at,
        dd.updated_at,
        TO_JSON_STRING(STRUCT(
            dd.goals,
            dd.decider_name,
            dd.creator_name,
            CONCAT(dd.outcome_summary, dd.outcome_explanation) as outcome,
            ARRAY_AGG(STRUCT(c.criteria)) as criteria,
            ARRAY_AGG(STRUCT(col.collaborator)) as collaborator,
            ARRAY_AGG(STRUCT(i.insights)) as insights
        )) as content
    FROM
        decision_data dd
    LEFT JOIN criteria c ON dd.decision_id = c.decision_id
    LEFT JOIN collaborators col ON dd.decision_id = col.decision_id
    LEFT JOIN insights i ON dd.decision_id = i.decision_id
    GROUP BY
        content_id,
        source,
        dd.title,
        dd.created_at,
        dd.updated_at,
        dd.goals,
        dd.decider_name,
        dd.creator_name,
        dd.outcome_summary,
        dd.outcome_explanation
    )

    SELECT
        *
    FROM
        content
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


def augment_content_chunks(content_chunks: list[dict]) -> list[dict]:
    for chunk in content_chunks:
        chunk["text_chunk"] = f"{chunk['title']} {chunk['content']}"
        chunk["created_at"] = chunk["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        chunk["updated_at"] = chunk["updated_at"].strftime("%Y-%m-%d %H:%M:%S")
    return content_chunks


def get_app_decisions_as_df(org_id: str = settings.organization_id) -> pd.DataFrame:
    """
    This method returns the in-app decisions from BigQuery, including options, criteria,
    their evaluations, and the summary from the ability table.
    Decisions are returned at the grain of one row per decision, where options, criteria,
    evaluations, and the summary are nested as df's within their respective columns.

    Helper functions, process_options_group, process_insights_group, and process_decision_group
    contain the resulting schemas.

    Args:
        org_id (str): The organization id to filter the decisions. Defaulted to Convictional.

    Returns:
        pd.DataFrame: A dataframe containing the decisions with their options, criteria,
                      evaluations, and summaries.
    """

    query = f"""
    WITH decision_data AS (
        SELECT
            d.id as decision_id,
            d.title as decision_title,
            d.goals,
            d.created_at,
            d.updated_at,
            decider.name as decider_name,
            creator.name as creator_name,
            MAX(CASE WHEN o.selected_at IS NOT NULL THEN TRUE ELSE FALSE END) as is_decided
        FROM
            {settings.gcp_project}.cloudsql_decide_public.decision d
        JOIN
            {settings.gcp_project}.cloudsql_decide_public.user creator on creator.id = d.creator_id
        JOIN
            {settings.gcp_project}.cloudsql_decide_public.user decider on decider.id = d.decider_id
        LEFT JOIN
            {settings.gcp_project}.cloudsql_decide_public.option o ON o.decision_id = d.id
        WHERE
            d.sharing <> 'private'
            AND d.organization_id = '{org_id}'
            AND d._fivetran_deleted = false
        GROUP BY
            d.id, d.title, d.goals, d.created_at, d.updated_at, decider.name, creator.name
    ),
    options AS (
        SELECT
            decision_id,
            id as option_id,
            title AS option_title,
            description AS option_description,
            source AS option_source
        FROM {settings.gcp_project}.cloudsql_decide_public.option
        WHERE _fivetran_deleted = false AND archived_at is null
    ),
    criteria AS (
        SELECT
            decision_id,
            id as criterion_id,
            title AS criterion_title,
            description AS criterion_description,
            source AS criterion_source
        FROM {settings.gcp_project}.cloudsql_decide_public.criterion
        WHERE _fivetran_deleted = false
    ),
    evaluations AS (
        SELECT
            id as evaluation_id,
            option_id,
            criterion_id,
            rating,
            explanation
        FROM {settings.gcp_project}.cloudsql_decide_public.evaluation
    ),
    collaborators AS (
        SELECT
            co.decision_id,
            STRING_AGG(u.name, ', ') as collaborator_names
        FROM
            {settings.gcp_project}.cloudsql_decide_public.collaborator co
        JOIN
            {settings.gcp_project}.cloudsql_decide_public.user u ON u.id = co.user_id
        GROUP BY co.decision_id
    ),
    insights AS (
        SELECT
            i.id as insight_id,
            i.decision_id,
            i.title as insight_title,
            i.description as insight_description,
            i.created_at as insight_created_at,
            i.updated_at as insight_updated_at,
            i.citations,
            i.position,
            i.source as insight_source,
            i.subtitle,
            creator.name as creator_name,
            assigned.name as assignee_name,
            completor.name as completor_name
        FROM
            {settings.gcp_project}.cloudsql_decide_public.insight i
        LEFT JOIN
            {settings.gcp_project}.cloudsql_decide_public.user creator on creator.id = i.creator_id
        LEFT JOIN
            {settings.gcp_project}.cloudsql_decide_public.user assigned on assigned.id = i.assignee_id
        LEFT JOIN
            {settings.gcp_project}.cloudsql_decide_public.user completor on completor.id = i.completer_id
        WHERE
            i._fivetran_deleted = false
    ),
    summaries AS (
        SELECT
            decision_id,
            summary
        FROM (
            SELECT
                decision_id,
                content AS summary,
                ROW_NUMBER() OVER (PARTITION BY decision_id ORDER BY created_at DESC) AS rn
            FROM
                {settings.gcp_project}.cloudsql_decide_public.ability
            WHERE
                ability_type = "summarize"
        ) t
        WHERE rn = 1
    )
    SELECT
        dd.*,
        s.summary,
        o.option_id,
        o.option_title,
        o.option_description,
        o.option_source,
        c.criterion_id,
        c.criterion_title,
        c.criterion_description,
        c.criterion_source,
        e.evaluation_id,
        e.rating,
        e.explanation,
        col.collaborator_names,
        i.insight_id,
        i.insight_title,
        i.insight_description,
        i.insight_created_at,
        i.insight_updated_at,
        i.citations,
        i.position,
        i.insight_source,
        i.subtitle,
        i.creator_name,
        i.assignee_name,
        i.completor_name
    FROM
        decision_data dd
    LEFT JOIN summaries s ON dd.decision_id = s.decision_id
    LEFT JOIN options o ON dd.decision_id = o.decision_id
    LEFT JOIN criteria c ON dd.decision_id = c.decision_id
    LEFT JOIN evaluations e ON o.option_id = e.option_id AND c.criterion_id = e.criterion_id
    LEFT JOIN collaborators col ON dd.decision_id = col.decision_id
    LEFT JOIN insights i ON dd.decision_id = i.decision_id
    """

    results = query_bq(query)
    print(f"Number of rows returned from BigQuery: {len(results)}")

    decisions = (
        pd.DataFrame(results).groupby("decision_id").apply(process_decision_group).reset_index(name="decision_data")
    )

    return decisions


def clean_content(content: str) -> str:
    """
    Clean the content by removing or replacing special characters.
    """
    # Example: Remove non-ASCII characters
    cleaned_content = re.sub(r"[^\x00-\x7F]+", " ", content)
    return cleaned_content


# Helper functions for wrangling many to one relationships with the decision
def process_option_group(y):
    return {
        "option_id": y["option_id"].iloc[0],
        "option_title": y["option_title"].iloc[0],
        "option_description": y["option_description"].iloc[0],
        "option_source": y["option_source"].iloc[0],
        "criteria_evaluations": y[
            [
                "criterion_id",
                "criterion_title",
                "criterion_description",
                "criterion_source",
                "evaluation_id",
                "rating",
                "explanation",
            ]
        ].to_dict("records"),
    }


def process_insight_group(y):
    return {
        "insight_id": y["insight_id"].iloc[0],
        "insight_title": y["insight_title"].iloc[0],
        "insight_description": y["insight_description"].iloc[0],
        "insight_created_at": y["insight_created_at"].iloc[0],
        "insight_updated_at": y["insight_updated_at"].iloc[0],
        "citations": y["citations"].iloc[0],
        "position": y["position"].iloc[0],
        "insight_source": y["insight_source"].iloc[0],
        "subtitle": y["subtitle"].iloc[0],
        "creator_name": y["creator_name"].iloc[0],
        "assignee_name": y["assignee_name"].iloc[0],
        "completor_name": y["completor_name"].iloc[0],
    }


def process_decision_group(x):
    return {
        "decision_id": x["decision_id"].iloc[0],
        "decision_title": x["decision_title"].iloc[0],
        "goals": x["goals"].iloc[0],
        "created_at": x["created_at"].iloc[0],
        "updated_at": x["updated_at"].iloc[0],
        "decider_name": x["decider_name"].iloc[0],
        "creator_name": x["creator_name"].iloc[0],
        "is_decided": x["is_decided"].iloc[0],
        "summary": x["summary"].iloc[0] if pd.notna(x["summary"].iloc[0]) else None,
        "collaborators": (
            [name.strip() for name in x["collaborator_names"].iloc[0].split(", ")]
            if pd.notna(x["collaborator_names"].iloc[0])
            else []
        ),
        "options": x.groupby("option_id").apply(process_option_group).to_dict(),
        "criteria": x[["criterion_id", "criterion_title", "criterion_description", "criterion_source"]]
        .drop_duplicates()
        .to_dict("records"),
        "insights": x.groupby("insight_id").apply(process_insight_group).to_dict(),
    }
