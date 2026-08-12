import aiohttp
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from google.cloud.bigquery import Client, QueryJobConfig

from .settings import settings, logger

# These queries stay the same, but stripped out the commented-out lines in meeting_usage.
thread_usage = """
SELECT
  thread.id as thread_id,
  question.id as question_id,
  user.id as user_id,
  thread.created_at as thread_created_at
FROM `${GCP_PROJECT}.cloudsql_decide_public.thread` thread
JOIN `${GCP_PROJECT}.cloudsql_decide_public.question` question on question.thread_id = thread.id
LEFT JOIN `${GCP_PROJECT}.cloudsql_decide_public.user` user on user.id = question.creator_id
WHERE user.organization_id = "00000000-0000-0000-0000-000000000000"
  AND thread.created_at >= TIMESTAMP '2024-11-14'
  AND NOT (user._fivetran_deleted OR question._fivetran_deleted OR thread._fivetran_deleted)
"""

decision_process_usage = """
SELECT
  id as decision_id,
  organization_id,
  created_at
FROM `${GCP_PROJECT}.cloudsql_decide_public.decisionprocess`
WHERE organization_id = "00000000-0000-0000-0000-000000000000"
  AND created_at >= TIMESTAMP '2024-11-14'
  AND NOT _fivetran_deleted
"""

meeting_usage = """
SELECT
  id as meeting_id,
  created_at,
  processed_transcript
FROM `${GCP_PROJECT}.cloudsql_decide_public.meeting`
WHERE NOT processed_transcript IS NULL
    AND NOT _fivetran_deleted
"""

llm_request_query = """
WITH llm_requests AS(
  SELECT
    job_metadata,
    JSON_VALUE(request_body, '$.model') AS llm_model,
    JSON_VALUE(request_body, '$.max_tokens') AS request_max_tokens,
    JSON_VALUE(request_body, '$.tool_choice.name') AS tool_choice_name,
    JSON_VALUE(job_metadata, '$.id.value') AS job_id,
    JSON_VALUE(job_metadata, '$.created_at.value') as created_at,
    JSON_VALUE(job_metadata, '$.queue') AS queue,
    JSON_VALUE(job_metadata, '$.job_type') AS job_type,
    JSON_VALUE(job_metadata, '$.job_details.user_id') AS user_id,
    JSON_VALUE(job_metadata, '$.job_details.meeting_id') AS meeting_id,
    JSON_VALUE(job_metadata, '$.job_details.question_id') AS question_id,
    JSON_VALUE(job_metadata, '$.job_details.thread_id') AS thread_id,
    JSON_VALUE(job_metadata, '$.job_details.organization_id') AS organization_id,
    JSON_VALUE(job_metadata, '$.job_details.decision_process_id') AS decision_process_id,
    JSON_VALUE(job_metadata, '$.job_details.collaborator_id') AS collaborator_id,
    CAST(JSON_VALUE(response_body, '$.usage.input_tokens') AS INT64) AS input_tokens,
    CAST(JSON_VALUE(response_body, '$.usage.output_tokens') AS INT64) AS output_tokens,
    (
      CAST(JSON_VALUE(response_body, '$.usage.input_tokens') AS INT64)
      + CAST(JSON_VALUE(response_body, '$.usage.output_tokens') AS INT64)
    ) AS total_tokens
  FROM `${GCP_PROJECT}.cloudsql_decide_public.llmrequest`
  WHERE NOT _fivetran_deleted
)
SELECT
  *
FROM llm_requests
WHERE CAST(created_at AS TIMESTAMP) >= TIMESTAMP '2024-11-14'
"""

MODEL_PRICING = {
    "claude-3-5-sonnet-20241022": {
        "input_cost_per_m": 3.0,  # $3 per million input tokens
        "output_cost_per_m": 15.0,  # $15 per million output tokens
    }
}
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


class RecallAIClient:
    """
    Simple client to query the Recall AI API for usage data.
    """

    def __init__(self):
        self.api_key = settings.recall_ai_api_key.get_secret_value()
        self.base_url = "https://us-east-1.recall.ai/api/v1"
        self.headers = {
            "Authorization": f"TOKEN {self.api_key}",
            "accept": "application/json",
            "Content-Type": "application/json",
        }

    async def get_billing_usage(self, start_date: datetime, end_date: datetime):
        url = f"{self.base_url}/billing/usage/"
        params = {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        }
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Recall AI API error: {response.status} - {await response.text()}")
                return await response.json()


def clear_output_directory(path: Path) -> None:
    """Clear and recreate the output directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)


def query_bq(query: str) -> pd.DataFrame:
    """Execute a BigQuery query and return results as a DataFrame."""
    client = Client(settings.gcp_project)
    job_config = QueryJobConfig(use_query_cache=False)
    query_job = client.query(query, job_config=job_config)
    return query_job.to_dataframe()


def get_last_timestamp(transcript) -> tuple[float | None, str | None]:
    """
    Extract the latest available timestamp from a transcript (end_time preferred, otherwise start_time).
    Returns (timestamp_in_seconds, error_message).
    """
    if not transcript:
        return None, "Empty transcript"

    try:
        data = json.loads(transcript)
        lines = data.get("lines", [])
        if not lines:
            return None, "No lines in transcript"

        # Try end_time first
        end_times = [line.get("end_time") for line in lines if line.get("end_time") is not None]
        if end_times:
            return max(end_times), None

        # Fall back to start_time
        start_times = [line.get("start_time") for line in lines if line.get("start_time") is not None]
        if start_times:
            return max(start_times), "Used start_time (no end_time available)"

        return None, "No valid timestamps found"

    except json.JSONDecodeError:
        return None, "Invalid JSON"
    except (KeyError, TypeError) as e:
        return None, f"Parsing error: {str(e)}"


def process_object_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add object_id and object_type columns based on whichever ID field is non-null.
    """
    id_columns = {
        "question_id": "question",
        "user_id": "user",
        "meeting_id": "meeting",
        "thread_id": "thread",
        "decision_process_id": "decision_process",
        "collaborator_id": "collaborator",
        "organization_id": "organization",
    }

    df["object_id"] = None
    df["object_type"] = None

    for idx, row in df.iterrows():
        for col, obj_type in id_columns.items():
            if pd.notna(row[col]):
                df.at[idx, "object_id"] = row[col]
                df.at[idx, "object_type"] = obj_type
                break

    return df


def calculate_costs(row: pd.Series) -> dict[str, float]:
    """
    Compute input/output/total cost for a row based on token usage and pricing config.
    Fallbacks to a default model if not recognized.
    """
    model = row.get("llm_model", DEFAULT_MODEL)
    if model not in MODEL_PRICING:
        model = DEFAULT_MODEL

    pricing = MODEL_PRICING[model]
    input_cost = (row["input_tokens"] * pricing["input_cost_per_m"]) / 1_000_000
    output_cost = (row["output_tokens"] * pricing["output_cost_per_m"]) / 1_000_000

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


async def get_recall_usage_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch meeting usage data from Recall AI, day by day, returning a DataFrame.
    Recall API uses US Eastern time, which we convert to UTC.
    """
    client = RecallAIClient()
    all_usage = []
    eastern = pytz.timezone("America/New_York")

    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + pd.Timedelta(days=1)
        try:
            # Convert to Eastern time for the API request
            current_eastern = current_date.astimezone(eastern)
            next_eastern = next_date.astimezone(eastern)

            usage = await client.get_billing_usage(current_eastern, next_eastern)
            if usage:
                # If "bot_total" is returned, store it
                if "bot_total" in usage:
                    all_usage.append(
                        {
                            "date": next_date.date(),
                            "recall_meeting_length": usage["bot_total"],
                            "meeting_id": f"bot_total_{current_date.date()}",
                            "created_at": next_date,
                        }
                    )
                # If "results" is returned, store those
                elif "results" in usage and usage["results"]:
                    for result in usage["results"]:
                        if "timestamp" in result:
                            dt = pd.to_datetime(result["timestamp"]).tz_localize(eastern)
                            result["created_at"] = dt.tz_convert(pytz.UTC)
                        result["date"] = current_date.date()
                    all_usage.extend(usage["results"])
        except Exception as e:
            logger.error(f"Error fetching Recall AI data for {current_date}: {e}")

        current_date = next_date

    return pd.DataFrame(all_usage) if all_usage else pd.DataFrame()


def process_meeting_data(
    meeting_data: pd.DataFrame, recall_data: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process meeting transcripts to estimate meeting_length in seconds,
    return (meeting_data, failed_transcripts, recall_usage_df).
    """
    meeting_data = meeting_data.copy()

    if "created_at" in meeting_data:
        meeting_data["created_at"] = pd.to_datetime(meeting_data["created_at"]).dt.tz_convert(pytz.UTC)

    # Extract transcript timestamps
    results = meeting_data["processed_transcript"].apply(get_last_timestamp)
    meeting_data["meeting_length"] = results.apply(lambda x: x[0]).fillna(0)
    meeting_data["processing_error"] = results.apply(lambda x: x[1])

    failed_transcripts = meeting_data[meeting_data["processing_error"].notna()].copy()

    # Process Recall data if provided
    recall_usage = pd.DataFrame()
    if recall_data is not None and not recall_data.empty:
        recall_usage = recall_data.copy()

        # If the data has a 'bot_total' format, it's already in recall_meeting_length
        if "duration" in recall_usage.columns and "recall_meeting_length" not in recall_usage.columns:
            recall_usage["recall_meeting_length"] = recall_usage["duration"]

        if "recall_meeting_length" not in recall_usage.columns:
            return meeting_data, failed_transcripts, pd.DataFrame()

        # Aggregate usage by created_at
        recall_usage = (
            recall_usage.groupby("created_at", dropna=False)
            .agg({"recall_meeting_length": "sum", "meeting_id": "count"})
            .reset_index()
        )
        recall_usage["recall_hours"] = recall_usage["recall_meeting_length"] / 3600

    return meeting_data, failed_transcripts, recall_usage
