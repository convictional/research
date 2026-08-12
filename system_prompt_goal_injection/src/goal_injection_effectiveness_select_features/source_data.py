import pandas as pd

from common.bigquery import query_bq
from common.io import dump_to_pickle_file, load_pickle_file
from ..settings import settings
from .constants import OUTPUT_PATH_ROOT


QUERY = """
with requests as (
  select
    id,
    created_at,
    request_body,
    response_body,
    url,
    JSON_EXTRACT(request_body, '$.tool_choice.name') AS request_response_model,
    JSON_EXTRACT(request_body, '$.system') AS request_system_prompt,
  from `${GCP_PROJECT}.cloudsql_decide_public.llmrequest`
  where
    created_at <= '2025-03-14'
    and created_at >= '2025-02-24'
    and _fivetran_deleted = false
),

filtered_requests as (
  select
    *,
    SUBSTR(
      request_system_prompt,
      STRPOS(request_system_prompt, '</organization_context>') + LENGTH('</organization_context>'),
      300
    ) AS extracted_text,
    request_system_prompt like '%These unwritten rules are implicit principles or norms that influence decision-making but are not formally documented.%' as is_decision_analysis_unwritten_rules,
    request_system_prompt like '%Provide a 3 bullet point analysis on these decisions the organization made.%' as is_decision_analysis_summary,
  from requests
  where 1=1
    and request_system_prompt like '%Mission: Convictional is the infrastructure that powers%' -- filter for Convictional requests
    and request_system_prompt like '%Goals of the organization:%' -- filter for having goals in the system prompt
)

select *
from filtered_requests
"""

LLM_REQUEST_DATA_OUTPUT_PATH = OUTPUT_PATH_ROOT / "llm_requests_data.pkl"


def get_llmrequest_data_from_bigquery(load_from_cache: bool = False) -> pd.DataFrame:
    """
    Get `llmrequest` data from BigQuery.
    """
    print("Getting `llmrequest` data...")

    if load_from_cache:
        print("Loading data from cache...")
        result_df = load_pickle_file(LLM_REQUEST_DATA_OUTPUT_PATH)
    else:
        print("Querying BigQuery...")
        result_df = query_bq(QUERY, settings.gcp_project)
        dump_to_pickle_file(result_df, LLM_REQUEST_DATA_OUTPUT_PATH)

    print(f"Loaded {len(result_df)} rows of `llmrequest` data.")

    print("Response model value counts:")
    print(
        result_df.assign(request_response_model=result_df["request_response_model"].fillna("NULL"))
        .groupby("request_response_model")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_string(index=False)
    )

    print("Decision analysis unwritten rules value counts:")
    print(result_df["is_decision_analysis_unwritten_rules"].value_counts())

    print("Decision analysis summary value counts:")
    print(result_df["is_decision_analysis_summary"].value_counts())

    print("Finished getting `llmrequest` data.")

    return result_df
