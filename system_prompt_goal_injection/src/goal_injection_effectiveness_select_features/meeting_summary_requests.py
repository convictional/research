import pandas as pd
import json

from ..helpers import get_request_bodies_and_remove_goals_from_system_prompt, make_requests_to_llm_using_request_bodies
from ..models import ResponseComparison
from common.io import dump_list_of_objects_to_csv
from .constants import OUTPUT_PATH_ROOT


MEETING_SUMMARY_RESULTS_PATH = OUTPUT_PATH_ROOT / "meeting_summary_results.csv"


async def process_meeting_summary_requests(request_data_df: pd.DataFrame):
    """
    Process meeting summary requests.

    That is:
    - Filter requests for meeting summaries
    - Remove goals from system prompts
    - Make requests without goals to the LLM model
    - Collect responses
    - Print results to file to upload to human evals app
    """
    print("Processing meeting summary requests...")

    # Filter requests for meeting summaries
    print("Filtering requests for meeting summaries...")
    meeting_summary_df = request_data_df[request_data_df["request_response_model"] == '"MeetingSummary"']
    print(f"Number of meeting summary requests: {len(meeting_summary_df)}")

    parsed_request_bodies_without_goals: list[dict] = get_request_bodies_and_remove_goals_from_system_prompt(
        meeting_summary_df
    )

    responses_without_goals: list[dict] = await make_requests_to_llm_using_request_bodies(
        parsed_request_bodies_without_goals
    )

    # Collect results into ResponseComparison objects
    print("Collecting results into ResponseComparison objects...")
    response_comparisons: list[ResponseComparison] = []
    for (_, row), request_body_without_goals, response_without_goals in zip(
        meeting_summary_df.iterrows(), parsed_request_bodies_without_goals, responses_without_goals
    ):
        db_request_body_with_goals = json.loads(row["request_body"])
        db_response_body_with_goals = json.loads(row["response_body"])

        response_comparisons.append(
            ResponseComparison(
                db_id=row["id"],
                db_created_at=str(row["created_at"]),
                db_url=row["url"],
                db_response_model=row["request_response_model"],
                db_request_body_with_goals=db_request_body_with_goals,
                db_response_body_with_goals=db_response_body_with_goals,
                request_with_goals_system_prompt=db_request_body_with_goals["system"],
                request_without_goals_system_prompt=request_body_without_goals["system"],
                request_body_without_goals=request_body_without_goals,
                response_body_without_goals=response_without_goals,
                main_response_with_goals=db_response_body_with_goals["content"][0]["input"]["summary"],
                main_response_without_goals=response_without_goals.get("content", [{}])[0]
                .get("input", {})
                .get("summary", "SUMMARY NOT FOUND"),
            )
        )

    # Print results to file to upload to human evals app
    print("Printing results to file to upload to human evals app...")
    dump_list_of_objects_to_csv(response_comparisons, MEETING_SUMMARY_RESULTS_PATH)
