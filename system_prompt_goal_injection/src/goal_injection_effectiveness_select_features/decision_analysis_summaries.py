import pandas as pd
import json

from ..helpers import get_request_bodies_and_remove_goals_from_system_prompt, make_requests_to_llm_using_request_bodies
from ..models import ResponseComparison
from common.io import dump_list_of_objects_to_csv
from .constants import OUTPUT_PATH_ROOT


DECISON_ANALYSIS_SUMMARIES_RESULTS_PATH = OUTPUT_PATH_ROOT / "decision_analysis_summaries_results.csv"


async def process_decision_analysis_summaries_requests(
    request_data_df: pd.DataFrame,
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
):
    """
    Process decision analysis summaries requests.

    That is:
    - Filter requests for decision analysis summaries
    - Remove goals from system prompts
    - Make requests without goals to the LLM model
    - Collect responses
    - Print results to file to upload to human evals app
    """
    print("Processing decision analysis summaries requests...")

    # Filter requests for decision analysis summaries
    print("Filtering requests for decision analysis summaries...")
    decision_analysis_summaries_df = request_data_df[request_data_df["is_decision_analysis_summary"]]
    print(f"Number of decision analysis summaries requests: {len(decision_analysis_summaries_df)}")

    parsed_request_bodies_without_goals: list[dict] = get_request_bodies_and_remove_goals_from_system_prompt(
        decision_analysis_summaries_df
    )

    responses_without_goals: list[dict] = await make_requests_to_llm_using_request_bodies(
        parsed_request_bodies_without_goals
    )

    # Collect results into ResponseComparison objects
    print("Collecting results into ResponseComparison objects...")
    response_comparisons: list[ResponseComparison] = []
    for (_, row), request_body_without_goals, response_without_goals in zip(
        decision_analysis_summaries_df.iterrows(), parsed_request_bodies_without_goals, responses_without_goals
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
                main_response_with_goals=db_response_body_with_goals["content"][0]["text"],
                main_response_without_goals=response_without_goals.get("content", [{}])[0].get(
                    "text", "TEXT NOT FOUND"
                ),
            )
        )

    # Print results to file to upload to human evals app
    print("Printing results to file to upload to human evals app...")
    dump_list_of_objects_to_csv(response_comparisons, DECISON_ANALYSIS_SUMMARIES_RESULTS_PATH)
