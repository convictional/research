import pandas as pd
import json

from ..helpers import get_request_bodies_and_remove_goals_from_system_prompt, make_requests_to_llm_using_request_bodies
from ..models import ResponseComparison
from common.io import dump_list_of_objects_to_csv
from .constants import OUTPUT_PATH_ROOT


LLM_TEMPERATURE = 0.0
DECISON_PROCESS_RELATIONS_RESPONSE_RESULTS_PATH = OUTPUT_PATH_ROOT / "decision_process_relations_response_results.csv"


async def process_decision_process_relations_response_requests(request_data_df: pd.DataFrame):
    """
    Process decision process relations response requests.

    Decision process relations responses are the responses for extracting options, criteria, and insights from
    a decision that was extracted from a meeting transcript.

    That is:
    - Filter requests for decision process relations responses
    - Remove goals from system prompts
    - Make requests without goals to the LLM model
    - Collect responses
    - Print results to file to upload to human evals app
    """
    print("Processing decision process relations response requests...")

    # Filter requests for decision process relations responses
    print("Filtering requests for decision process relations responses...")
    decision_process_relations_response_df = request_data_df[
        request_data_df["request_response_model"] == '"DecisionProcessRelationsResponse"'
    ]
    print(f"Number of decision process relations response requests: {len(decision_process_relations_response_df)}")

    parsed_request_bodies_without_goals: list[dict] = get_request_bodies_and_remove_goals_from_system_prompt(
        decision_process_relations_response_df
    )

    responses_without_goals: list[dict] = await make_requests_to_llm_using_request_bodies(
        parsed_request_bodies_without_goals, temperature=LLM_TEMPERATURE
    )

    # Collect results into ResponseComparison objects
    print("Collecting results into ResponseComparison objects...")
    response_comparisons: list[ResponseComparison] = []
    for (_, row), request_body_without_goals, response_without_goals in zip(
        decision_process_relations_response_df.iterrows(), parsed_request_bodies_without_goals, responses_without_goals
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
                main_response_with_goals=_build_main_response_from_response_body(db_response_body_with_goals),
                main_response_without_goals=_build_main_response_from_response_body(response_without_goals),
            )
        )

    # Print results to file to upload to human evals app
    print("Printing results to file to upload to human evals app...")
    dump_list_of_objects_to_csv(response_comparisons, DECISON_PROCESS_RELATIONS_RESPONSE_RESULTS_PATH)


def _build_main_response_from_response_body(response_body: dict) -> str:
    """
    Build the main response from the response body.
    """
    try:
        response_result = response_body.get("content", [{}])[0].get("input", {})
        result_options = response_result.get("options", [])
        result_criteria = response_result.get("criteria", [])
        result_insights = response_result.get("insights", [])

        main_response_string = ""

        # Add options
        main_response_string += "Options:\n-------------------\n"
        for option in result_options:
            main_response_string += (
                f"- Title: {option.get("title", "")}\n  Description: {option.get("description", "")}\n\n"
            )

        # Add criteria
        main_response_string += "Criteria:\n-------------------\n"
        for criterion in result_criteria:
            main_response_string += (
                f"- Title: {criterion.get("title", "")}\n  Description: {criterion.get("description", "")}\n\n"
            )

        # Add insights
        main_response_string += "Insights:\n-------------------\n"
        for insight in result_insights:
            main_response_string += f"- Title: {insight.get("title", "")}\n  User name: {insight.get("user_name", "")}\n  Description: {insight.get("description", "")}\n\n"

        return main_response_string
    except Exception as e:
        print(f"Error building main response from response body: {e}")
        return "ERROR BUILDING MAIN RESPONSE"
