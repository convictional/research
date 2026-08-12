from .source_data import get_llmrequest_data_from_bigquery
from .meeting_summary_requests import process_meeting_summary_requests
from .decision_analysis_unwritten_rules import process_decision_analysis_unwritten_rules_requests
from .decision_analysis_summaries import process_decision_analysis_summaries_requests
from .decision_process_relations_response_requests import process_decision_process_relations_response_requests


async def goal_injection_effectiveness_select_features():
    """
    Test whether or not injecting goals into the system prompt is actually useful for select features.

    This is a subexperiment of the system_prompt_goal_injection experiment.

    Features:
    - Meeting summary requests
    - Decision analysis unwritten rules requests
    - Decision analysis summary requests
    - Decision process relations response requests (decision processes extracted from meetings)

    Basically, the `llmrequest` table will be queried for a list of requests that have been made to the LLM model.
    We will make requests to the LLM model without any goals injected into the system prompt
    (i.e. modifyy the system prompt in the `llmrequest` table to not include any goals).
    The resulting response will serve as a comparison to the previous responses where goals were injected into the system prompt.

    Thus, we will have a set of responses with goals injected into the system prompt and a set of responses without goals injected into the system prompt.
    We can then upload these responses to our human evals app to get feedback on the effectiveness of goal injection.
    """
    print("Running goal_injection_effectiveness subexperiment...")

    # Get `llmrequest` data from BigQuery
    request_data_df = get_llmrequest_data_from_bigquery(load_from_cache=True)

    # Process meeting summary requests
    # await process_meeting_summary_requests(request_data_df)

    # Process decision analysis unwritten rules requests
    # await process_decision_analysis_unwritten_rules_requests(request_data_df)

    # Process decision analysis summary requests
    # await process_decision_analysis_summaries_requests(request_data_df)

    # Process decision process relations response requests
    await process_decision_process_relations_response_requests(request_data_df)
