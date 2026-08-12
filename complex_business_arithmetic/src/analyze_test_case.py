from .models import TestCase, LLMCalculationRequest, AuditorResponse
from .settings import settings
from .instruct_llm import set_async_instructor_client
from .print_results_to_file import print_results_to_file
from .conversation import run_conversation
from .audit_messages import audit_messages_output


async def analyze_test_case(test_case: TestCase):
    """
    Analyze the test case and return the results
    """
    print("Analyzing test case...")
    # Set instructor client
    set_async_instructor_client(api_key=settings.anthropic_api_key)

    # Run conversation with LLM
    messages: list[dict]
    calculation_steps: list[LLMCalculationRequest]
    non_calculation_request_assistant_texts: list[str]
    messages, calculation_steps, non_calculation_request_assistant_texts = await run_conversation(test_case)

    # Audit the messages output using another LLM call
    audit_analysis: AuditorResponse = await audit_messages_output(messages)

    # Print results to file
    print_results_to_file(
        test_case, messages, calculation_steps, non_calculation_request_assistant_texts, audit_analysis
    )
