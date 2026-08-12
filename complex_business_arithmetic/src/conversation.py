import time
from anthropic.types import Message

from .models import TestCase, LLMCalculationRequest
from common.prompt_template_engine import build_prompt
from .settings import settings
from .llm_tools import batch_calculation_tool
from .handle_llm_response import handle_llm_response
from .instruct_llm import ainstruct_llm


LLM_TEMPERATURE = 0.0


async def run_conversation(test_case: TestCase) -> tuple[list[dict], list[LLMCalculationRequest], list[str]]:
    """
    Start the conversation with the LLM model.
    """
    print("Starting conversation with LLM model...")

    # System and initial user prompt setup
    # system_prompt = build_prompt("system_cautionary.txt.jinja")
    system_prompt = build_prompt("system.txt.jinja")
    user_prompt = build_prompt("initial_user.txt.jinja", test_case=test_case)

    # Initial message setup
    messages = [{"role": "user", "content": user_prompt}]

    non_calculation_request_assistant_texts: list[str] = []
    calculation_steps: list[LLMCalculationRequest] = []
    num_llm_responses = 0
    cumulative_num_input_tokens = 0
    cumulative_num_output_tokens = 0
    last_llm_response_num_input_tokens = 0

    conversation_start_time = time.time()

    # Start conversational loop
    while True:
        print("STATUS: Sending request to LLM...")
        request_start_time = time.time()
        llm_response: Message = await ainstruct_llm(
            system_prompt=system_prompt,
            messages=messages,
            response_model=None,
            llm_model=settings.llm_model,
            temperature=LLM_TEMPERATURE,
            tools=[batch_calculation_tool],
        )
        request_end_time = time.time()

        # Print response details
        print(f"STATUS: Number of input tokens = {llm_response.usage.input_tokens}")
        print(f"STATUS: Number of output tokens = {llm_response.usage.output_tokens}")
        print(f"STATUS: Time taken for request = {request_end_time - request_start_time:.2f} seconds")

        # Update cumulative totals
        num_llm_responses += 1
        cumulative_num_input_tokens += llm_response.usage.input_tokens
        cumulative_num_output_tokens += llm_response.usage.output_tokens
        last_llm_response_num_input_tokens = llm_response.usage.input_tokens

        # Handle response
        new_messages: list[dict]
        new_calculation_steps: list[LLMCalculationRequest]
        new_non_calculation_request_assistant_texts: list[str]
        new_messages, new_calculation_steps, new_non_calculation_request_assistant_texts = handle_llm_response(
            llm_response
        )

        # Update messages and lists to track other things
        messages.extend(new_messages)
        calculation_steps.extend(new_calculation_steps)
        non_calculation_request_assistant_texts.extend(new_non_calculation_request_assistant_texts)

        # Print new messages to user
        for message in new_messages:
            print(f"MESSAGE: {message['role'].upper()}: {message['content']}")

        # If no calculations were performed, get input from the user to type in
        calculations_were_performed = len(new_calculation_steps) > 0
        if not calculations_were_performed:
            print("STATUS: No calculations were performed. Asking user for input.")
            user_input = input("Enter your response (enter 'exit' to exit): ")
            messages.append({"role": "user", "content": user_input})
            if user_input.lower() == "exit":
                print("Exiting conversation...")
                break

    conversation_end_time = time.time()

    # Update test case metric properties
    test_case.run_duration_s = conversation_end_time - conversation_start_time
    test_case.num_calculation_steps = len(calculation_steps)
    test_case.num_messages = len(messages)
    test_case.num_llm_responses = num_llm_responses
    test_case.cumulative_num_input_tokens = cumulative_num_input_tokens
    test_case.cumulative_num_output_tokens = cumulative_num_output_tokens
    test_case.last_llm_response_num_input_tokens = last_llm_response_num_input_tokens

    print(f"STATUS: Total time taken for conversation = {test_case.run_duration_s:.2f} seconds")

    return messages, calculation_steps, non_calculation_request_assistant_texts
