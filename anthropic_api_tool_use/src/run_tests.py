from pathlib import Path
from typing import Tuple

from .settings import settings, CLAUDE_SONNET_37
from common.prompt_template_engine import build_prompt
from .instruct_llm import set_async_instructor_client, ainstruct_llm
from .tools import calculation_tool
from .models import CalculationRequest, FinalAnswer


MAX_ITERATIONS = 10
LLM_TEMPERATURE = 0.0
EXTENDED_THINKING_BUDGET_TOKENS = 2000


async def run_complex_arithmetic_test(run_with_extended_thinking: bool):
    """
    Run complex arithmetic experiment test
    """
    print("Running complex arithmetic test...")

    # System and initial user prompt setup
    system_prompt = build_prompt("complex_arithmetic_test_system.txt.jinja")
    user_prompt = build_prompt("complex_arithmetic_test_user.txt.jinja")

    # Initial message setup
    messages = [{"role": "user", "content": user_prompt}]

    calculation_steps, final_result = await iterative_call_llm_to_get_final_result(
        system_prompt, messages, run_with_extended_thinking
    )

    # Print response for reference. Comment if not needed
    print("Final result:")
    print(final_result)

    # Print output to file
    output_path = settings.output_path / "complex_arithmetic_test_results.txt"
    print_output_to_file(calculation_steps, final_result, output_path)


async def iterative_call_llm_to_get_final_result(
    system_prompt: str, messages: list[dict], run_with_extended_thinking: bool
) -> Tuple[list[CalculationRequest], FinalAnswer]:
    """
    Make iterative calls to the LLM model to get the final result.

    Iterate a maximum of MAX_ITERATIONS times to get the final result.

    Start by sending the initial messages to the LLM model with tools to use.
    If the LLM model does not call any tools, we're done, and we make a final API call to get the final result.
    If the LLM model calls tools, we process the tool calls and prepare tool results, then send the messages back to the LLM model.
    The messages list is updated with new/previous messages and tool results for each iteration.

    If the maximum number of iterations is reached,
    return the calculation steps and a dummy final answer to indicate the maximum iterations were reached.
    """
    # TODO: revisit supporting extended thinking once instructor catches up with the API changes
    print("Iteratively calling LLM to get final result...")

    # Store calculation steps for the final answer
    calculation_steps: list[CalculationRequest] = []

    # Start iterating to get the final result
    for iteration in range(MAX_ITERATIONS):
        print(f"Running iteration {iteration+1}...")

        # Make API call with tools set but no response model
        # If a response model is set, the API call seems to return with that response model, even if tool calls should be made first
        # Even at that, I only see a thinking block in the response for the first iteration, and not for subsequent iterations :confused:
        response = await ainstruct_llm(
            system_prompt=system_prompt,
            messages=messages,
            response_model=None,
            llm_model=CLAUDE_SONNET_37,
            temperature=1.0 if run_with_extended_thinking else LLM_TEMPERATURE,
            tools=[calculation_tool],
            thinking={"type": "enabled", "budget_tokens": EXTENDED_THINKING_BUDGET_TOKENS}
            if run_with_extended_thinking
            else {"type": "disabled"},
        )

        # Print response for reference. Comment if not needed
        print("LLM result:")
        print(response)

        # Check for tool calls in the response
        tool_calls: list[dict] = []
        for content_item in response.content:
            if content_item.type == "tool_use":
                tool_calls.append(
                    {
                        "name": content_item.name,
                        "id": content_item.id,
                        "input": content_item.input,
                    }
                )

        # Print response for reference. Comment if not needed
        print("Tool calls:")
        print(tool_calls)

        # If no tool calls, we're done - make a final API call to get the final result
        if not tool_calls:
            text_content = convert_structured_content_to_raw_message_format(response.content)

            # Add a final message to ensure Claude returns the expected format
            messages.append({"role": "assistant", "content": text_content})
            messages.append(
                {
                    "role": "user",
                    "content": "Now provide the final result with all steps taken in the format of the requested response model",
                }
            )

            # Right now, instructor does not allow for a response model and extended thinking to be used together
            # Might have to change the client mode, and wait for instructor to update
            final_response = await ainstruct_llm(
                system_prompt=system_prompt,
                messages=messages,
                response_model=FinalAnswer,
                llm_model=CLAUDE_SONNET_37,
                temperature=LLM_TEMPERATURE,
            )

            return calculation_steps, final_response

        # If we get here we have tool calls to process. So, process tool calls and prepare tool results
        tool_results: list[dict[str, str]]
        new_calculation_steps: list[CalculationRequest]
        tool_results, new_calculation_steps = handle_tool_calls(tool_calls)
        calculation_steps.extend(new_calculation_steps)

        # Print response for reference. Comment if not needed
        print("Tool results:")
        print(tool_results)

        messages.append(
            {"role": "assistant", "content": convert_structured_content_to_raw_message_format(response.content)}
        )
        messages.append({"role": "user", "content": tool_results})

    # If we get here we've reached the maximum number of iterations
    # Thus, return the calculation steps and a dummy final answer to indicate the maximum iterations were reached
    return calculation_steps, FinalAnswer(answer="", explanation="Maximum iterations reached")


def handle_tool_calls(tool_calls: list[dict]) -> tuple[list[dict[str, str]], list[CalculationRequest]]:
    """
    Handle tool calls
    """
    print("Handling tool calls to get tool results...")

    tool_results: list[dict[str, str]] = []
    new_calculation_steps: list[CalculationRequest] = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["input"]
        tool_id = tool_call["id"]

        if tool_name == "perform_calculation":
            # Convert to calculation request model
            calculation_request = CalculationRequest(**tool_input)

            # Calculate the result deterministically
            calculation_request.calculate()

            # Record this calculation
            new_calculation_steps.append(calculation_request)

            # Create tool result to send back to LLM
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": str(calculation_request.result),  # needs to be a string to send back as a message
                }
            )

    return tool_results, new_calculation_steps


def convert_structured_content_to_raw_message_format(content):
    """
    Convert the structured content from Anthropic API to the format expected (i.e. raw format) in messages.

    We have to do this since instructor returns a structured response, but the API expects raw format messages, for both input and output.
    """
    formatted_content = []

    # For each content item, convert it to the appropriate format based on its type
    for item in content:
        if hasattr(item, "type"):
            if item.type == "text":
                formatted_content.append({"type": "text", "text": item.text})
            elif item.type == "thinking":
                formatted_content.append({"type": "thinking", "thinking": item.thinking, "signature": item.signature})
            elif item.type == "tool_use":
                formatted_content.append(
                    {
                        "type": "tool_use",
                        "id": item.id,
                        "name": item.name,
                        "input": item.input,
                    }
                )

    return formatted_content


def print_output_to_file(calculation_steps: list[CalculationRequest], final_result: FinalAnswer, output_path: Path):
    """
    Print the output to a file for convenience and reference
    """
    print("Printing output to file...")

    output_string = "\n\n".join(
        [
            f"Calculation step {i+1}:\noperation: {step.operation}\noperands: {step.operands}\ncontext: {step.context}\nresult: {step.result} = {step.result}"
            for i, step in enumerate(calculation_steps)
        ]
    )
    output_string += f"\n\nFinal result:\nanswer: {final_result.answer}\nexplanation: {final_result.explanation}"

    with open(output_path, "w") as f:
        f.write(output_string)


async def run_tests():
    """
    Main function for running experiment test

    The test is just a single complex arithmetic problem.
    """
    print("Running tests...")

    # Set instructor client
    set_async_instructor_client(api_key=settings.anthropic_api_key)

    # Complex arithmetic test
    await run_complex_arithmetic_test(run_with_extended_thinking=False)
