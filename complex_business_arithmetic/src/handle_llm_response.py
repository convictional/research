from anthropic.types import Message
from .models import LLMCalculationRequest
import json


def handle_llm_response(response: Message) -> tuple[list[dict], list[LLMCalculationRequest], list[str]]:
    """
    Handle the response from the LLM model
    """
    print("STATUS: Handling LLM response...")

    # Uncomment below for debugging
    # print(response)

    new_messages: list[dict] = []
    new_calculation_steps: list[LLMCalculationRequest] = []

    new_non_calculation_request_assistant_texts: list[str] = [
        content_item.text for content_item in response.content if content_item.type == "text"
    ]

    # Check for any tools calls
    tool_calls: list[dict] = []
    for content_item in response.content:
        if content_item.type == "tool_use":  # type(content_item) == ToolUseBlock
            tool_calls.append(
                {
                    "name": content_item.name,
                    "id": content_item.id,
                    "input": content_item.input,
                }
            )

    if tool_calls:
        print(f"STATUS: There are {len(tool_calls)} tool calls to process")

        # Process tool calls
        tool_results: list[dict[str, str]]
        llm_calculations: list[LLMCalculationRequest]
        tool_results, llm_calculations = handle_tool_calls(tool_calls)

        new_messages.append(
            {"role": "assistant", "content": convert_structured_content_to_raw_message_format(response.content)}
        )
        new_messages.append({"role": "user", "content": tool_results})
        new_calculation_steps.extend(llm_calculations)
    else:
        print("STATUS: No tool calls to process")
        for c in response.content:
            if c.type == "text":
                new_messages.append({"role": "assistant", "content": c.text})

    return new_messages, new_calculation_steps, new_non_calculation_request_assistant_texts


def handle_tool_calls(tool_calls: list[dict]) -> tuple[list[dict[str, str]], list[LLMCalculationRequest]]:
    """
    Handle tool calls
    """
    tool_results: list[dict[str, str]] = []
    llm_calculations: list[LLMCalculationRequest] = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_id = tool_call["id"]
        tool_input = tool_call["input"]

        if tool_name == "perform_batch_calculations":
            print("STATUS: Executing tool call: perform_batch_calculations")
            input_calculations = tool_input["calculations"]
            print(f"STATUS: There are {len(input_calculations)} calculations to process")

            for input_calculation in input_calculations:
                # convert to calculation request model
                llm_calculation = LLMCalculationRequest(**input_calculation)
                llm_calculation.tool_use_id = tool_id

                llm_calculation.calculate()

                llm_calculations.append(llm_calculation)

            tool_result_content_dict = {
                "results": [
                    {
                        "id": llm_calculation.id,
                        "operation": llm_calculation.operation,
                        "result": str(llm_calculation.result),
                    }
                    for llm_calculation in llm_calculations
                ],
                "status": "success",
                "calculation_count": str(len(llm_calculations)),
            }

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(tool_result_content_dict),  # needs to be a string to send back as a message
                }
            )

    return tool_results, llm_calculations


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
