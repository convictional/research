"""Convert Anthropic API format to OpenAI format for Vertex AI GPT-OSS endpoint."""


def convert_anthropic_tool_to_openai(anthropic_tool: dict) -> dict:
    """
    Convert Anthropic tool format to OpenAI function format.

    Args:
        anthropic_tool: Tool in Anthropic format with 'name', 'description', 'input_schema'

    Returns:
        Tool in OpenAI format with 'type' and 'function'
    """
    return {
        "type": "function",
        "function": {
            "name": anthropic_tool.get("name", ""),
            "description": anthropic_tool.get("description", ""),
            "parameters": anthropic_tool.get("input_schema", {}),
        },
    }


def convert_anthropic_tool_choice_to_openai(anthropic_tool_choice: dict | str) -> dict | str:
    """
    Convert Anthropic tool_choice format to OpenAI format.

    Note: GPT-OSS on Vertex AI does not support forced tool calling,
    so forced tool choices are converted to "auto" instead.

    Args:
        anthropic_tool_choice: Tool choice in Anthropic format

    Returns:
        Tool choice in OpenAI format (always "auto" for GPT-OSS)
    """
    return "auto"


def convert_anthropic_to_openai(anthropic_request: dict, target_model: str = "openai/gpt-oss-120b-maas") -> dict:
    """
    Convert Anthropic API request format to OpenAI format for Vertex AI GPT-OSS.

    Args:
        anthropic_request: Request in Anthropic format
        target_model: Target model name for OpenAI format

    Returns:
        Request in OpenAI format suitable for Vertex AI GPT-OSS endpoint
    """
    openai_request = {"model": target_model, "messages": []}

    if "system" in anthropic_request and anthropic_request["system"]:
        openai_request["messages"].append({"role": "system", "content": anthropic_request["system"]})

    if "messages" in anthropic_request:
        openai_request["messages"].extend(anthropic_request["messages"])

    if "tools" in anthropic_request and anthropic_request["tools"]:
        openai_request["tools"] = [convert_anthropic_tool_to_openai(tool) for tool in anthropic_request["tools"]]

    if "tool_choice" in anthropic_request and anthropic_request["tool_choice"]:
        openai_request["tool_choice"] = convert_anthropic_tool_choice_to_openai(anthropic_request["tool_choice"])

    if "max_tokens" in anthropic_request:
        openai_request["max_tokens"] = anthropic_request["max_tokens"]

    if "temperature" in anthropic_request:
        openai_request["temperature"] = anthropic_request["temperature"]

    if "stream" in anthropic_request:
        openai_request["stream"] = anthropic_request["stream"]

    return openai_request


def calculate_prompt_length(anthropic_request: dict) -> int:
    """
    Calculate total character length of prompt including system and user messages.

    Args:
        anthropic_request: Request in Anthropic format

    Returns:
        Total character count
    """
    length = 0

    if "system" in anthropic_request and anthropic_request["system"]:
        length += len(anthropic_request["system"])

    if "messages" in anthropic_request:
        for msg in anthropic_request["messages"]:
            if isinstance(msg, dict) and "content" in msg:
                length += len(msg["content"])

    return length


def has_tools(request: dict) -> bool:
    """Check if request includes tools."""
    return "tools" in request and bool(request["tools"])


def has_system_prompt(request: dict) -> bool:
    """Check if request includes system prompt."""
    return "system" in request and bool(request["system"])
