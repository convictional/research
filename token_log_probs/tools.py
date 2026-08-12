from openai.types.chat import ChatCompletionNamedToolChoiceParam, ChatCompletionToolParam

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "structure_decision_function",
            "description": "Structure your consultation for a strategic business decision posed by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short description of the decision being considered"},
                    "description": {
                        "type": "string",
                        "description": "A tweet-length description of the decision being considered.",
                    },
                    "alternatives": {
                        "type": "string",
                        "description": """A bulleted list of alternative options that could be considered which
                    would achieve a similar outcome to the decision being considered. Provide no more than 5,
                    and always a "do nothing" option, a very contrarian option, and a very conservative option.""",
                    },
                    "context": {
                        "type": "string",
                        "description": """Cite any context or sources you used to inform the alternatives you have
                    identified. This could be a list of data sources, or a brief description of the context you
                    used to inform your thinking.""",
                    },
                    "recommended_decision": {
                        "type": "string",
                        "description": """Given all of the context and alternatives you have identified, what is the
                    decision you would recommend and why? Note that the user will always make the final judgement,
                    but you should provide a clear recommendation.""",
                    },
                },
                "required": ["name", "description", "alternatives", "context", "recommended_decision"],
            },
        },
    }
]


tool_choice: ChatCompletionNamedToolChoiceParam = {
    "type": "function",
    "function": {"name": "structure_decision_function"},
}
