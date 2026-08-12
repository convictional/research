calculation_tool = {
    "name": "perform_calculation",
    "description": "Perform basic arithmetic operations: add, subtract, multiply, or divide two numbers",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform",
            },
            "operands": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "The numbers to operate on",
            },
            "context": {"type": "string", "description": "Explanation of what this calculation represents"},
        },
        "required": ["operation", "operands", "context"],
    },
}
