batch_calculation_tool = {
    "name": "perform_batch_calculations",
    "description": "Calculate multiple basic arithmetic operations in a single request: add, subtract, multiply, or divide numbers. Do not attempt to divide by zero. You can use this for a single calculation or for efficiency when multiple calculations are needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "calculations": {
                "type": "array",
                "items": {
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
                            "maxItems": 100,  # Set a reasonable upper limit
                            "description": "Array/list of the numbers to operate on. Each element must be a single number. For add: sum all numbers. For subtract: start with first number and subtract the rest. For multiply: multiply all numbers. For divide: start with first number and divide by the rest. CANNOT divide by zero.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Explanation of what this calculation represents",
                        },
                        "id": {"type": "string", "description": "Unique identifier for this calculation"},
                    },
                    "required": ["operation", "operands", "id"],
                },
                "minItems": 1,
                "description": "Array of calculations to perform simultaneously",
            },
        },
        "required": ["calculations"],
    },
}
