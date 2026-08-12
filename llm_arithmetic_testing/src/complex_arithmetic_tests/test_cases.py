import json
import random

from ..settings import settings
from pydantic import BaseModel, Field

test_cases_file = settings.input_path / "complex_arithmetic_tests" / "complex_arithmetic_test_cases.json"


class OperationTestCase(BaseModel):
    numbers: list[float] = Field(..., title="The numbers in the operation")
    operation_types: list[str] = Field(..., title="The operations to perform")
    operation_symbols: list[str] = Field(..., title="The symbols for the operations")
    true_answer: float = Field(..., title="The true answer to the operation")


def generate_test_cases(num_cases: int) -> list[OperationTestCase]:
    """
    This function generates test cases for complex arithmetic operations and writes them to a JSON file.
    Complex here means multiple operations in a single prompt.
    """
    print("Generating test cases for complex arithmetic operations...")

    possible_operations = ["add", "subtract", "multiply", "divide"]
    test_cases = []

    for _ in range(num_cases):
        # Generate lists of numbers and operations for each test case
        numbers = [round(random.uniform(-100.00, 100.00), 2) for _ in range(6)]
        operations = random.choices(possible_operations, k=5)

        # Ensure no division by zero
        for i, op in enumerate(operations):
            if op == "divide":
                numbers[i + 1] = round(random.uniform(0.01, 100.00), 2)

        test_case = {"numbers": numbers, "operations": operations}
        test_cases.append(test_case)

    # Write to JSON file
    print(f"Writing test cases to {test_cases_file}...")
    with open(test_cases_file, "w") as f:
        json.dump({"test_cases": test_cases}, f, indent=2)

    return load_test_cases()


def load_test_cases() -> list[OperationTestCase]:
    print(f"Loading test cases from {test_cases_file}...")

    with open(test_cases_file, "r") as file:
        data = json.load(file)

    test_cases = []
    for case in data["test_cases"]:
        # Validate the test case
        test_case = OperationTestCase(
            numbers=case["numbers"],
            operation_types=case["operations"],
            operation_symbols=get_operation_symbols(case["operations"]),
            true_answer=get_true_answer_using_pemdas(case),
        )
        test_cases.append(test_case)

    # Access the test cases
    print(f"Loaded {len(test_cases)} test cases.")

    return test_cases


def get_operation_symbols(operation_types: list[str]) -> list[str]:
    symbols = []

    for operation_type in operation_types:
        if operation_type == "add":
            symbol = "+"
        elif operation_type == "subtract":
            symbol = "-"
        elif operation_type == "multiply":
            symbol = "*"
        elif operation_type == "divide":
            symbol = "/"
        else:
            symbol = None

        if symbol is None:
            raise ValueError(f"Invalid operation type: {operation_type}")

        symbols.append(symbol)

    return symbols


def get_true_answer_using_pemdas(case: dict) -> float:
    """
    This function calculates the true answer for a complex arithmetic operation using PEMDAS order of operations.
    Note, I got 100% of this from Claude
    """
    numbers = case["numbers"]
    operations = case["operations"]

    # Create a list of the full expression
    expression = []
    expression.append(numbers[0])
    for i, op in enumerate(operations):
        if op == "multiply":
            expression.append("*")
        elif op == "divide":
            expression.append("/")
        elif op == "add":
            expression.append("+")
        elif op == "subtract":
            expression.append("-")
        expression.append(numbers[i + 1])

    # First handle multiplication and division left to right
    i = 0
    while i < len(expression):
        if expression[i] in ["*", "/"]:
            left = float(expression[i - 1])
            right = float(expression[i + 1])
            if expression[i] == "*":
                result = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result = left / right
            # Replace the operation and numbers with result
            expression[i - 1 : i + 2] = [result]
            i = 0  # Start over to catch any remaining mul/div
        else:
            i += 1

    # Then handle addition and subtraction left to right
    i = 0
    while i < len(expression):
        if expression[i] in ["+", "-"]:
            left = float(expression[i - 1])
            right = float(expression[i + 1])
            if expression[i] == "+":
                result = left + right
            else:
                result = left - right
            expression[i - 1 : i + 2] = [result]
            i = 0  # Start over to catch any remaining add/sub
        else:
            i += 1

    # Round only at the end
    return round(float(expression[0]), 2)
