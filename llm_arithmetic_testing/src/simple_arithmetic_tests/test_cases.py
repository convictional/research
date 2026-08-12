import json
import random

from ..settings import settings
from pydantic import BaseModel, Field


test_cases_file = settings.input_path / "simple_arithmetic_tests" / "simple_arithmetic_test_cases.json"


class OperationTestCase(BaseModel):
    num_a: float = Field(..., title="The first number in the operation")
    num_b: float = Field(..., title="The second number in the operation")
    operation_type: str = Field(..., title="The type of operation to perform")
    operation_symbol: str = Field(..., title="The symbol for the operation")
    true_answer: float = Field(..., title="The true answer to the operation")


def generate_test_cases(num_cases: int) -> list[OperationTestCase]:
    """
    This function generates test cases for simple arithmetic operations and writes them to a JSON file.
    """
    print("Generating test cases for simple arithmetic operations...")

    operations = ["add", "subtract", "multiply", "divide"]
    test_cases = []

    for _ in range(num_cases):
        # For division, ensure we don't divide by zero and result is clean
        operation = random.choice(operations)
        if operation == "divide":
            # Ensure non-zero divisor and reasonable numbers for division
            num_b = round(random.uniform(0.01, 100.00), 2)
            # Make num_a a multiple of num_b to ensure clean division
            num_a = round(random.uniform(-1000.00, 1000.00), 2)
        else:
            num_a = round(random.uniform(-1000.00, 1000.00), 2)
            num_b = round(random.uniform(-1000.00, 1000.00), 2)

        test_case = {"num_a": num_a, "num_b": num_b, "operation_type": operation}
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
            num_a=case["num_a"],
            num_b=case["num_b"],
            operation_type=case["operation_type"],
            operation_symbol=get_operation_symbol(case["operation_type"]),
            true_answer=get_true_answer(case),
        )
        test_cases.append(test_case)

    # Access the test cases
    print(f"Loaded {len(test_cases)} test cases.")

    return test_cases


def get_operation_symbol(operation_type: str) -> str:
    if operation_type == "add":
        return "+"
    elif operation_type == "subtract":
        return "-"
    elif operation_type == "multiply":
        return "*"
    elif operation_type == "divide":
        return "/"

    raise ValueError(f"Invalid operation type: {operation_type}")


def get_true_answer(case: dict) -> int:
    num_a = case["num_a"]
    num_b = case["num_b"]
    operation = case["operation_type"]

    if operation == "add":
        return round(num_a + num_b, 2)
    elif operation == "subtract":
        return round(num_a - num_b, 2)
    elif operation == "multiply":
        return round(num_a * num_b, 2)
    elif operation == "divide":
        return round(num_a / num_b, 2)

    raise ValueError(f"Invalid operation type: {operation}")
