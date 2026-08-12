import asyncio
from pydantic import BaseModel, Field
from typing import Optional, Literal, TypedDict
from pathlib import Path

from ..complex_arithmetic_tests.test_cases import load_test_cases, OperationTestCase
from ..settings import settings, CLAUDE_SONNET, OPENAI_O1, CLAUDE_SONNET_37
from common.instruct_llm import ainstruct_llm, set_async_instructor_client
from common.async_helper import limited_task, execute_tasks_with_manual_pbar
from common.prompt_template_engine import build_prompt
from common.io import dump_list_of_objects_to_csv


max_llm_iterations = 10
LLM_TEMPERATURE = 0.0


class CalculationRequest(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="The arithmetic operation to perform"
    )
    operands: list[float] = Field(description="The numbers to operate on", min_items=2, max_items=2)
    context: str = Field(description="Explanation of what this calculation represents")


class FinalAnswer(BaseModel):
    answer: str = Field(description="The final answer to the user query")
    explanation: str = Field(description="Explanation and description of the final answer")


class LLMResponse(BaseModel):
    reasoning: str = Field(description="Explanation of the next step in solving the problem")
    request_type: Literal["calculation", "final_answer"] = Field(
        description="Whether this is a calculation request or the final answer"
    )
    calculation: Optional[CalculationRequest] = Field(
        None, description="The calculation to perform, if request_type is 'calculation'"
    )
    final_answer: Optional[FinalAnswer] = Field(
        None, description="The final answer, if request_type is 'final_answer'"
    )


class CalculationRecord(TypedDict):
    operation: str
    operands: list[float]
    result: float
    reasoning: str
    context: str


class PrintableResult(BaseModel):
    user_query: str = Field(..., title="The user query")
    llm_calculation_history: str = Field(..., title="The LLM calculation history")
    llm_final_answer_value: str = Field(..., title="The LLM final answer value")
    true_answer: str = Field(..., title="The true answer")
    llm_final_answer_explanation: str = Field(..., title="The LLM final answer explanation")


def get_user_queries() -> tuple[list[str], list[OperationTestCase]]:
    """
    Get user queries from the complex arithmetic source input data
    """
    print("Getting user queries...")

    print("Getting user queries from complex arithmetic test cases...")
    test_cases: list[OperationTestCase] = load_test_cases()

    user_queries = []
    for case in test_cases:
        expression = [str(case.numbers[0])] + [
            case.operation_symbols[i] + " " + str(case.numbers[i + 1]) for i in range(len(case.operation_symbols))
        ]
        user_queries.append(
            "Please calculate the result for this arithmetic operation:\n"
            + " ".join(expression)
            + "\nReturn the result of the operation with 2 decimal place precision."
        )

    print(f"Loaded {len(user_queries)} user queries.")

    return user_queries, test_cases


async def do_calculator_tools_tests(
    user_queries: list[str],
    llm_model: str,
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> list[tuple[FinalAnswer, list[CalculationRecord]]]:
    """
    For each user query, run the LLM model with the user query to solve the user query
    """
    print("Running calculator tools tests...")

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        limited_task(
            do_calculation_problem(user_query=user_query, llm_model=llm_model),
            semaphore,
            delay_between_tasks,
        )
        for user_query in user_queries
    ]

    results: list[tuple[FinalAnswer, list[CalculationRecord]]] = await execute_tasks_with_manual_pbar(tasks)

    return results


async def do_calculation_problem(user_query: str, llm_model: str) -> tuple[FinalAnswer, list[CalculationRecord]]:
    """
    Given a user query, iteratively run the LLM model to solve the user query
    """
    calculation_history: list[CalculationRecord] = []
    for _ in range(max_llm_iterations):
        system_prompt, user_prompt = get_system_and_user_prompts(user_query, llm_model, calculation_history)

        response = await ainstruct_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=LLMResponse,
            llm_model=llm_model,
            temperature=LLM_TEMPERATURE,
        )

        if response.request_type == "calculation":
            calculation = response.calculation
            calculation_result = calculate_result(calculation)
            calculation_record = {
                "operation": calculation.operation,
                "operands": calculation.operands,
                "result": calculation_result,
                "reasoning": response.reasoning,
                "context": calculation.context,
            }
            calculation_history.append(calculation_record)
        else:
            return response.final_answer, calculation_history

    print("Maximum iterations reached")
    return FinalAnswer(answer="", explanation="Maximum iterations reached"), calculation_history


def calculate_result(calculation: CalculationRequest) -> float:
    """
    Calculate the result of the given calculation
    """
    if calculation.operation == "add":
        return calculation.operands[0] + calculation.operands[1]
    elif calculation.operation == "subtract":
        return calculation.operands[0] - calculation.operands[1]
    elif calculation.operation == "multiply":
        return calculation.operands[0] * calculation.operands[1]
    elif calculation.operation == "divide":
        return calculation.operands[0] / calculation.operands[1]


def get_system_and_user_prompts(
    user_query: str, llm_model: str, calculation_history: list[CalculationRecord]
) -> tuple[str, str]:
    """
    Get the system prompt and user prompt for the given LLM model and response history
    """
    if llm_model in [CLAUDE_SONNET, CLAUDE_SONNET_37]:
        system_prompt = build_prompt(
            "calculator_tools_complex_arithmetic_tests/calculator_tools_complex_arithmetic_tests_sonnet_system.txt.jinja"
        )
        user_prompt = build_prompt(
            "calculator_tools_complex_arithmetic_tests/calculator_tools_complex_arithmetic_tests_sonnet_user.txt.jinja",
            user_query=user_query,
            calculation_history=calculation_history,
        )

    return system_prompt, user_prompt


def print_results_csv(
    user_queries: list[str],
    test_cases: list[OperationTestCase],
    results: list[tuple[FinalAnswer, list[CalculationRecord]]],
    output_path: Path,
):
    print("Printing results to CSV...")

    results = [
        PrintableResult(
            user_query=user_query,
            llm_calculation_history="\n".join(
                [
                    f"Calculation {i + 1}:\noperation: {record["operation"]}\noperands: {record["operands"]}\nresult: {record["result"]}\nreasoning: {record["reasoning"]}\ncontext: {record["context"]}\n"
                    for i, record in enumerate(calculation_history)
                ]
            ),
            llm_final_answer_value=final_answer.answer,
            true_answer=str(test_case.true_answer),
            llm_final_answer_explanation=final_answer.explanation,
        )
        for user_query, test_case, (final_answer, calculation_history) in zip(user_queries, test_cases, results)
    ]

    dump_list_of_objects_to_csv(results, output_path)


async def calculator_tools_complex_arithmetic_tests(
    run_claude_35_tests: bool, run_openai_o1_tests: bool, run_claude_37_tests: bool
):
    """
    Solve complex arithmetic problems with deterministic calculator tools given to the LLM
    """

    # get user queries from the source input data
    user_queries, test_cases = get_user_queries()

    # Run tests
    # Anthropic Sonnet 3.5 tests
    if run_claude_35_tests:
        print("Running Anthropic Claude Sonnet 3.5 tests...")

        llm_model = CLAUDE_SONNET
        output_path = (
            settings.output_path
            / "calculator_tools_complex_arithmetic_tests"
            / "calculator_tools_complex_arithmetic_tests_sonnet_35_results.csv"
        )

        set_async_instructor_client(llm_model=llm_model, api_key=settings.anthropic_api_key)
        results = await do_calculator_tools_tests(user_queries, llm_model)
        print_results_csv(user_queries, test_cases, results, output_path)

    # Anthropic Sonnet 3.7 tests
    if run_claude_37_tests:
        print("Running Anthropic Claude Sonnet 3.7 tests...")

        llm_model = CLAUDE_SONNET_37
        output_path = (
            settings.output_path
            / "calculator_tools_complex_arithmetic_tests"
            / "calculator_tools_complex_arithmetic_tests_sonnet_37_results.csv"
        )

        set_async_instructor_client(llm_model=llm_model, api_key=settings.anthropic_api_key)
        results = await do_calculator_tools_tests(user_queries, llm_model)
        print_results_csv(user_queries, test_cases, results, output_path)
