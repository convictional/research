import asyncio
from pydantic import BaseModel, Field
from pathlib import Path

from ..settings import settings, CLAUDE_SONNET, OPENAI_O1, CLAUDE_SONNET_37
from .test_cases import generate_test_cases, load_test_cases, OperationTestCase
from common.prompt_template_engine import build_prompt
from common.async_helper import limited_task, execute_tasks_with_manual_pbar
from common.instruct_llm import ainstruct_llm, set_async_instructor_client
from common.io import dump_list_of_objects_to_csv


num_test_cases = 500
LLM_TEMPERATURE = 0.0


class OperationResponse(BaseModel):
    result: float = Field(..., title="The result of the arithmetic operation")


class PrintableOperationTestCase(OperationTestCase):
    llm_answer: float = Field(..., title="The answer from the LLM model")
    is_llm_correct: bool = Field(..., title="Whether the LLM answer is correct")
    percent_error: float = Field(..., title="The percent error of the LLM answer, 0-100")


async def do_arithmetic_tests(
    system_prompt: str,
    user_prompts: list[str],
    llm_model: str,
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> list[OperationResponse]:
    print("Running arithmetic tests...")

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        limited_task(
            ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=OperationResponse,
                llm_model=llm_model,
                temperature=LLM_TEMPERATURE,
            ),
            semaphore,
            delay_between_tasks,
        )
        for user_prompt in user_prompts
    ]

    arithmetic_results: list[OperationResponse] = await execute_tasks_with_manual_pbar(tasks)

    return arithmetic_results


def print_results_csv(
    test_cases: list[PrintableOperationTestCase], arithmetic_results: list[OperationResponse], output_path: Path
):
    print("Printing results to CSV...")

    results = [
        PrintableOperationTestCase(
            numbers=test_case.numbers,
            operation_types=test_case.operation_types,
            operation_symbols=test_case.operation_symbols,
            true_answer=test_case.true_answer,
            llm_answer=arithmetic_result.result,
            is_llm_correct=test_case.true_answer == arithmetic_result.result,
            percent_error=abs(test_case.true_answer - arithmetic_result.result)
            * 100.0
            / abs(test_case.true_answer + 0.0000000001),  # add epsilon to avoid division by zero
        )
        for test_case, arithmetic_result in zip(test_cases, arithmetic_results)
    ]

    dump_list_of_objects_to_csv(results, output_path)


async def complex_arithmetic_tests(
    load_test_cases_from_file: bool, run_claude_35_tests: bool, run_openai_o1_tests: bool, run_claude_37_tests: bool
):
    print("Running complex arithmetic tests...")

    # Get test cases
    test_cases = load_test_cases() if load_test_cases_from_file else generate_test_cases(num_test_cases)

    # Build prompts
    system_prompt = build_prompt("complex_arithmetic_tests/complex_arithmetic_tests_system.txt.jinja")
    user_prompts = [
        build_prompt("complex_arithmetic_tests/complex_arithmetic_tests_user.txt.jinja", test_case=test_case)
        for test_case in test_cases
    ]

    # Run tests
    # Anthropic Sonnet 3.5 tests
    if run_claude_35_tests:
        print("Running Anthropic Claude Sonnet 3.5 tests...")

        llm_model = CLAUDE_SONNET
        output_path = (
            settings.output_path / "complex_arithmetic_tests" / "complex_arithmetic_tests_sonnet_35_results.csv"
        )

        set_async_instructor_client(llm_model=llm_model, api_key=settings.anthropic_api_key)
        results = await do_arithmetic_tests(system_prompt, user_prompts, llm_model)
        print_results_csv(test_cases, results, output_path)

    # Anthropic Sonnet 3.7 tests
    if run_claude_37_tests:
        print("Running Anthropic Claude Sonnet 3.7 tests...")

        llm_model = CLAUDE_SONNET_37
        output_path = (
            settings.output_path / "complex_arithmetic_tests" / "complex_arithmetic_tests_sonnet_37_results.csv"
        )

        set_async_instructor_client(llm_model=llm_model, api_key=settings.anthropic_api_key)
        results = await do_arithmetic_tests(system_prompt, user_prompts, llm_model)
        print_results_csv(test_cases, results, output_path)

    # Open AI o1 tests
    if run_openai_o1_tests:
        print("Running OpenAI o1 tests...")

        llm_model = OPENAI_O1
        output_path = settings.output_path / "complex_arithmetic_tests" / "complex_arithmetic_tests_o1_results.csv"

        set_async_instructor_client(
            llm_model=llm_model, api_key=settings.openai_api_key, openai_organization=settings.openai_organization
        )
        results = await do_arithmetic_tests(system_prompt, user_prompts, llm_model)
        print_results_csv(test_cases, results, output_path)
