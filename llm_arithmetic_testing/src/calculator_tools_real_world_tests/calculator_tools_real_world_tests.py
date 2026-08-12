import asyncio
from pydantic import BaseModel, Field
from typing import Optional, Literal
from pathlib import Path

from .test_cases import get_user_query_test_cases, UserQueryTestCase
from ..settings import settings, CLAUDE_SONNET, OPENAI_O1, OPENAI_O1_MINI, CLAUDE_SONNET_37
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
    operands: list[float] = Field(description="The ordered numbers to operate on", min_items=2, max_items=2)
    context: str = Field(description="Explanation of what this calculation represents")


class FinalAnswer(BaseModel):
    answer: str = Field(
        description="The final answer that solves the user's query given the user's goal, and is presented to the user. This answer should be detailed and clear, and include any information or intermediate numbers to inform the user. The answer should be a maximum of 5 sentences."
    )
    answer_reasoning: str = Field(
        description="Reasoning and technical background of the final answer. This should be a comprehensive explanation of the reasoning behind the final answer, and MUST show all intermediate calculations, numbers, and steps taken to arrive at the final answer."
    )


# TODO: could add another option for the LLM, something like an insight, where the LLM could submit information
# that could be useful for future considerations when solving the problem
# But it is unclear how useful it would actually be
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


class ReasoningRecord(BaseModel):
    type: str = Field(..., title="The type of reasoning response")
    details: str = Field(..., title="The details of the reasoning response")


class PrintableResult(BaseModel):
    user_query: str = Field(..., title="The user query")
    user_query_goal: str = Field(..., title="The user query goal")
    llm_reasoning_history: str = Field(..., title="The LLM reasoning history")
    llm_final_answer: str = Field(..., title="The LLM final answer explanation")
    llm_final_answer_reasoning: str = Field(..., title="The LLM final answer reasoning")


async def run_tests(
    user_queries: list[UserQueryTestCase],
    llm_model: str,
    use_calculator_tools: bool = True,
    llm_reasoning_effort: str = "not valid",  # Low, medium, high
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> list[tuple[FinalAnswer, list[ReasoningRecord]]]:
    """
    For each user query, run the LLM model with the user query to solve the user query
    """
    print(
        f"Running calculator tools tests with model: {llm_model} and use_calculator_tools: {use_calculator_tools}, and llm_reasoning_effort: {llm_reasoning_effort}"
    )

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    if use_calculator_tools and llm_model in [CLAUDE_SONNET, CLAUDE_SONNET_37, OPENAI_O1]:
        tasks = [
            limited_task(
                do_problem_with_calculator_tools(
                    user_query=user_query, llm_model=llm_model, llm_reasoning_effort=llm_reasoning_effort
                ),
                semaphore,
                delay_between_tasks,
            )
            for user_query in user_queries
        ]

        results: list[tuple[FinalAnswer, list[ReasoningRecord]]] = await execute_tasks_with_manual_pbar(tasks)

        return results
    elif not use_calculator_tools and llm_model == OPENAI_O1:
        tasks = [
            limited_task(
                do_problem_no_calculator_tools(
                    user_query=user_query, llm_model=llm_model, llm_reasoning_effort=llm_reasoning_effort
                ),
                semaphore,
                delay_between_tasks,
            )
            for user_query in user_queries
        ]

        results: list[tuple[FinalAnswer, list[ReasoningRecord]]] = await execute_tasks_with_manual_pbar(tasks)

        return results
    elif use_calculator_tools and llm_model == OPENAI_O1_MINI:
        tasks = [
            limited_task(
                do_problem_with_calculator_tools(
                    user_query=user_query, llm_model=llm_model, llm_reasoning_effort=llm_reasoning_effort
                ),
                semaphore,
                delay_between_tasks,
            )
            for user_query in user_queries
        ]

        results: list[tuple[FinalAnswer, list[ReasoningRecord]]] = await execute_tasks_with_manual_pbar(tasks)

        return results
    else:
        raise NotImplementedError(
            f"Running tests not implemented for llm_model: {llm_model} and use_calculator_tools: {use_calculator_tools}"
        )


async def do_problem_with_calculator_tools(
    user_query: UserQueryTestCase, llm_model: str, llm_reasoning_effort: str
) -> tuple[FinalAnswer, list[ReasoningRecord]]:
    """
    Given a user query, iteratively run the LLM model to solve the user query using calculator tools
    """
    reasoning_history: list[ReasoningRecord] = []
    for _ in range(max_llm_iterations):
        system_prompt, user_prompt = get_system_and_user_prompts(user_query, llm_model, reasoning_history)

        response = await ainstruct_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=LLMResponse,
            llm_model=llm_model,
            temperature=LLM_TEMPERATURE,
            reasoning_effort=llm_reasoning_effort,
        )

        if response.request_type == "calculation":
            calculation = response.calculation
            calculation_result = calculate_result(calculation)
            reasoning_result = ReasoningRecord(
                type="calculation",
                details=f"operation: {calculation.operation}\noperands: {calculation.operands}\nresult: {calculation_result}\nreasoning reason: {response.reasoning}\ncontext: {calculation.context}",
            )
            reasoning_history.append(reasoning_result)
        else:
            return response.final_answer, reasoning_history

    print("Maximum iterations reached")
    return FinalAnswer(answer="", answer_reasoning="Maximum iterations reached"), reasoning_history


async def do_problem_no_calculator_tools(
    user_query: UserQueryTestCase, llm_model: str, llm_reasoning_effort: str
) -> tuple[FinalAnswer, list[ReasoningRecord]]:
    """
    Given a user query, iteratively run the LLM model to solve the user query without calculator tools
    """
    reasoning_history: list[ReasoningRecord] = []
    system_prompt, user_prompt = get_system_and_user_prompts(
        user_query, llm_model, reasoning_history, use_calculator_tools=False
    )

    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=FinalAnswer,
        llm_model=llm_model,
        temperature=LLM_TEMPERATURE,
        reasoning_effort=llm_reasoning_effort,
    )

    return response, reasoning_history


def get_system_and_user_prompts(
    user_query: str, llm_model: str, reasoning_history: list[ReasoningRecord], use_calculator_tools: bool = True
) -> tuple[str, str]:
    """
    Get the system prompt and user prompt for the given LLM model and response history
    """
    if llm_model in [CLAUDE_SONNET, CLAUDE_SONNET_37] and use_calculator_tools:
        system_prompt = build_prompt(
            "calculator_tools_real_world_tests/calculator_tools_real_world_tests_sonnet_system.txt.jinja"
        )
        user_prompt = build_prompt(
            "calculator_tools_real_world_tests/calculator_tools_real_world_tests_sonnet_user.txt.jinja",
            user_query=user_query,
            reasoning_history=reasoning_history,
        )
    elif llm_model == OPENAI_O1 and use_calculator_tools:
        system_prompt = build_prompt(
            "calculator_tools_real_world_tests/calculator_tools_real_world_tests_openai_o1_system.txt.jinja"
        )
        user_prompt = build_prompt(
            "calculator_tools_real_world_tests/calculator_tools_real_world_tests_openai_o1_user.txt.jinja",
            user_query=user_query,
            reasoning_history=reasoning_history,
        )
    elif llm_model == OPENAI_O1 and not use_calculator_tools:
        system_prompt = build_prompt(
            "calculator_tools_real_world_tests/calculator_tools_real_world_tests_openai_o1_no_calculator_tools_system.txt.jinja"
        )
        user_prompt = build_prompt(
            "calculator_tools_real_world_tests/calculator_tools_real_world_tests_openai_o1_no_calculator_tools_user.txt.jinja",
            user_query=user_query,
        )
    elif llm_model == OPENAI_O1_MINI and use_calculator_tools:
        system_prompt = None
        user_prompt = build_prompt(
            "calculator_tools_real_world_tests/calculator_tools_real_world_tests_openai_o1_mini_user.txt.jinja",
            user_query=user_query,
            reasoning_history=reasoning_history,
        )
    else:
        raise ValueError(
            f"When building prompts, unknown LLM model: {llm_model} for use_calculator_tools: {use_calculator_tools}"
        )

    return system_prompt, user_prompt


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


def print_results_csv(
    user_queries: list[UserQueryTestCase],
    results: list[tuple[FinalAnswer, list[ReasoningRecord]]],
    output_path: Path,
):
    print("Printing results to CSV...")

    results = [
        PrintableResult(
            user_query=user_query.query,
            user_query_goal=user_query.goal,
            llm_reasoning_history="\n".join(
                [
                    f"Reasoning step {i + 1}:\nStep type: {record.type}\n{record.details}\n"
                    for i, record in enumerate(calculation_history)
                ]
            ),
            llm_final_answer=final_answer.answer,
            llm_final_answer_reasoning=final_answer.answer_reasoning,
        )
        for user_query, (final_answer, calculation_history) in zip(user_queries, results)
    ]

    dump_list_of_objects_to_csv(results, output_path)


async def calculator_tools_real_world_tests(
    run_claude_35_tests: bool,
    run_openai_o1_tests: bool,
    run_openai_o1_no_calculator_tools_tests: bool,
    run_openai_o1_mini_tests: bool,
    run_claude_37_tests: bool,
):
    """
    Give the LLM a calculator tool and test it on real world test cases

    There are 3 options to run:
    - Sonnet 3.5 with calculator tools
    - OpenAI o1 with calculator tools
    - OpenAI o1 without calculator tools
    - OpenAI o1 mini with calculator tools
    - Sonnet 3.7 with calculator tools
    """
    print("Running calculator tools with real world test cases...")

    user_queries: list[UserQueryTestCase] = get_user_query_test_cases()

    # Run tests
    # Anthropic Sonnet 3.5 tests
    if run_claude_35_tests:
        print("Running Anthropic Claude Sonnet 3.5 with calculator tools tests...")

        llm_model = CLAUDE_SONNET
        output_path = (
            settings.output_path
            / "calculator_tools_real_world_tests"
            / "calculator_tools_real_world_tests_sonnet_35_results.csv"
        )

        set_async_instructor_client(llm_model=llm_model, api_key=settings.anthropic_api_key)
        results = await run_tests(user_queries=user_queries, llm_model=llm_model)
        print_results_csv(user_queries, results, output_path)

    # Anthropic Sonnet 3.7 tests
    if run_claude_37_tests:
        print("Running Anthropic Claude Sonnet 3.7 with calculator tools tests...")

        llm_model = CLAUDE_SONNET_37
        output_path = (
            settings.output_path
            / "calculator_tools_real_world_tests"
            / "calculator_tools_real_world_tests_sonnet_37_results.csv"
        )

        set_async_instructor_client(llm_model=llm_model, api_key=settings.anthropic_api_key)
        results = await run_tests(user_queries=user_queries, llm_model=llm_model)
        print_results_csv(user_queries, results, output_path)

    # Open AI o1 with calculator tools tests
    if run_openai_o1_tests:
        print("Running OpenAI o1 with calculator tools tests...")

        llm_model = OPENAI_O1
        output_path = (
            settings.output_path
            / "calculator_tools_real_world_tests"
            / "calculator_tools_real_world_tests_openai_o1_results.csv"
        )

        set_async_instructor_client(
            llm_model=llm_model, api_key=settings.openai_api_key, openai_organization=settings.openai_organization
        )
        results = await run_tests(user_queries=user_queries, llm_model=llm_model, llm_reasoning_effort="high")
        print_results_csv(user_queries, results, output_path)

    # Open AI o1 without calculator tools tests
    if run_openai_o1_no_calculator_tools_tests:
        print("Running OpenAI o1 without calculator tools tests...")

        llm_model = OPENAI_O1
        output_path = (
            settings.output_path
            / "calculator_tools_real_world_tests"
            / "calculator_tools_real_world_tests_openai_o1_no_calculator_tools_results.csv"
        )

        set_async_instructor_client(
            llm_model=llm_model, api_key=settings.openai_api_key, openai_organization=settings.openai_organization
        )
        results = await run_tests(
            user_queries=user_queries, llm_model=llm_model, use_calculator_tools=False, llm_reasoning_effort="high"
        )
        print_results_csv(user_queries, results, output_path)

    # Open AI o1 mini tests
    if run_openai_o1_mini_tests:
        print("Running OpenAI o1 mini tests...")

        llm_model = OPENAI_O1_MINI
        output_path = (
            settings.output_path
            / "calculator_tools_real_world_tests"
            / "calculator_tools_real_world_tests_openai_o1_mini_results.csv"
        )

        set_async_instructor_client(
            llm_model=llm_model, api_key=settings.openai_api_key, openai_organization=settings.openai_organization
        )
        results = await run_tests(user_queries=user_queries, llm_model=llm_model)
        print_results_csv(user_queries, results, output_path)
