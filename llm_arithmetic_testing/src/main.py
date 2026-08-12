from .settings import settings
from common.prompt_template_engine import initialize_and_register_prompt_templates

from .simple_arithmetic_tests.simple_arithmetic_tests import simple_arithmetic_tests
from .complex_arithmetic_tests.complex_arithmetic_tests import complex_arithmetic_tests
from .calculator_tools_complex_arithmetic_tests.calculator_tools_complex_arithmetic_tests import (
    calculator_tools_complex_arithmetic_tests,
)
from .calculator_tools_real_world_tests.calculator_tools_real_world_tests import calculator_tools_real_world_tests


async def main():
    # prompt templates
    initialize_and_register_prompt_templates(settings.root / "src" / "prompts")

    # simple arithmetic tests
    # await simple_arithmetic_tests(
    #     load_test_cases_from_file=True, run_claude_35_tests=False, run_openai_o1_tests=False, run_claude_37_tests=False
    # )

    # complex arithmetic tests
    # await complex_arithmetic_tests(
    #     load_test_cases_from_file=True, run_claude_35_tests=False, run_openai_o1_tests=False, run_claude_37_tests=False
    # )

    # calculator tools
    # await calculator_tools_complex_arithmetic_tests(
    #     run_claude_35_tests=False, run_openai_o1_tests=False, run_claude_37_tests=False
    # )

    # calculator tools with real world test cases
    await calculator_tools_real_world_tests(
        run_claude_35_tests=False,
        run_openai_o1_tests=False,
        run_openai_o1_no_calculator_tools_tests=False,
        run_openai_o1_mini_tests=False,
        run_claude_37_tests=False,
    )
