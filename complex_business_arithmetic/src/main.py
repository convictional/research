from .settings import settings
from common.prompt_template_engine import initialize_and_register_prompt_templates
from .source_data import load_test_case_from_file
from .models import TestCaseFilter
from .data_vis.data_vis import plot_data
from .analyze_test_case import analyze_test_case


async def main():
    # prompt templates initialization
    initialize_and_register_prompt_templates(settings.root / "src" / "prompts")

    # Make charts of attachments data
    # plot_data()

    # Get example test case problem to process
    test_case_filter = TestCaseFilter(key="id", value=8)
    test_case = load_test_case_from_file(test_case_filter)

    # Do analysis of test case
    await analyze_test_case(test_case)
