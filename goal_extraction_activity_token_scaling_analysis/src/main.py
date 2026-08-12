from .settings import settings
from common.prompt_template_engine import initialize_and_register_prompt_templates

from .input_data_analysis import input_data_analysis
from .token_counting_comparison.token_counting_data_generation import run_token_counting_data_generation
from .token_counting_comparison.generate_synthetic_text_data import generate_synthetic_text_data_if_needed
from .token_counting_comparison.token_counting_analysis import run_token_counting_analysis
from .data_modelling import run_data_modelling


async def main():
    # prompt templates
    initialize_and_register_prompt_templates(settings.root / "src" / "prompts")

    # Token counting comparison sub-experiment
    # Basically, we want to see how the comparison between tiktoken token counts and anthropic token counts scales
    # This will serve as the transformation when doing the main data analysis, since the token counts were generated using tiktoken,
    # but we use Anthropic in the app for goal extraction
    generate_synthetic_text_data_if_needed()
    await run_token_counting_data_generation()
    run_token_counting_analysis()

    # Run input data analysis
    await input_data_analysis()

    # Run scaling modelling
    await run_data_modelling()
