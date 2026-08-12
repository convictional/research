from .settings import settings
from common.prompt_template_engine import initialize_and_register_prompt_templates

from .run_tests import run_tests


async def main():
    # prompt templates
    initialize_and_register_prompt_templates(settings.root / "src" / "prompts")

    # Run experiment tests
    await run_tests()
