from .settings import settings
from common.prompt_template_engine import initialize_and_register_prompt_templates
from .goal_injection_effectiveness_select_features.goal_injection_effectiveness_select_features import (
    goal_injection_effectiveness_select_features,
)


async def main():
    print("Starting main experiment block...")

    # prompt templates initialization
    initialize_and_register_prompt_templates(settings.root / "src" / "prompts")

    # subexperiment: Is goal injection into the system prompt actually effective for select features?
    await goal_injection_effectiveness_select_features()
