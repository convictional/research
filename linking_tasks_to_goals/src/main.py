from .settings import settings
from common.prompt_template_engine import initialize_and_register_prompt_templates
from .convictional_goals import get_convictional_goals
from .models import Goal
from .match_platform_tasks import match_platform_tasks_to_goals


async def main():
    # prompt templates initialization
    initialize_and_register_prompt_templates(settings.root / "src" / "prompts")

    # Get goals objects
    goals: list[Goal] = get_convictional_goals(load_from_cache=True)

    # Experiment Part 1
    # Platform tasks <> goals mapping
    await match_platform_tasks_to_goals(goals)

    # do GitHub issues <> tasks mapping tests, including comments

    # Play with Github data in the content table and try matching it to goals
