from .settings import settings
from common.prompt_template_engine import initialize_and_register_prompt_templates
from .named_entity_knowledge_store_direct_effects import (
    named_entity_knowledge_store_direct_effects_of_decision_options,
)


async def main():
    # prompt templates
    initialize_and_register_prompt_templates(settings.root / "src" / "prompts")

    # Use named entity knowledge store
    await named_entity_knowledge_store_direct_effects_of_decision_options()
