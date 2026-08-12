from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import re
import jinja2

from common.json import to_json


prompt_templates = None


def initialize_and_register_prompt_templates(prompts_path: Path):
    """
    This function is a convenience function to initialize and register prompt templates,
    and mimics logic in the app.

    The function takes a path to the prompt templates and initializes the prompt templates.

    For example:
        initialize_and_register_prompt_templates(settings.root / "src" / "prompts")
        where settings.root is the root directory of the project, defined in settings.py.
    """
    # initialize prompt templates
    prompt_templates = Environment(loader=FileSystemLoader(searchpath=prompts_path))
    prompt_templates.filters["to_json"] = to_json
    # register prompt templates
    _register_prompt_templates(prompt_templates)


def _register_prompt_templates(env: jinja2.Environment):
    """
    This function mimics logic in the app, and is needed when setting up prompt templates.
    Specifically, this function assigns a global variable to the environment.
    """
    global prompt_templates
    prompt_templates = env


def build_prompt(template: str, **kwargs) -> str:
    """
    This function mimics logic in the app, and builds a specific prompt to use.

    The function takes a template name and (optional) keyword arguments, and returns a cleaned prompt.

    For example:
        build_prompt("named_entity_knowledge_store/knowledge_store_query_user.txt.jinja", decision=decision, option=option)
    Or, simply just:
        build_prompt("named_entity_knowledge_store/knowledge_store_query_system.txt.jinja")
    """
    if not prompt_templates:
        raise ValueError("Prompt templates have not been registered")

    rendered = prompt_templates.get_template(template).render(**kwargs)
    cleaned = _clean_template(rendered)
    return cleaned


def _clean_template(text: str) -> str:
    # Remove whitespace from the beginning of each line
    cleaned = "\n".join(line.strip() for line in text.split("\n"))

    # Remove any more than two consecutive newlines
    sanitized_newlines = re.sub(r"\n{3,}", "\n\n", cleaned)

    # remove leading & trailing newline characters these are caused by macros without whitespace removal
    return sanitized_newlines.strip("\n")
