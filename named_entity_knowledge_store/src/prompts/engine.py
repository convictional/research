import re

import jinja2

prompt_templates = None


def register_prompt_templates(env: jinja2.Environment):
    global prompt_templates
    prompt_templates = env


def build_prompt(template: str, **kwargs) -> str:
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
