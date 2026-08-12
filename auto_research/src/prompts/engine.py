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
    cleaned = "\n".join(line.strip() for line in text.split("\n"))
    sanitized_newlines = re.sub(r"\n{3,}", "\n\n", cleaned)
    return sanitized_newlines.strip("\n")
