"""Prompt templates for decision DAG generation and evaluation."""

from pathlib import Path
from common.prompt_template_engine import initialize_and_register_prompt_templates

# Initialize prompt templates for this experiment
PROMPTS_DIR = Path(__file__).parent
initialize_and_register_prompt_templates(PROMPTS_DIR)
