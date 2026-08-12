from jinja2 import Environment, FileSystemLoader

from .settings import settings
from .utils.json import to_json
from .prompts.engine import register_prompt_templates
from .source_data.source_data_bigquery_via_fivetran_model import get_content_data_from_bigquery_via_fivetran_model
from .models import GitHubSourceContent
from .initial_project_extraction import initial_fine_grained_project_extraction
from .fine_grained_into_coarse_grained_projects import fine_grained_projects_into_coarse_grained_projects
from .final_summarization_of_projects import final_summarization_of_projects
from .people_involved_in_projects import people_involved_in_projects
from .final_projects import create_final_projects


async def main():
    """
    Main function for running R&D project extraction.

    There are a few steps:
    1. Get the source content data
    2. Do initial extraction of fine-grained projects from source content
    3. Iteratively summarize fine-grained projects into coarse-grained projects
    4. Final summarization of coarse-grained projects into final projects
    5. Get people involved in the projects
    6. Create final projects and print to file
    """
    # prompt templates
    prompt_templates = Environment(loader=FileSystemLoader(searchpath=settings.root / "src" / "prompts"))
    prompt_templates.filters["to_json"] = to_json
    register_prompt_templates(prompt_templates)

    # Get content source data
    content_data: list[GitHubSourceContent] = get_content_data_from_bigquery_via_fivetran_model(load_from_cache=True)

    # Do initial extraction of fine-grained projects from source content
    await initial_fine_grained_project_extraction(content_data)

    # Iteratively summarize fine-grained projects into coarse-grained projects
    await fine_grained_projects_into_coarse_grained_projects()

    # Final summarization of coarse-grained projects into final projects
    await final_summarization_of_projects()

    # Get people involved in the projects
    await people_involved_in_projects(content_data)

    # Create final projects and print to file
    create_final_projects(content_data)
