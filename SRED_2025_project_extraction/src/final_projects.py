from datetime import datetime

from .settings import settings
from .models import SourceContent, HighLevelResearchAndDevelopmentProject
from .utils.io import load_pickle_file, dump_list_of_objects_to_csv
from .final_summarization_of_projects import FINAL_PROJECTS_OUTPUT_PATH
from .people_involved_in_projects import PeopleInvolvedPerProjectResponse, PEOPLE_PER_PROJECT_OUTPUT_PATH


PROJECTS_CSV_OUTPUT_PATH = settings.output_path / "final_projects" / "final_projects.csv"


def load_all_data() -> tuple[list[HighLevelResearchAndDevelopmentProject], list[PeopleInvolvedPerProjectResponse]]:
    """
    Load all required data.
    - High-level projects from final summarization
    - People involved in projects
    """
    print("Loading all required data...")

    # Load high-level projects from final summarization
    input_projects: list[HighLevelResearchAndDevelopmentProject] = load_pickle_file(FINAL_PROJECTS_OUTPUT_PATH)
    print(f"Loaded {len(input_projects)} high-level projects that are type {type(input_projects[0])}")

    people_involved: list[PeopleInvolvedPerProjectResponse] = load_pickle_file(PEOPLE_PER_PROJECT_OUTPUT_PATH)
    print(f"Loaded people involved for {len(people_involved)} projects that are type {type(people_involved[0])}")

    return input_projects, people_involved


def create_final_project_objects(
    content_data: list[SourceContent],
    input_projects: list[HighLevelResearchAndDevelopmentProject],
    people_involved_per_project: list[PeopleInvolvedPerProjectResponse],
) -> list[HighLevelResearchAndDevelopmentProject]:
    """
    Create final project objects.
    """
    print("Creating final project objects...")

    projects = [
        HighLevelResearchAndDevelopmentProject(
            name=project.name,
            description=project.description,
            source_content_ids=project.source_content_ids,
            source_content_urls=get_source_urls(content_data, project),
            project_owner=people_involved.project_owner,
            project_owner_reason_summary=people_involved.project_owner_reason_summary,
            other_people_involved=people_involved.other_people_involved,
            project_start=get_project_start(content_data, project.source_content_ids),
            project_end=get_project_end(content_data, project.source_content_ids),
        )
        for project, people_involved in zip(input_projects, people_involved_per_project)
    ]

    return projects


def get_source_urls(content_data: list[SourceContent], project: HighLevelResearchAndDevelopmentProject) -> str:
    """
    Get the source URLs for the project
    """
    list_of_urls = []
    for id in project.source_content_ids:
        matching_content = [content for content in content_data if content.content_id == id]
        list_of_urls.append(matching_content[0].url) if matching_content else None

    return "\n".join(list_of_urls)


def get_project_start(content_data: list[SourceContent], source_content_ids: list[str]) -> datetime:
    """
    Get the project start date.
    This is the minimum of all created_at dates of the source content.
    """
    created_ats = [content.created_at for content in content_data if content.content_id in source_content_ids]

    return min(created_ats)


def get_project_end(content_data: list[SourceContent], source_content_ids: list[str]) -> datetime:
    """
    Get the project end date.
    This is the maximum of all last_comment_at dates of the source content.
    """
    last_comment_ats = [
        content.last_comment_at for content in content_data if content.content_id in source_content_ids
    ]

    # Filter out none values
    last_comment_ats = [dt for dt in last_comment_ats if dt is not None]

    return max(last_comment_ats)


def create_final_projects(content_data: list[SourceContent]):
    """
    Create final projects.
    This involves:
    - Loading all required data
    - Consolidating high-level projects, people involved, and other project data together
    - Printing the final projects to a file
    """
    print("Creating final projects and printing to file...")

    # Load all required data
    input_projects: list[HighLevelResearchAndDevelopmentProject]
    people_involved_per_project: list[PeopleInvolvedPerProjectResponse]
    input_projects, people_involved_per_project = load_all_data()

    # Create final project objects
    final_projects: list[HighLevelResearchAndDevelopmentProject] = create_final_project_objects(
        content_data, input_projects, people_involved_per_project
    )

    # Dump to CSV
    dump_list_of_objects_to_csv(final_projects, PROJECTS_CSV_OUTPUT_PATH)
