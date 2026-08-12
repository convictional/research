from pydantic import BaseModel, Field
from tqdm import tqdm
import asyncio

from .models import GitHubSourceContent, HighLevelResearchAndDevelopmentProject
from .final_summarization_of_projects import FINAL_PROJECTS_OUTPUT_PATH
from .prompts.engine import build_prompt
from .utils.tokens import count_tokens
from .utils.async_helper import limited_task, execute_tasks_with_manual_pbar
from .utils.instruct_llm import ainstruct_llm
from .settings import settings
from .utils.io import load_pickle_file, dump_to_pickle_file, dump_list_of_objects_to_csv


LLM_TEMPERATURE = 0.0
OUTPUT_PATH_ROOT = settings.output_path / "people_involved_in_projects"
PEOPLE_PER_CHUNK_PER_PROJECT_OUTPUT_PATH = OUTPUT_PATH_ROOT / "people_per_chunk_per_project.pkl"
PEOPLE_PER_PROJECT_OUTPUT_PATH = OUTPUT_PATH_ROOT / "people_per_project.pkl"
PEOPLE_PER_PROJECT_CSV_OUTPUT_PATH = OUTPUT_PATH_ROOT / "people_per_project.csv"

CONTENT_ANALYSIS = """
Analysis number: {analysis_number}
People involved: {people_involved}
Reason summary:
{reason_summary}
"""

# Names that should never be attributed as project owners (e.g. executives and
# operations staff who comment on engineering work without owning it). The real
# list was removed before open-sourcing — populate it for your own organisation.
# An empty list is valid; it just means no exclusions are applied.
PEOPLE_WHO_ARE_NOT_PROJECT_OWNERS = """
Person A
Person B
"""


class ProjectSourceContent(BaseModel):
    content_data: list[GitHubSourceContent] = Field(..., title="List of source content data per project.")


class PeopleInvolvedPerContentResponse(BaseModel):
    people_involved: list[str] = Field(
        ...,
        title="List of people involved in the project. There MUST be at least one person in the list.",
    )
    reason_summary: str = Field(
        ...,
        title="A summary of the reasons why people are involved in the project. This should include roles and responsibilities. The summary is detailed and limited to 10 sentences.",
    )


class PeopleInvolvedPerProjectResponse(BaseModel):
    project_owner: str = Field(
        ...,
        title="The person who owns the project. The project owner is NEVER one of the people listed in PEOPLE_WHO_ARE_NOT_PROJECT_OWNERS.",
    )
    project_owner_reason_summary: str = Field(
        ...,
        title="A summary of the reasons why you selected the project owner. The summary is detailed and limited to 5 sentences.",
    )
    other_people_involved: list[str] = Field(
        [], title="List of other people involved in the project. The project owner is NEVER in this list."
    )


def load_final_projects() -> list[HighLevelResearchAndDevelopmentProject]:
    """
    Load final projects from cache
    """
    print("Loading final projects from cache...")

    projects = load_pickle_file(FINAL_PROJECTS_OUTPUT_PATH)
    print(f"Number of final projects loaded: {len(projects)}")

    return projects


def get_project_source_content_data(
    projects: list[HighLevelResearchAndDevelopmentProject], content_data: list[GitHubSourceContent]
) -> list[ProjectSourceContent]:
    """
    Get source content data per project.
    """
    print("Getting source content data per project...")

    project_source_content_data = []
    for project in projects:
        project_content_data = [
            content for content in content_data if content.content_id in project.source_content_ids
        ]
        project_source_content_data.append(ProjectSourceContent(content_data=project_content_data))

    print(f"Loaded content data for {len(project_source_content_data)} projects.")

    print("Number of content items per project:")
    for i, p in enumerate(project_source_content_data):
        print(f"Project {i + 1}: {len(p.content_data)} content items")

    return project_source_content_data


async def extract_people_per_project_per_content_chunk(
    projects: list[HighLevelResearchAndDevelopmentProject],
    project_source_content_data: list[ProjectSourceContent],
    max_concurrent_tasks: int = 30,
    delay_between_tasks: float = 0.1,
):
    """
    Extract people per project per content chunk.

    For each project, loop over the content data dne xtract the peopel involved for each chunk.
    Have to do the loop since the total tokens for the content data is likely too large to process at once.
    These "people involved per content chunk" will be distilled later into a single "people involved per project".
    """
    print("Extracting people per project per content chunk...")

    people_involved_per_content_chunk_per_project: list[list[PeopleInvolvedPerContentResponse]] = []
    for project, project_source_content in tqdm(
        zip(projects, project_source_content_data), total=len(projects), desc="Looping through projects..."
    ):
        content_data = project_source_content.content_data

        system_prompt = build_prompt("people_involved_in_projects/content_people_system.txt.jinja")
        user_prompts = [
            build_prompt(
                "people_involved_in_projects/content_people_user.txt.jinja",
                project_name=project.name,
                project_description=project.description,
                github_content_data=content.text_chunk,
            )
            for content in content_data
        ]

        # below is useful for debugging
        # print number of tokens per user prompt
        # for i, user_prompt in enumerate(user_prompts):
        #     print(f"User prompt {i + 1} has {count_tokens(user_prompt)} tokens")

        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            limited_task(
                ainstruct_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=PeopleInvolvedPerContentResponse,
                    temperature=LLM_TEMPERATURE,
                ),
                semaphore,
                delay_between_tasks,
            )
            for user_prompt in user_prompts
        ]

        people_involved_results = await execute_tasks_with_manual_pbar(tasks)

        people_involved: list[PeopleInvolvedPerContentResponse] = [
            PeopleInvolvedPerContentResponse(
                people_involved=people.people_involved, reason_summary=people.reason_summary
            )
            for people in people_involved_results
        ]

        people_involved_per_content_chunk_per_project.append(people_involved)

    print(
        f"Finished getting people involved per content chunk per project for {len(people_involved_per_content_chunk_per_project)} projects"
    )

    dump_to_pickle_file(people_involved_per_content_chunk_per_project, PEOPLE_PER_CHUNK_PER_PROJECT_OUTPUT_PATH)


async def get_project_owners_and_other_people_involved(
    projects: list[HighLevelResearchAndDevelopmentProject],
    max_concurrent_tasks: int = 30,
    delay_between_tasks: float = 0.1,
):
    """
    For each project, extract the project owner and other people involved.
    This makes use of the "people involved per content chunk" data that was extracted earlier.
    """
    print("Getting project owners and other people involved...")

    # load people involved per content chunk per project
    people_involved_per_content_chunk_per_project: list[list[PeopleInvolvedPerContentResponse]] = load_pickle_file(
        PEOPLE_PER_CHUNK_PER_PROJECT_OUTPUT_PATH
    )
    print(
        f"Loaded people involved per content chunk per project for {len(people_involved_per_content_chunk_per_project)} projects"
    )

    content_analyses = get_list_of_content_data_people_involved(people_involved_per_content_chunk_per_project)

    system_prompt = build_prompt(
        "people_involved_in_projects/project_people_system.txt.jinja",
        people_who_cannot_be_project_owner=PEOPLE_WHO_ARE_NOT_PROJECT_OWNERS,
    )
    user_prompts = [
        build_prompt(
            "people_involved_in_projects/project_people_user.txt.jinja",
            project_name=project.name,
            project_description=project.description,
            content_data_analysis=content_analysis,
        )
        for project, content_analysis in zip(projects, content_analyses)
    ]

    # token counts for user prompts
    print("Token counts for user prompts:")
    for i, user_prompt in enumerate(user_prompts):
        print(f"User prompt {i + 1} has {count_tokens(user_prompt)} tokens")

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        limited_task(
            ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=PeopleInvolvedPerProjectResponse,
                temperature=LLM_TEMPERATURE,
            ),
            semaphore,
            delay_between_tasks,
        )
        for user_prompt in user_prompts
    ]

    people_involved_results = await execute_tasks_with_manual_pbar(tasks)

    people_involved: list[PeopleInvolvedPerProjectResponse] = [
        PeopleInvolvedPerProjectResponse(
            project_owner=people.project_owner,
            project_owner_reason_summary=people.project_owner_reason_summary,
            other_people_involved=people.other_people_involved,
        )
        for people in people_involved_results
    ]

    print(f"Finished getting project owners and other people involved for {len(people_involved)} projects")

    dump_to_pickle_file(people_involved, PEOPLE_PER_PROJECT_OUTPUT_PATH)
    dump_list_of_objects_to_csv(people_involved, PEOPLE_PER_PROJECT_CSV_OUTPUT_PATH)


def get_list_of_content_data_people_involved(
    people_involved_per_content_chunk_per_project: list[list[PeopleInvolvedPerContentResponse]],
) -> list[str]:
    """
    Get list of content data people involved for each project.
    Basically, for each project, join the people involved per content chunk together into a single string.
    """
    print("Getting list of content data people involved for each project...")

    lists_of_content_data_people_involved: list[list[str]] = []
    for people_involved_per_content_chunk in people_involved_per_content_chunk_per_project:
        lists_of_content_data_people_involved.append(
            [
                CONTENT_ANALYSIS.format(
                    analysis_number=i + 1,
                    people_involved=", ".join(people_involved.people_involved),
                    reason_summary=people_involved.reason_summary,
                )
                for i, people_involved in enumerate(people_involved_per_content_chunk)
            ]
        )

    return [
        "\n".join(list_of_content_data_people_involved)
        for list_of_content_data_people_involved in lists_of_content_data_people_involved
    ]


async def people_involved_in_projects(content_data: list[GitHubSourceContent]):
    """
    Get people involved in the projects.
    This is done in multiple steps:
    - First, get source content data per project
    - Second, extract people per project per content chunk.
      That is, for each project, loop over the content data and extract the people involved for each chunk.
    - Third, get project owners and other people involved in each project using the extracted people per content chunk.
    """
    print("Getting people involved in projects...")

    # Load final projects
    projects: list[HighLevelResearchAndDevelopmentProject] = load_final_projects()

    # Get project source content data
    project_source_content_data: list[ProjectSourceContent] = get_project_source_content_data(projects, content_data)

    # Extract people per content chunk per project
    await extract_people_per_project_per_content_chunk(projects, project_source_content_data)

    # Get project owners and other people involved in projects
    await get_project_owners_and_other_people_involved(projects)
