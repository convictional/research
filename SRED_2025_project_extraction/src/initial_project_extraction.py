import asyncio

from .models import SourceContent, ResearchAndDevelopmentProject, ResearchAndDevelopmentProjects
from .prompts.engine import build_prompt
from .utils.async_helper import limited_task, execute_tasks_with_manual_pbar
from .utils.instruct_llm import ainstruct_llm
from .settings import settings
from .utils.io import dump_to_pickle_file, dump_list_of_objects_to_csv


LLM_TEMPERATURE = 0.0
INITIAL_PROJECTS_OUTPUT_PATH = settings.output_path / "extracted_projects" / "initial_projects_extract.pkl"
INITIAL_PROJECTS_CSV_OUTPUT_PATH = settings.output_path / "extracted_projects" / "initial_projects_extract.csv"


async def initial_fine_grained_project_extraction(
    content_data: list[SourceContent], max_concurrent_tasks: int = 30, delay_between_tasks: float = 0.1
):
    """
    Initial extraction of fine-grained projects from source content.
    """
    print("Starting initial fine-grained project extraction...")

    # Build prompts
    system_prompt = build_prompt("initial_extraction/system.txt.jinja")
    user_prompts = [build_prompt("initial_extraction/user.txt.jinja", content=content) for content in content_data]

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = [
        limited_task(
            ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=ResearchAndDevelopmentProjects,
                temperature=LLM_TEMPERATURE,
            ),
            semaphore,
            delay_between_tasks,
        )
        for user_prompt in user_prompts
    ]

    results: list[ResearchAndDevelopmentProjects] = await execute_tasks_with_manual_pbar(tasks)

    projects: list[ResearchAndDevelopmentProject] = []
    for r in results:
        projects.extend(r.projects)

    print(f"Number of projects extracted: {len(projects)}")

    dump_to_pickle_file(projects, INITIAL_PROJECTS_OUTPUT_PATH)
    dump_list_of_objects_to_csv(projects, INITIAL_PROJECTS_CSV_OUTPUT_PATH)
