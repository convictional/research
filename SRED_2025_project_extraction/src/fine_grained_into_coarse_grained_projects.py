from pathlib import Path
import random
import asyncio
import json

from .initial_project_extraction import INITIAL_PROJECTS_OUTPUT_PATH
from .utils.io import load_pickle_file, dump_to_pickle_file, dump_list_of_objects_to_csv
from .models import ResearchAndDevelopmentProject, ResearchAndDevelopmentProjects
from .utils.tokens import count_tokens
from .prompts.engine import build_prompt
from .utils.async_helper import limited_task, execute_tasks_with_manual_pbar
from .utils.instruct_llm import ainstruct_llm
from .settings import settings, CLAUDE_HAIKU


LLM_TEMPERATURE = 0.0
NUM_GLEANS_PER_ITERATION = 3
OUTPUT_FILE_PATH_ROOT = settings.output_path / "fine_to_coarse_grained_extracted_projects"
OUTPUT_FILE_NAME_ROOT = "fine_to_coarse_grained_projects"


def load_and_randomize_projects(input_file_path: Path, randomization_strategy: str = "shuffle") -> list[ResearchAndDevelopmentProject]:
    """
    Randomization strategies:
    - "shuffle": random shuffle
    - "sort_by_name": sort by project name
    """
    print("Loading and randomizing projects...")

    projects = load_pickle_file(input_file_path)
    if randomization_strategy == "shuffle":
        random.shuffle(projects)
    elif randomization_strategy == "sort_by_name":
        projects.sort(key=lambda p: p.name)
    else:
        raise ValueError(f"Unknown randomization strategy: {randomization_strategy}")
    print(f"Number of projects loaded: {len(projects)}")

    return projects


def group_projects(
    projects: list[ResearchAndDevelopmentProject], max_size_of_project_groups: int
) -> list[list[ResearchAndDevelopmentProject]]:
    """
    Group projects into groups of size at most max_size_of_project_groups.
    """
    print("Grouping projects...")

    grouped_projects = [
        projects[i : i + max_size_of_project_groups] for i in range(0, len(projects), max_size_of_project_groups)
    ]

    print(f"Grouped projects into {len(grouped_projects)} groups")
    print("Size of each group:")
    for i, p in enumerate(grouped_projects):
        print(f"Group {i + 1}: {len(p)} projects")

    # Want an idea about number of tokens per group of projects
    num_tokens_per_group = [
        count_tokens(
            build_prompt(
                "fine_to_coarse_grained_extraction/user.txt.jinja",
                projects_already_distilled_from_context="\n\n",
                project_json_context_to_analyze="\n\n".join(get_json_dumps_projects(group)),
            )
        )
        for group in grouped_projects
    ]
    print("Number of tokens per group with no current projects:")
    for i, num_tokens in enumerate(num_tokens_per_group):
        print(f"Group {i + 1}: {num_tokens} tokens")

    return grouped_projects


async def get_summarized_projects_from_groups(
    project_groups: list[list[ResearchAndDevelopmentProject]],
    num_gleans: int,
    max_concurrent_tasks: int = 30,
    delay_between_tasks: float = 0.1,
) -> list[ResearchAndDevelopmentProject]:
    """
    Get summarized (coarse-grained) projects from each group of projects.

    For each group, ask LLM to distill coarse-grained projects from the group. Repeat this num_gleans times.
    """
    print("Getting summarized projects from groups...")

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        limited_task(execute_gleans(group, num_gleans), semaphore, delay_between_tasks) for group in project_groups
    ]

    print(f"Executing {len(tasks)} group of projects with {num_gleans} gleans each...")
    results: list[list[ResearchAndDevelopmentProject]] = await execute_tasks_with_manual_pbar(tasks)

    new_projects: list[ResearchAndDevelopmentProject] = []
    for i, result in enumerate(results):
        new_projects.extend(result)
        print(f"Group {i + 1} has {len(result)} new projects")

    print(f"Number of NEW projects extracted: {len(new_projects)}")

    # make elements of new_projects source_content_ids unique
    for p in new_projects:
        p.source_content_ids = sorted(list(set(p.source_content_ids)))

    return new_projects


async def execute_gleans(
    project_group: list[ResearchAndDevelopmentProject], num_gleans: int
) -> list[ResearchAndDevelopmentProject]:
    """
    Execute num_gleans gleans for a group of projects.
    """
    current_new_projects: list[ResearchAndDevelopmentProject] = []

    for i in range(num_gleans):
        system_prompt = build_prompt("fine_to_coarse_grained_extraction/system.txt.jinja")
        user_prompt = build_prompt(
            "fine_to_coarse_grained_extraction/user.txt.jinja",
            projects_already_distilled_from_context="\n\n".join(get_json_dumps_projects(current_new_projects)),
            project_json_context_to_analyze="\n\n".join(get_json_dumps_projects(project_group)),
        )

        new_projects = await ainstruct_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ResearchAndDevelopmentProjects,
            temperature=LLM_TEMPERATURE,
            # override max tokens to be very large to accommodate larger responses
            # Note, this required specifying a (large) timeout in the AsyncAnthropic client creation in instruct_llm.py
            max_tokens=60000,
        )

        # Useful for debugging (bottom 2 lines)
        # print(f"Number of current new projects: {len(current_new_projects)}")
        # print(f"Number of new projects extracted: {len(new_projects.projects)}")

        current_new_projects.extend(new_projects.projects)

    return current_new_projects


def get_json_dumps_projects(projects: list[ResearchAndDevelopmentProject]) -> list[json.dumps]:
    json_projects = [get_json_dump_of_project(p) for p in projects]

    return json_projects


def get_json_dump_of_project(project: ResearchAndDevelopmentProject) -> json.dumps:
    dictionary = {
        "name": project.name,
        "description": project.description,
        "source_content_ids": project.source_content_ids,
    }

    return json.dumps(dictionary, indent=2)


def dump_new_projects_to_files(new_projects: list[ResearchAndDevelopmentProject], iteration: int):
    """
    Dump the new projects for a given iteration to files.
    2 files: pickle and csv
    """
    print(f"Dumping new projects for iteration {iteration} to files...")

    output_pickle_path = OUTPUT_FILE_PATH_ROOT / f"{OUTPUT_FILE_NAME_ROOT}_{iteration}.pkl"
    output_csv_path = OUTPUT_FILE_PATH_ROOT / f"{OUTPUT_FILE_NAME_ROOT}_{iteration}.csv"

    dump_to_pickle_file(new_projects, output_pickle_path)
    dump_list_of_objects_to_csv(new_projects, output_csv_path)


async def summarize_projects_iteration(
    input_file_path: Path,
    max_size_of_project_groups: int,
    iteration: int,
    num_gleans: int,
    randomization_strategy: str = "shuffle",
):
    """
    Single iteration of summarizing projects.

    Specifically:
    - Read in the projects from the input file (initial extraction, previous iteration, etc)
    - Group projects together randomly
    - For each group, ask LLM to distill a coarse-grained project from the group. Repeat this num_gleans times.
    """
    print(f"Starting iteration {iteration} of summarizing projects...")

    # Read in the projects
    projects: list[ResearchAndDevelopmentProject] = load_and_randomize_projects(input_file_path, randomization_strategy)

    # Group projects together
    project_groups: list[list[ResearchAndDevelopmentProject]] = group_projects(projects, max_size_of_project_groups)

    # Get coarse-grained projects from each group
    new_projects: list[ResearchAndDevelopmentProject] = await get_summarized_projects_from_groups(
        project_groups, num_gleans
    )

    # Dump the new projects to files
    dump_new_projects_to_files(new_projects, iteration)


async def fine_grained_projects_into_coarse_grained_projects():
    """
    Iterative scheme to convert fine-grained projects into coarse-grained projects.

    For each iteration:
    - Group fine-grained projects together randomly
    - For each group, ask LLM to distill coarse-grained project from the group
    - Repeat this until number of coarse-grained projects stabilizes

    NOTE, IMPORTANT, if max_size_of_project_groups is too large and the there aren't many projects to group together,
    the LLM output token limit can be reached. In that case, reduce the max_size_of_project_group
    """
    print("Starting to summarize fine-grained projects into coarse-grained projects...")

    # Number of input fine grained projects: 132

    # # iteration 1 # 97 projects result
    # # Took about 7 minutes
    # await summarize_projects_iteration(
    #     input_file_path=INITIAL_PROJECTS_OUTPUT_PATH,
    #     max_size_of_project_groups=7,
    #     iteration=1,
    #     num_gleans=NUM_GLEANS_PER_ITERATION,
    #     randomization_strategy="sort_by_name",
    # )

    # # iteration 2 # 94 projects result
    # # Took about 9 minutes
    # await summarize_projects_iteration(
    #     input_file_path=OUTPUT_FILE_PATH_ROOT / f"{OUTPUT_FILE_NAME_ROOT}_1.pkl",
    #     max_size_of_project_groups=7,
    #     iteration=2,
    #     num_gleans=NUM_GLEANS_PER_ITERATION,
    #     randomization_strategy="sort_by_name",
    # )

    # # iteration 3 # 90 projects result
    # # Took about 24 minutes
    # await summarize_projects_iteration(
    #     input_file_path=OUTPUT_FILE_PATH_ROOT / f"{OUTPUT_FILE_NAME_ROOT}_2.pkl",
    #     max_size_of_project_groups=10,
    #     iteration=3,
    #     num_gleans=NUM_GLEANS_PER_ITERATION,
    # )

    # iteration 4 # 89 projects result
    # Took about 10 minutes
    await summarize_projects_iteration(
        input_file_path=OUTPUT_FILE_PATH_ROOT / f"{OUTPUT_FILE_NAME_ROOT}_3.pkl",
        max_size_of_project_groups=8,
        iteration=4,
        num_gleans=NUM_GLEANS_PER_ITERATION,
    )
