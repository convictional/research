from pydantic import BaseModel, Field
from typing import Dict
import json
import time

from .settings import settings
from .models import ResearchAndDevelopmentProject, HighLevelResearchAndDevelopmentProject
from .utils.io import load_pickle_file, dump_to_pickle_file, dump_list_of_objects_to_csv
from .fine_grained_into_coarse_grained_projects import OUTPUT_FILE_PATH_ROOT as final_iteration_output_file_path_root
from .fine_grained_into_coarse_grained_projects import OUTPUT_FILE_NAME_ROOT as final_iteration_output_file_name_root
from .prompts.engine import build_prompt
from .utils.tokens import count_tokens
from .utils.instruct_llm import ainstruct_llm
from .utils.async_helper import execute_tasks_with_manual_pbar


FINAL_ITERATION_NUM_IN_COARSE_GRAINED_PROJECTS = 4
MAX_NUM_HIGH_LEVEL_PROJECT_THEMES = 25
LLM_TEMPERATURE = 0.0

OUTPUT_PATH_ROOT = settings.output_path / "final_summarization"
LLM_GROUPED_PROJECTS_DICT_OUTPUT_PATH = OUTPUT_PATH_ROOT / "llm_grouped_projects_dict.pkl"
FINAL_PROJECTS_OUTPUT_PATH = OUTPUT_PATH_ROOT / "final_projects.pkl"
FINAL_PROJECTS_CSV_OUTPUT_PATH = OUTPUT_PATH_ROOT / "final_projects.csv"


class LLMGroupProjectsResponse(BaseModel):
    high_level_groupings: Dict[int, list[int]] = Field(
        ...,
        title="High-level groupings of research and development projects. Each key is a high-level theme numeric ID and the value is a list of project IDs that fall under that high-level theme.",
    )


def load_projects(final_iteration_num: int) -> list[ResearchAndDevelopmentProject]:
    """
    Load projects from a given iteration.
    """
    print(f"Loading projects from coarse-grained iteration {final_iteration_num}...")

    projects = load_pickle_file(
        final_iteration_output_file_path_root / f"{final_iteration_output_file_name_root}_{final_iteration_num}.pkl"
    )
    print(f"Number of projects loaded: {len(projects)}")

    return projects


async def group_projects_into_high_level_themes(
    input_projects: list[ResearchAndDevelopmentProject], max_num_high_level_themes: int
):
    """
    Group projects together into high-level themes using arbitrary indexing.
    """
    print(f"Grouping {len(input_projects)} input projects into {max_num_high_level_themes} high-level themes...")

    system_prompt = build_prompt(
        "final_summarization/project_grouping_system.txt.jinja", max_num_themes=max_num_high_level_themes
    )

    json_input_projects = [
        json.dumps({"project_id": i, "name": p.name, "description": p.description}, indent=2)
        for i, p in enumerate(input_projects)
    ]

    user_prompt = build_prompt(
        "final_summarization/project_grouping_user.txt.jinja",
        research_and_development_projects="\n\n".join(json_input_projects),
    )

    num_tokens_user_prompt = count_tokens(user_prompt)
    print(f"Number of tokens in user prompt: {num_tokens_user_prompt}")

    llm_call_start_ime = time.perf_counter()
    print("Calling LLM...")
    response = await ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=LLM_TEMPERATURE,
        response_model=LLMGroupProjectsResponse,
    )
    llm_call_end_time = time.perf_counter()
    print(f"LLM call took {llm_call_end_time - llm_call_start_ime} seconds.")

    groupings: Dict[int, list[int]] = response.high_level_groupings

    print("High-level groupings:")
    print(groupings)

    # Do some analysis on the groupings to see if there are any issues
    analyze_groupings_response(groupings, input_projects)

    dump_to_pickle_file(groupings, LLM_GROUPED_PROJECTS_DICT_OUTPUT_PATH)


def analyze_groupings_response(groupings: Dict[int, list[int]], input_projects: list[ResearchAndDevelopmentProject]):
    """
    Some analysis on the grouping that the LLM came up with to see if there are any issues.
    """
    print("Analyzing groupings...")

    # High-level stats
    num_high_level_themes = len(groupings)
    num_projects_grouped = sum([len(v) for v in groupings.values()])
    num_unique_project_ids = len(set([id for project_ids in groupings.values() for id in project_ids]))

    print(f"Number of high-level themes: {num_high_level_themes}")
    print(f"Total number of projects grouped: {num_projects_grouped}")
    print(f"Total number of unique project IDs grouped: {num_unique_project_ids}")

    # Check if there are high-level themes with only one project
    themes_with_one_project = []
    for k, v in groupings.items():
        print(f"Number of projects in high-level theme {k}: {len(v)}")
        if len(v) == 1:
            themes_with_one_project.append(k)
    if len(themes_with_one_project) > 0:
        print("WARNING: There are high-level themes with only one project.")
        print(f"High-level themes with only one project: {themes_with_one_project}")
    else:
        print("PASS: All high-level themes have more than one project.")

    # Check if there are more projects grouped than the number of projects, and if so, print out duplicates
    if num_projects_grouped > len(input_projects):
        print("WARNING: Number of projects grouped is greater than the number of projects.")
        seen_projects = set()
        duplicates = []
        for project_ids in groupings.values():
            for id in project_ids:
                if id in seen_projects:
                    duplicates.append(id)
                seen_projects.add(id)

        print(f"Duplicate project IDs: {sorted(duplicates)}")
    else:
        print("PASS: Number of projects grouped is less than or equal to the number of projects.")

    # Check if there are any invalid project IDs
    unique_project_ids = set([id for project_ids in groupings.values() for id in project_ids])
    invalid_ids = [id for id in unique_project_ids if id < 0 or id >= len(input_projects)]
    if len(invalid_ids) > 0:
        print("WARNING: There are invalid project IDs.")
        print(f"Invalid project IDs: {invalid_ids}")
    else:
        print("PASS: All project IDs are valid.")

    # Find project IDs that are used in more than one grouping
    project_ids_used_in_multiple_groupings = []
    seen_project_ids = set()
    for project_ids in groupings.values():
        for id in project_ids:
            if id in seen_project_ids:
                project_ids_used_in_multiple_groupings.append(id)
            seen_project_ids.add(id)
    if len(project_ids_used_in_multiple_groupings) > 0:
        print("There are project IDs used in multiple groupings.")
        print(f"Project IDs used in multiple groupings: {project_ids_used_in_multiple_groupings}")
    else:
        print("PASS: No project IDs are used in multiple groupings.")

    print("Finished analyzing groupings.")


async def get_summarized_projects_from_high_level_themes(
    input_projects: list[ResearchAndDevelopmentProject],
) -> list[HighLevelResearchAndDevelopmentProject]:
    """
    Get summarized projects from high-level themes.
    That is, take the groups of projects and ask LLM to distill a project from each group.
    """
    print("Getting summarized projects from high-level themes...")

    # Load the high-level groupings
    print("Loading high-level groupings...")
    groupings: Dict[int, list[int]] = load_pickle_file(LLM_GROUPED_PROJECTS_DICT_OUTPUT_PATH)
    print(f"Loaded groupings with {len(groupings)} high-level themes.")

    projects_per_group: list[list[ResearchAndDevelopmentProject]] = []
    for _, project_ids in groupings.items():
        projects_for_group = [input_projects[project_id] for project_id in project_ids]
        projects_per_group.append(projects_for_group)

    for i, project_group in enumerate(projects_per_group):
        print(f"Number of projects in group {i + 1}: {len(project_group)}")

    system_prompt = build_prompt("final_summarization/project_summarization_system.txt.jinja")
    user_prompts = [
        build_prompt(
            "final_summarization/project_summarization_user.txt.jinja",
            research_and_development_projects="\n\n".join(
                [json.dumps({"name": p.name, "description": p.description}, indent=2) for p in project_group]
            ),
        )
        for project_group in projects_per_group
    ]

    for i, prompt in enumerate(user_prompts):
        print(f"Number of tokens in user prompt for group {i + 1}: {count_tokens(prompt)}")

    tasks = [
        ainstruct_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=LLM_TEMPERATURE,
            response_model=HighLevelResearchAndDevelopmentProject,
        )
        for user_prompt in user_prompts
    ]

    new_projects: list[HighLevelResearchAndDevelopmentProject] = await execute_tasks_with_manual_pbar(tasks)

    print(f"Number of high-level projects extracted: {len(new_projects)}")

    summarized_projects: list[HighLevelResearchAndDevelopmentProject] = []
    for project, group in zip(new_projects, projects_per_group):
        unique_source_content_ids = sorted(list(set([id for p in group for id in p.source_content_ids])))
        summarized_projects.append(
            HighLevelResearchAndDevelopmentProject(
                name=project.name,
                description=project.description,
                source_content_ids=unique_source_content_ids,
            )
        )

    return summarized_projects


def dump_final_projects_to_files(final_projects: list[HighLevelResearchAndDevelopmentProject]):
    """
    Dump the new projects for a given iteration to files.
    2 files: pickle and csv
    """
    print("Dumping summarized projects to files...")

    dump_to_pickle_file(final_projects, FINAL_PROJECTS_OUTPUT_PATH)
    dump_list_of_objects_to_csv(final_projects, FINAL_PROJECTS_CSV_OUTPUT_PATH)


async def final_summarization_of_projects():
    """
    This function will do the final summarization of projects.
    There are two main steps:
    - Grouping projects together into themes, using arbitrary indexing
    - Taking those groups and asking LLM to distill a project from each group.
    """
    print(
        f"Starting final summarization of projects using max number of high-level project themes = {MAX_NUM_HIGH_LEVEL_PROJECT_THEMES}..."
    )

    # Read in last iteration of coarse-grained projects
    input_projects: list[ResearchAndDevelopmentProject] = load_projects(
        final_iteration_num=FINAL_ITERATION_NUM_IN_COARSE_GRAINED_PROJECTS
    )

    # Group projects together into high-level themes
    await group_projects_into_high_level_themes(input_projects, MAX_NUM_HIGH_LEVEL_PROJECT_THEMES)

    # Get summarized projects from high-level themes
    final_projects: list[
        HighLevelResearchAndDevelopmentProject
    ] = await get_summarized_projects_from_high_level_themes(input_projects)

    # Dump the new projects to files
    dump_final_projects_to_files(final_projects)
