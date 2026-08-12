from datetime import datetime, timezone

from .utils.io import load_pickle_file, dump_list_of_objects_to_csv
from .final_summarization_of_projects import FINAL_PROJECTS_OUTPUT_PATH
from .models import HighLevelResearchAndDevelopmentProject, SourceContentBase
from .utils.io import load_pickle_file, dump_list_of_objects_to_csv
from .source_data.source_data_bigquery import execute_bigquery_query, SourceContentDictAppTask, DATETIME_FORMAT, handle_string_timestamp
from .source_data.queries import APP_TASKS_WITH_TIMESTAMPS_QUERY
from .settings import settings


TASK_CONTENT_DATA_OUTPUT_PATH = settings.output_path / "task_content_data" / "task_content_data.csv"


def load_final_projects_data() -> list[HighLevelResearchAndDevelopmentProject]:
    """
    Load final projects data from pickle file.
    """
    print("Loading final projects data...")

    projects = load_pickle_file(FINAL_PROJECTS_OUTPUT_PATH)
    print(f"Loaded {len(projects)} final projects.")
    print(f"Type of project objects: {type(projects[0])}")

    return projects


def get_source_task_ids_from_projects(projects: list[HighLevelResearchAndDevelopmentProject]) -> list[str]:
    """
    Extract all of the app task ids from the list of final projects.
    """
    print("Extracting source task ids from final projects...")

    # content ids are in the form 'tasks/app_task_id', so we split on 'tasks/' and take the second part
    source_task_ids = list(set(
        content_id.split("tasks/")[1]
        for project in projects
        for content_id in project.source_content_ids
        if "tasks/" in content_id
    ))

    print(f"Got {len(source_task_ids)} unique source task ids from final projects.")

    return source_task_ids


def get_task_content_from_task_ids(source_task_ids: list[str]) -> list[SourceContentBase]:
    """
    Get the relevant app task content from BigQuery for the given list of source task ids.

    Note, we use a slightly different query than for the main source data gathering,
    since we didn't include timestamps in the main source data gathering for app tasks.
    """
    print("Getting relevant app task content data from BigQuery...")

    all_task_content_data = raw_data = execute_bigquery_query(APP_TASKS_WITH_TIMESTAMPS_QUERY)

    # Convert raw data to typed dicts
    source_content_dict: list[SourceContentDictAppTask] = [SourceContentDictAppTask(**r) for r in all_task_content_data]
    print(f"Processed {len(source_content_dict)} app tasks from raw data.")

    # filter to only the relevant task ids
    source_content_dict_filtered = [
        item for item in source_content_dict if item['task_id'] in source_task_ids
    ]
    print(f"Filtered to {len(source_content_dict_filtered)} relevant app tasks based on source task ids.")

    # Convert to SourceContentBase objects
    source_content_base: list[SourceContentBase] = [
        SourceContentBase(
            content_id=f"tasks/{c['task_id']}",
            title=c["task_title"],
            content=c["combined_content"],
            created_at=datetime.strptime(str(c["task_created_at"]), DATETIME_FORMAT).replace(tzinfo=timezone.utc),
            last_comment_at=handle_string_timestamp(str(c["last_comment_at"])),
            closed_at=handle_string_timestamp(str(c["task_closed_at"])),
            url=f"https://app.example.com/tasks/{c['task_id']}",
        )
        for c in source_content_dict_filtered
    ]
    print(f"Converted to {len(source_content_base)} SourceContentBase objects.")

    return source_content_base


def dump_relevant_task_content_to_csv(relevant_task_content: list[SourceContentBase]):
    """
    Dump the relevant app task content to a CSV file for SREDio.
    """
    print(f"Dumping relevant app task content to CSV at {TASK_CONTENT_DATA_OUTPUT_PATH}...")

    dump_list_of_objects_to_csv(relevant_task_content, TASK_CONTENT_DATA_OUTPUT_PATH)


def gather_app_task_data_for_sredio():
    """
    Gather relevant app task data for SREDio.
    This will be provided to SREDio for context, since they don't have access to our app task system
    (but they do have an integration with our GitHub data)

    The steps:
    1. Gather task content ids from the list of final projects
    2. For the task content ids, gather the relevant app task data from BigQuery
        Note, we use a slightly different query than for the main source data gathering,
        since we didn't include timestamps in the main source data gathering for app tasks.
        Since this is a post-processing step, we can augment the main source data gathering query
        in the future to include timestamps for app tasks as well
        (Basically, I, Matt, don't want to re-run everything again at this stage of the 2025 project extraction work).
    """
    print("Gathering app task data for SREDio...\n==============================")

    projects = load_final_projects_data()

    source_task_ids = get_source_task_ids_from_projects(projects)

    relevant_task_content: list[SourceContentBase] = get_task_content_from_task_ids(source_task_ids)

    dump_relevant_task_content_to_csv(relevant_task_content)
