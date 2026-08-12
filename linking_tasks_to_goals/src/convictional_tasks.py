from pydantic import BaseModel, Field

from common.bigquery import query_bq
from common.io import dump_to_pickle_file, load_pickle_file, dump_list_of_objects_to_csv
from .settings import settings
from .models import Task, Workspace, WorkspaceComment, LinkedGoal


TASKS_OUTPUT_PATH = settings.output_path / "tasks_cache.pkl"
TASKS_CSV_OUTPUT_PATH = settings.output_path / "tasks_cache.csv"

# To get a good task representation, get a mix of tasks with and without linked goals
# So, we will have a good comparison for tasks that should and shouldn't have linked goals
TASKS_BIGQUERY_QUERY = """
with

tasks as (
  select *
  from `${GCP_PROJECT}.cloudsql_decide_public.task`
  where 1=1
    and organization_id = '00000000-0000-0000-0000-000000000000' -- set this to your organization's id
    and _fivetran_deleted = false
    and deleted_at is null
    and sharing = 'organization'
),

workspaces_with_linked_goals as (
  select distinct workspace_id
  from `${GCP_PROJECT}.cloudsql_decide_public.workspacegoal`
),

tasks_linked_goals as (
  select
    t.*,
    wwlg.workspace_id is not null as has_linked_goals
  from tasks as t
  left join workspaces_with_linked_goals as wwlg
    on wwlg.workspace_id = t.workspace_id
),

tasks_with_linked_goals as (
  select *
  from tasks_linked_goals
  where 1=1
    and has_linked_goals = true
    and created_at <= '2025-04-29' -- to keep the dataset consistent, i.e. upper bound cutoff so the number of tasks is the same whenever this query is run
),

tasks_without_linked_goals as (
  select *
  from tasks_linked_goals
  where 1=1
    and has_linked_goals = false
    and created_at >= '2025-02-25' -- Just filtering to get a smaller dataset
    and created_at <= '2025-02-26'
),

unioned as (
  select *
  from tasks_with_linked_goals

  union all

  select *
  from tasks_without_linked_goals
)

select *
from unioned
"""


WORKSPACE_COMMENTS_BIGQUERY_QUERY = """
select
  c.*,
  u.name as user_name
from `${GCP_PROJECT}.cloudsql_decide_public.comment` c
left join `${GCP_PROJECT}.cloudsql_decide_public.user` u
  on u.id = c.user_id
where 1=1
  and workspace_id in (
    {workspace_ids}
  )
  and c._fivetran_deleted = false
"""

LINKED_GOALS_BIGQUERY_QUERY = """
select
  wg.*,
  g.title as goal_title,
  g.created_at as goal_created_at,
from `${GCP_PROJECT}.cloudsql_decide_public.workspacegoal` wg
left join `${GCP_PROJECT}.cloudsql_decide_public.goal` g
  on g.id = wg.goal_id
where wg.workspace_id in (
  {workspace_ids}
)
"""


class TaskCSVRow(BaseModel):
    """
    Represents a row in the CSV file for tasks.
    """

    id: str = Field(..., description="The unique identifier for the task")
    organization_id: str = Field(..., description="The ID of the organization to which the task belongs")
    workspace_id: str = Field(..., description="The ID of the workspace attached to the task")
    title: str = Field(..., description="The title of the task")
    description: str = Field(..., description="The description of the task")
    sharing: str = Field(..., description="The sharing setting for the task")
    created_at: str = Field(..., description="The datetime when the task was created, in ISO format")
    workspace_comments: str = Field(
        ...,
        description="The workspace associated with the goal, with comments as a newline separated string, separated in ascending created at for comments",
    )
    linked_goals: str = Field(
        ...,
        description="The linked goals associated with the task, as a newline separated string",
    )
    task_url: str = Field(..., description="The platform URL for the task")


def get_convictional_tasks(load_from_cache: bool) -> list[Task]:
    """
    This function gets tasks from the Convictional platform.

    In production, this would be fetched from Prod Postgres.

    For this experiment, we fetch tasks from BigQuery,
    since connection to Postgres is not available and is a bit of a pain to implement for local Postgres.
    """
    print("Fetching Convictional tasks...")

    if load_from_cache:
        print("Loading Convictional tasks from cache...")
        tasks = load_pickle_file(TASKS_OUTPUT_PATH)
    else:
        print("Fetching Convictional tasks from BigQuery...")

        # Get tasks data from BigQuery
        tasks_data: list[dict] = query_bq(TASKS_BIGQUERY_QUERY, settings.gcp_project).to_dict(orient="records")
        print(f"Fetched {len(tasks_data)} tasks from BigQuery.")

        # Get task workspace comments from BigQuery
        task_workspace_ids = [task["workspace_id"] for task in tasks_data]
        print("Getting workspace comments data from BigQuery...")
        workspace_comments_data: list[dict] = query_bq(
            WORKSPACE_COMMENTS_BIGQUERY_QUERY.format(
                workspace_ids=", ".join(f"'{workspace_id}'" for workspace_id in task_workspace_ids)
            ),
            settings.gcp_project,
        ).to_dict(orient="records")
        print(f"Fetched {len(workspace_comments_data)} workspace comments from BigQuery.")

        # Map workspace IDs to comments
        workspace_id_to_comments_mapping: dict[str, list[dict]] = map_workspace_ids_to_comments(
            task_workspace_ids, workspace_comments_data
        )

        # Get linked goals from BigQuery
        print("Getting linked goals data from BigQuery...")
        linked_goals_data: list[dict] = query_bq(
            LINKED_GOALS_BIGQUERY_QUERY.format(
                workspace_ids=", ".join(f"'{workspace_id}'" for workspace_id in task_workspace_ids)
            ),
            settings.gcp_project,
        ).to_dict(orient="records")
        print(f"Fetched {len(linked_goals_data)} linked goals from BigQuery.")

        # Map workspace IDs to linked goals
        workspace_id_to_linked_goals_mapping: dict[str, list[dict]] = map_workspace_ids_to_linked_goals(
            task_workspace_ids, linked_goals_data
        )

        # Convert the tasks data to Task objects
        tasks: list[Task] = convert_tasks_data_to_tasks_objects(
            tasks_data, workspace_id_to_comments_mapping, workspace_id_to_linked_goals_mapping
        )
        print(f"Converted {len(tasks)} tasks to Task objects.")

        # Order workspace comments by created_at
        for task in tasks:
            task.workspace.comments.sort(key=lambda x: x.created_at)

        # Dump the tasks to a pickle file for caching
        dump_to_pickle_file(tasks, TASKS_OUTPUT_PATH)

    print(f"Finished processing {len(tasks)} tasks, with:")
    print(f"    - {sum(len(task.workspace.comments) for task in tasks)} total workspace comments")
    print(f"    - {sum(len(task.linked_goals) for task in tasks)} total linked goals")

    # Print CSV version of tasks to file
    dump_tasks_to_csv(tasks)

    return tasks


def map_workspace_ids_to_comments(
    workspace_ids: list[str], workspace_comments_data: list[dict]
) -> dict[str, list[dict]]:
    """
    Map workspace IDs to their comments.
    """
    print("Mapping workspace IDs to comments...")

    workspace_id_to_comments_mapping = {workspace_id: [] for workspace_id in workspace_ids}

    for comment in workspace_comments_data:
        workspace_id = comment["workspace_id"]
        if workspace_id in workspace_id_to_comments_mapping:
            workspace_id_to_comments_mapping[workspace_id].append(comment)

    return workspace_id_to_comments_mapping


def map_workspace_ids_to_linked_goals(
    workspace_ids: list[str], linked_goals_data: list[dict]
) -> dict[str, list[dict]]:
    """
    Map workspace IDs to their linked goals.
    """
    print("Mapping workspace IDs to linked goals...")

    workspace_id_to_linked_goals_mapping = {workspace_id: [] for workspace_id in workspace_ids}

    for linked_goal in linked_goals_data:
        workspace_id = linked_goal["workspace_id"]
        if workspace_id in workspace_id_to_linked_goals_mapping:
            workspace_id_to_linked_goals_mapping[workspace_id].append(linked_goal)

    return workspace_id_to_linked_goals_mapping


def convert_tasks_data_to_tasks_objects(
    tasks_data: list[dict],
    workspace_id_to_comments_mapping: dict[str, list[dict]],
    workspace_id_to_linked_goals_mapping: dict[str, list[dict]],
) -> list[Task]:
    """
    Convert the tasks data to Task objects.
    """
    print("Converting tasks data to Task objects...")

    tasks = [
        Task(
            id=task["id"],
            organization_id=task["organization_id"],
            workspace_id=task["workspace_id"],
            title=task["title"],
            description=task["description"],
            sharing=task["sharing"],
            created_at=task["created_at"].to_pydatetime(),
            workspace=Workspace(
                id=task["workspace_id"],
                comments=[
                    WorkspaceComment(
                        id=comment["id"],
                        workspace_id=comment["workspace_id"],
                        user_id=comment["user_id"],
                        user_name="" if comment["user_name"] is None else comment["user_name"],
                        content=comment["content"],
                        created_at=comment["created_at"].to_pydatetime(),
                    )
                    for comment in workspace_id_to_comments_mapping.get(task["workspace_id"], [])
                ],
            ),
            linked_goals=[
                LinkedGoal(
                    goal_id=linked_goal["goal_id"],
                    title=linked_goal["goal_title"],
                    created_at=linked_goal["goal_created_at"].to_pydatetime(),
                )
                for linked_goal in workspace_id_to_linked_goals_mapping.get(task["workspace_id"], [])
            ],
            task_url=f"https://app.example.com/tasks/{task["id"]}",
        )
        for task in tasks_data
    ]

    return tasks


def dump_tasks_to_csv(tasks: list[Task]):
    print("Writing tasks to CSV file...")

    tasks_csv = [
        TaskCSVRow(
            id=task.id,
            organization_id=task.organization_id,
            workspace_id=task.workspace_id,
            title=task.title,
            description=task.description,
            sharing=task.sharing,
            created_at=task.created_at.isoformat(),
            workspace_comments="\n\n".join(
                [
                    f"ID: {comment.id}\nWorkspace ID: {comment.workspace_id}\nUser ID: {comment.user_id}\nUser Name: {comment.user_name}\nContent: {comment.content}\nCreated At: {comment.created_at}"
                    for comment in task.workspace.comments
                ]
            ),
            linked_goals="\n\n".join(
                [
                    f"Goal ID: {goal.goal_id}\nTitle: {goal.title}\nCreated at: {goal.created_at}\nGoal URL: https://app.example.com/goals/{goal.goal_id}"
                    for goal in task.linked_goals
                ]
            ),
            task_url=task.task_url,
        )
        for task in tasks
    ]

    dump_list_of_objects_to_csv(tasks_csv, TASKS_CSV_OUTPUT_PATH)
