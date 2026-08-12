from pydantic import BaseModel, Field

from common.bigquery import query_bq
from common.io import dump_to_pickle_file, load_pickle_file, dump_list_of_objects_to_csv
from .settings import settings
from .models import Goal, SuccessCondition, Workspace, WorkspaceComment


GOALS_OUTPUT_PATH = settings.output_path / "goals_cache.pkl"
GOALS_CSV_OUTPUT_PATH = settings.output_path / "goals_cache.csv"


GOALS_BIGQUERY_QUERY = """
select *
from `${GCP_PROJECT}.cloudsql_decide_public.goal`
where 1=1
  and organization_id = '00000000-0000-0000-0000-000000000000' -- set this to your organization's id
  and _fivetran_deleted = false
  and deleted_at is null
  and sharing = 'organization'
  and created_at <= '2025-04-29' -- to keep the dataset consistent, i.e. upper bound cutoff so the number of tasks is the same whenever this query is run
"""

SUCCESS_CONDITIONS_BIGQUERY_QUERY = """
select *
from `${GCP_PROJECT}.cloudsql_decide_public.successcondition`
where 1=1
  and goal_id in (
    {goal_ids}
  )
  and _fivetran_deleted = false
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


class GoalCSVRow(BaseModel):
    """
    Represents a row in the CSV file for goals.
    """

    id: str = Field(..., description="The unique identifier for the goal")
    organization_id: str = Field(..., description="The ID of the organization to which the goal belongs")
    workspace_id: str = Field(..., description="The ID of the workspace attached to the goal")
    title: str = Field(..., description="The title of the goal")
    status: str = Field(..., description="The current status of the goal")
    sharing: str = Field(..., description="The sharing setting for the goal")
    created_at: str = Field(..., description="The datetime when the goal was created, in ISO format")
    success_conditions: str = Field(
        ..., description="The success conditions for the goal, as a newline separated string"
    )
    workspace_comments: str = Field(
        ...,
        description="The workspace associated with the goal, with comments as a newline separated string, separated in ascending created at for comments",
    )
    goal_url: str = Field(..., description="The platform URL for the goal")


def get_convictional_goals(load_from_cache: bool) -> list[Goal]:
    """
    This function gets goals from the Convictional platform.

    In production, this would be fetched from Prod Postgres.

    For this experiment, we fetch goals from BigQuery,
    since connection to Postgres is not available and is a bit of a pain to implement for local Postgres.
    """
    print("Fetching Convictional goals...")

    if load_from_cache:
        print("Loading Convictional goals from cache...")
        goals = load_pickle_file(GOALS_OUTPUT_PATH)
    else:
        print("Fetching Convictional goals from BigQuery...")

        # Get goals data from BigQuery
        print("Getting goals data from BigQuery...")
        goals_data: list[dict] = query_bq(GOALS_BIGQUERY_QUERY, settings.gcp_project).to_dict(orient="records")
        print(f"Fetched {len(goals_data)} goals from BigQuery.")

        # Get success conditions data from BigQuery, using the goal IDs from the goals data
        goal_ids: list[str] = [goal["id"] for goal in goals_data]
        print("Getting success conditions data from BigQuery...")
        success_conditions_data: list[dict] = query_bq(
            SUCCESS_CONDITIONS_BIGQUERY_QUERY.format(goal_ids=", ".join(f"'{goal_id}'" for goal_id in goal_ids)),
            settings.gcp_project,
        ).to_dict(orient="records")
        print(f"Fetched {len(success_conditions_data)} success conditions from BigQuery.")

        # Map goal IDs to success conditions
        goal_id_to_success_conditions_mapping: dict[str, list[dict]] = map_goal_ids_to_success_conditions(
            goal_ids, success_conditions_data
        )

        # Get goal workspace comments from BigQuery
        goal_workspace_ids = [goal["workspace_id"] for goal in goals_data]
        print("Getting workspace comments data from BigQuery...")
        workspace_comments_data: list[dict] = query_bq(
            WORKSPACE_COMMENTS_BIGQUERY_QUERY.format(
                workspace_ids=", ".join(f"'{workspace_id}'" for workspace_id in goal_workspace_ids)
            ),
            settings.gcp_project,
        ).to_dict(orient="records")
        print(f"Fetched {len(workspace_comments_data)} workspace comments from BigQuery.")

        # Map workspace IDs to comments
        workspace_id_to_comments_mapping: dict[str, list[dict]] = map_workspace_ids_to_comments(
            goal_workspace_ids, workspace_comments_data
        )

        # Convert the goals data to Goal objects
        goals: list[Goal] = convert_goals_data_to_goals_objects(
            goals_data, goal_id_to_success_conditions_mapping, workspace_id_to_comments_mapping
        )
        print(f"Converted {len(goals)} goals to Goal objects.")

        # Order workspace comments by created_at
        for goal in goals:
            goal.workspace.comments.sort(key=lambda x: x.created_at)

        # Dump the goals to a pickle file for caching
        dump_to_pickle_file(goals, GOALS_OUTPUT_PATH)

    print(f"Finished processing {len(goals)} goals, with:")
    print(f"    - {sum(len(goal.success_conditions) for goal in goals)} total success conditions")
    print(f"    - {sum(len(goal.workspace.comments) for goal in goals)} total workspace comments")

    # Print CSV version of goals to file
    dump_goals_to_csv(goals)

    return goals


def map_goal_ids_to_success_conditions(
    goal_ids: list[str], success_conditions_data: list[dict]
) -> dict[str, list[dict]]:
    """
    Map goal IDs to their success conditions.
    """
    print("Mapping goal IDs to success conditions...")

    goal_id_to_success_conditions_mapping = {goal_id: [] for goal_id in goal_ids}

    for success_condition in success_conditions_data:
        goal_id = success_condition["goal_id"]
        if goal_id in goal_id_to_success_conditions_mapping:
            goal_id_to_success_conditions_mapping[goal_id].append(success_condition)

    return goal_id_to_success_conditions_mapping


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


def convert_goals_data_to_goals_objects(
    goals_data: list[dict],
    goal_id_to_success_conditions_mapping: dict[str, list[dict]],
    workspace_id_to_comments_mapping: dict[str, list[dict]],
) -> list[Goal]:
    """
    Convert the goals data from BigQuery to Goal objects.
    """
    print("Converting goals data to Goal objects...")

    goals = [
        Goal(
            id=goal["id"],
            organization_id=goal["organization_id"],
            workspace_id=goal["workspace_id"],
            title=goal["title"],
            status=goal["status"],
            sharing=goal["sharing"],
            created_at=goal["created_at"].to_pydatetime(),
            success_conditions=[
                SuccessCondition(
                    id=success_condition["id"],
                    goal_id=success_condition["goal_id"],
                    description=success_condition["description"],
                    status=success_condition["status"],
                    tracking_url=success_condition["tracking_url"],
                    created_at=success_condition["created_at"].to_pydatetime(),
                )
                for success_condition in goal_id_to_success_conditions_mapping.get(goal["id"], [])
            ],
            workspace=Workspace(
                id=goal["workspace_id"],
                comments=[
                    WorkspaceComment(
                        id=comment["id"],
                        workspace_id=comment["workspace_id"],
                        user_id=comment["user_id"],
                        user_name=comment["user_name"],
                        content=comment["content"],
                        created_at=comment["created_at"].to_pydatetime(),
                    )
                    for comment in workspace_id_to_comments_mapping.get(goal["workspace_id"], [])
                ],
            ),
            goal_url=f"https://app.example.com/goals/{goal["id"]}",
        )
        for goal in goals_data
    ]

    return goals


def dump_goals_to_csv(goals: list[Goal]):
    print("Writing goals to CSV...")

    goals_csv: list[GoalCSVRow] = [
        GoalCSVRow(
            id=goal.id,
            organization_id=goal.organization_id,
            workspace_id=goal.workspace_id,
            title=goal.title,
            status=goal.status,
            sharing=goal.sharing,
            created_at=goal.created_at.isoformat(),
            success_conditions="\n\n".join(
                [
                    f"ID: {success_condition.id}\nGoal ID: {success_condition.goal_id}\nDescription: {success_condition.description}\nStatus: {success_condition.status}\nTracking URL: {success_condition.tracking_url}\nCreated At: {success_condition.created_at}"
                    for success_condition in goal.success_conditions
                ]
            ),
            workspace_comments="\n\n".join(
                [
                    f"ID: {comment.id}\nWorkspace ID: {comment.workspace_id}\nUser ID: {comment.user_id}\nUser Name: {comment.user_name}\nContent: {comment.content}\nCreated At: {comment.created_at}"
                    for comment in goal.workspace.comments
                ]
            ),
            goal_url=goal.goal_url,
        )
        for goal in goals
    ]

    dump_list_of_objects_to_csv(goals_csv, GOALS_CSV_OUTPUT_PATH)
