import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from datetime import datetime

from ..models import Goal, Task
from ..settings import settings
from common.embeddings import aembed, cosine_similarity
from common.async_helper import limited_task, execute_tasks_with_manual_pbar
from common.io import dump_to_pickle_file, load_pickle_file, dump_list_of_objects_to_csv


GOAL_EMBEDDINGS_OUTPUT_PATH = settings.output_path / "approach_2_goal_embeddings.pkl"
TASK_EMBEDDINGS_OUTPUT_PATH = settings.output_path / "approach_2_task_embeddings.pkl"
TASK_MATCH_RESULTS_CSV_OUTPUT_PATH = settings.output_path / "approach_2_results.csv"


class MatchResult(BaseModel):
    """
    Represents a match result between a task and a goal.
    """

    task_id: str = Field(..., description="The ID of the task")
    goal_id: str = Field(..., description="The ID of the goal")
    similarity_score: float = Field(..., description="The similarity score between the task and the goal")
    goal_title: str = Field(..., description="The title of the goal")
    goal_created_at: datetime = Field(..., description="The datetime when the goal was created")
    goal_url: str = Field(..., description="The URL of the goal")


class TaskMatchResultsCSVRow(BaseModel):
    """
    Represents the task with its goal match results
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
    task_url: str = Field(..., description="The platform URL for the task")
    linked_goals: str = Field(
        ...,
        description="The linked goals associated with the task, as a newline separated string",
    )
    goal_match_results: str = Field(
        ...,
        description="The goal match results associated with the task, as a newline separated string",
    )
    similarity_scores_list: list[float] = Field(
        ...,
        description="The similarity scores list associated with the task, as a list of floats",
    )


async def approach_2(goals: list[Goal], tasks: list[Task], load_embeddings_from_cache: bool):
    """
    Approach 2 for matching platform tasks to goals.

    In this approach we use embeddings to match tasks to goals.
        - Task embeddings: title + description + all workspace comments
        - Goal embeddings: title + all success condition descriptions + all workspace comments
    """
    print("Starting matching platform tasks to goals using Approach 2...")

    if load_embeddings_from_cache:
        print("Loading embeddings from cache...")
        goal_embeddings: list[list[float]] = load_pickle_file(GOAL_EMBEDDINGS_OUTPUT_PATH)
        task_embeddings: list[list[float]] = load_pickle_file(TASK_EMBEDDINGS_OUTPUT_PATH)
    else:
        print("Loading embeddings using OpenAI API...")

        # Embed goals
        print("Embedding goals...")
        goal_texts_to_embed = [
            f"{goal.title} {' '.join([sc.description for sc in goal.success_conditions])} {' '.join([comment.content for comment in goal.workspace.comments])}"
            for goal in goals
        ]
        goal_embeddings: list[list[float]] = await embed_list_of_texts(goal_texts_to_embed)

        # Embed tasks
        print("Embedding tasks...")
        task_texts_to_embed = [
            f"{task.title} {task.description} {' '.join([comment.content for comment in task.workspace.comments])}"
            for task in tasks
        ]
        task_embeddings: list[list[float]] = await embed_list_of_texts(task_texts_to_embed)

        # Save embeddings to cache
        print("Saving embeddings to cache...")
        dump_to_pickle_file(goal_embeddings, GOAL_EMBEDDINGS_OUTPUT_PATH)
        dump_to_pickle_file(task_embeddings, TASK_EMBEDDINGS_OUTPUT_PATH)

    print(f"Goal embeddings: {len(goal_embeddings)}")
    print(f"Task embeddings: {len(task_embeddings)}")

    # Do cosine similarity
    cosine_similarities: list[list[float]] = []
    for task_embedding in task_embeddings:
        similarities = [cosine_similarity(task_embedding, goal_embedding) for goal_embedding in goal_embeddings]
        cosine_similarities.append(similarities)

    # Create MatchResult objects
    match_results: list[list[MatchResult]] = []
    for i, task in enumerate(tasks):
        task_match_results: list[MatchResult] = []
        for j, goal in enumerate(goals):
            task_match_results.append(
                MatchResult(
                    task_id=task.id,
                    goal_id=goal.id,
                    similarity_score=cosine_similarities[i][j],
                    goal_title=goal.title,
                    goal_created_at=goal.created_at,
                    goal_url=goal.goal_url,
                )
            )
        match_results.append(task_match_results)

    # Create CSV rows
    tasks_csv_rows: list[TaskMatchResultsCSVRow] = []
    for i, task in enumerate(tasks):
        tasks_csv_rows.append(
            TaskMatchResultsCSVRow(
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
                task_url=task.task_url,
                linked_goals="\n\n".join(
                    [
                        f"Goal ID: {goal.goal_id}\nTitle: {goal.title}\nCreated at: {goal.created_at}\nGoal URL: https://app.example.com/goals/{goal.goal_id}"
                        for goal in task.linked_goals
                    ]
                ),
                goal_match_results="\n\n".join(
                    [
                        f"Similarity Score: {result.similarity_score}\nGoal Title: {result.goal_title}\nGoal Created At: {result.goal_created_at.isoformat()}\nGoal URL: {result.goal_url}"
                        for result in match_results[i]
                    ]
                ),
                similarity_scores_list=[result.similarity_score for result in match_results[i]],
            )
        )

    # Dump the task match results to a CSV file
    dump_list_of_objects_to_csv(tasks_csv_rows, TASK_MATCH_RESULTS_CSV_OUTPUT_PATH)


async def embed_list_of_texts(
    list_of_texts: list[str],
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> list[list[float]]:
    print("Embedding texts...")

    open_ai_client = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(), organization=settings.openai_organization
    )

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = [
        limited_task(
            aembed(
                async_openai_client=open_ai_client,
                text=text,
                embedding_model=settings.embedding_model,
                embedding_dim=settings.embedding_dimension,
            ),
            semaphore,
            delay_between_tasks,
        )
        for text in list_of_texts
    ]

    embeddings = await execute_tasks_with_manual_pbar(tasks)

    return embeddings
