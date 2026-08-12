import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from datetime import datetime

from ..models import Goal, Task
from ..settings import settings
from common.instruct_llm import set_async_instructor_client, ainstruct_llm
from common.prompt_template_engine import build_prompt
from common.async_helper import limited_task, execute_tasks_with_manual_pbar
from common.embeddings import aembed, cosine_similarity
from common.io import dump_to_pickle_file, load_pickle_file, dump_list_of_objects_to_csv


LLM_TEMPERATURE = 0.0

GOAL_EMBEDDINGS_OUTPUT_PATH = settings.output_path / "approach_3_goal_embeddings.pkl"
TASK_EMBEDDINGS_OUTPUT_PATH = settings.output_path / "approach_3_task_embeddings.pkl"
GOAL_SUMMARIES_OUTPUT_PATH = settings.output_path / "approach_3_goal_summaries.pkl"
TASK_SUMMARIES_OUTPUT_PATH = settings.output_path / "approach_3_task_summaries.pkl"
TASK_MATCH_RESULTS_CSV_OUTPUT_PATH = settings.output_path / "approach_3_results.csv"


class GoalSummaryResponse(BaseModel):
    summary: str = Field(..., title="The summary of the goal")


class TaskSummaryResponse(BaseModel):
    summary: str = Field(..., title="The summary of the task")


class MatchResult(BaseModel):
    """
    Represents a match result between a task and a goal.
    """

    task_id: str = Field(..., description="The ID of the task")
    goal_id: str = Field(..., description="The ID of the goal")
    similarity_score: float = Field(..., description="The similarity score between the task and the goal")
    goal_title: str = Field(..., description="The title of the goal")
    goal_llm_summary: str = Field(..., description="The LLM generated summary of the goal, used for the embedding")
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
    task_llm_summary: str = Field(..., description="The LLM generated summary of the task, used for the embedding")
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


async def approach_3(goals: list[Goal], tasks: list[Task], load_embeddings_from_cache: bool):
    """
    Approach 3 for matching platform tasks to goals.

    This approach uses embeddings of LLM generated sumaries of the tasks and goals to match them.
    """
    print("Starting matching platform tasks to goals using Approach 3...")

    if load_embeddings_from_cache:
        print("Loading embeddings from cache...")
        goal_embeddings: list[list[float]] = load_pickle_file(GOAL_EMBEDDINGS_OUTPUT_PATH)
        task_embeddings: list[list[float]] = load_pickle_file(TASK_EMBEDDINGS_OUTPUT_PATH)
        goal_summaries: list[GoalSummaryResponse] = load_pickle_file(GOAL_SUMMARIES_OUTPUT_PATH)
        task_summaries: list[TaskSummaryResponse] = load_pickle_file(TASK_SUMMARIES_OUTPUT_PATH)
    else:
        print("Loading embeddings using OpenAI API...")

        # Get goal summaries from LLM
        print("Summarizing goals...")

        system_prompt = build_prompt("approach_3_goal_summary_system.txt.jinja")
        user_prompts = [build_prompt("approach_3_goal_summary_user.txt.jinja", goal=goal) for goal in goals]
        goal_summaries: list[GoalSummaryResponse] = await get_summaries_from_llm(
            system_prompt, user_prompts, GoalSummaryResponse
        )

        # Get task summaries from LLM
        print("Summarizing tasks...")
        system_prompt = build_prompt("approach_3_task_summary_system.txt.jinja")
        user_prompts = [build_prompt("approach_3_task_summary_user.txt.jinja", task=task) for task in tasks]
        task_summaries: list[TaskSummaryResponse] = await get_summaries_from_llm(
            system_prompt, user_prompts, TaskSummaryResponse
        )

        # Save summaries to cache
        print("Saving summaries to cache...")
        # Because of some pickling error, we need to recreate the same objects
        # Can't be bothered to dig into it right now, so just doign this workaround
        goal_summaries = [GoalSummaryResponse(summary=goal_summary.summary) for goal_summary in goal_summaries]
        task_summaries = [TaskSummaryResponse(summary=task_summary.summary) for task_summary in task_summaries]
        dump_to_pickle_file(goal_summaries, GOAL_SUMMARIES_OUTPUT_PATH)
        dump_to_pickle_file(task_summaries, TASK_SUMMARIES_OUTPUT_PATH)

        # Get embeddings for goal summaries
        print("Embedding goal summaries...")
        goal_texts_to_embed = [goal_summary.summary for goal_summary in goal_summaries]
        goal_embeddings: list[list[float]] = await embed_list_of_texts(goal_texts_to_embed)

        # Get embeddings for task summaries
        print("Embedding task summaries...")
        task_texts_to_embed = [task_summary.summary for task_summary in task_summaries]
        task_embeddings: list[list[float]] = await embed_list_of_texts(task_texts_to_embed)

        # Save embeddings to cache
        print("Saving embeddings to cache...")
        dump_to_pickle_file(goal_embeddings, GOAL_EMBEDDINGS_OUTPUT_PATH)
        dump_to_pickle_file(task_embeddings, TASK_EMBEDDINGS_OUTPUT_PATH)

    print(f"Goal embeddings: {len(goal_embeddings)}")
    print(f"Task embeddings: {len(task_embeddings)}")
    print(f"Goal summaries: {len(goal_summaries)}")
    print(f"Task summaries: {len(task_summaries)}")

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
                    goal_llm_summary=goal_summaries[j].summary,
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
                task_llm_summary=task_summaries[i].summary,
                linked_goals="\n\n".join(
                    [
                        f"Goal ID: {goal.goal_id}\nTitle: {goal.title}\nCreated at: {goal.created_at}\nGoal URL: https://app.example.com/goals/{goal.goal_id}"
                        for goal in task.linked_goals
                    ]
                ),
                goal_match_results="\n\n".join(
                    [
                        f"Similarity Score: {result.similarity_score}\nGoal Title: {result.goal_title}\nGoal Created At: {result.goal_created_at.isoformat()}\nGoal URL: {result.goal_url}\nGoal LLM Summary: {result.goal_llm_summary}"
                        for result in match_results[i]
                    ]
                ),
                similarity_scores_list=[result.similarity_score for result in match_results[i]],
            )
        )

    # Dump the task match results to a CSV file
    dump_list_of_objects_to_csv(tasks_csv_rows, TASK_MATCH_RESULTS_CSV_OUTPUT_PATH)


async def get_summaries_from_llm(
    system_prompt: str,
    user_prompts: list[str],
    response_model: type[BaseModel],
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> list[type[BaseModel]]:
    """
    Get summaries for the given user prompts using the specified system prompt.
    """
    print("Getting summaries...")

    set_async_instructor_client(llm_model=settings.llm_model, api_key=settings.anthropic_api_key)

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        limited_task(
            ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                llm_model=settings.llm_model,
                temperature=LLM_TEMPERATURE,
            ),
            semaphore,
            delay_between_tasks,
        )
        for user_prompt in user_prompts
    ]

    summaries: list[type[BaseModel]] = await execute_tasks_with_manual_pbar(tasks)

    return summaries


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
