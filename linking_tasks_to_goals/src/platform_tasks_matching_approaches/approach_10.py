import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal

from ..models import Goal, Task
from ..settings import settings
from common.instruct_llm import set_async_instructor_client, ainstruct_llm
from common.prompt_template_engine import build_prompt
from common.async_helper import limited_task, execute_tasks_with_manual_pbar
from common.embeddings import aembed, cosine_similarity
from common.io import dump_to_pickle_file, load_pickle_file, dump_list_of_objects_to_csv


LLM_TEMPERATURE = 0.0
# These are rather arbitrary thresholds, but are based on experimentation heuristics.
# They are meant to be set to filter out low quality matches,
# but still be flexible to account for noise in the similarity scores.
# So, we should expect some false positives - these will be handled later using LLMaaJ.
KEYWORDS_SIMILARITY_LOWER_THRESHOLD = 0.3
ENTITIES_SIMILARITY_LOWER_THRESHOLD = 0.3

GOAL_KEYWORDS_OUTPUT_PATH = settings.output_path / "approach_10_goal_keywords.pkl"
TASK_KEYWORDS_OUTPUT_PATH = settings.output_path / "approach_10_task_keywords.pkl"
GOAL_ENTITIES_OUTPUT_PATH = settings.output_path / "approach_10_goal_entities.pkl"
TASK_ENTITIES_OUTPUT_PATH = settings.output_path / "approach_10_task_entities.pkl"
GOAL_KEYWORDS_EMBEDDINGS_OUTPUT_PATH = settings.output_path / "approach_10_goal_keywords_embeddings.pkl"
TASK_KEYWORDS_EMBEDDINGS_OUTPUT_PATH = settings.output_path / "approach_10_task_keywords_embeddings.pkl"
GOAL_ENTITIES_EMBEDDINGS_OUTPUT_PATH = settings.output_path / "approach_10_goal_entities_embeddings.pkl"
TASK_ENTITIES_EMBEDDINGS_OUTPUT_PATH = settings.output_path / "approach_10_task_entities_embeddings.pkl"
TASK_MATCH_RESULTS_CSV_OUTPUT_PATH = settings.output_path / "approach_10_results.csv"


class GoalKeywordsResponse(BaseModel):
    keywords: str = Field(..., title="The keywords of the goal")


class TaskKeywordsResponse(BaseModel):
    keywords: str = Field(..., title="The keywords of the task")


class GoalEntitiesResponse(BaseModel):
    entities: str = Field(..., title="The entities of the goal")


class TaskEntitiesResponse(BaseModel):
    entities: str = Field(..., title="The entities of the task")


class AugmentedGoal(Goal):
    """
    Represents a goal with additional attributes for matching.
    """

    keywords: Optional[str] = Field(None, title="The keywords of the goal")
    entities: Optional[str] = Field(None, title="The entities of the goal")
    keywords_embeddings: Optional[list[float]] = Field(None, title="The embeddings of the keywords of the goal")
    entities_embeddings: Optional[list[float]] = Field(None, title="The embeddings of the entities of the goal")
    keywords_cosine_similarity_score: Optional[float] = Field(
        None, title="The cosine similarity score of the keywords of the goal and task"
    )
    entities_cosine_similarity_score: Optional[float] = Field(
        None, title="The cosine similarity score of the entities of the goal and task"
    )


class LLMaaJAssumption(BaseModel):
    """
    Represents an assumption made by the LLM in the analysis of the match between a task and a goal.
    """

    assumption: str = Field(
        ...,
        title="An assumption identified by the LLM in the analysis of the match between a task and a goal",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., title="The confidence of the assumption. Can only be 'high', 'medium' or 'low'"
    )


class LLMaaJAnalysisResponse(BaseModel):
    """
    Represents the response from the LLM for the analysis of the match between a task and a goal.
    """

    analysis: str = Field(
        ...,
        title="The analysis of the match between a goal and a task, limited to 10 sentences, based on the evaluation principles",
    )
    assumptions: list[LLMaaJAssumption] = Field(
        ...,
        title="A list of assumptions made by the LLM to reach the decision, and the assumption confidence levels.",
    )


class LLMaaJAnalysisResult(BaseModel):
    """
    Represents the result of the LLM analysis of the match between a task and a goal.
    """

    goal: AugmentedGoal = Field(..., title="The goal being evaluated against the task")
    llmaaj_analysis: LLMaaJAnalysisResponse = Field(
        ..., title="The analysis of the match between the task and the goal"
    )


class LLMaaJResponse(BaseModel):
    """
    Represents the response from the LLM judging the match between a task and a goal.
    """

    # decision can only be "yes" or "no"
    decision: Literal["yes", "no"] = Field(
        ..., title="The decision of judging the match between a goal and a task. Can only be 'yes' or 'no'"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., title="The confidence of the decision. Can only be 'high', 'medium' or 'low'"
    )
    reasoning: str = Field(
        ..., title="A brief reasoning of the decision, limited to 3-4 sentences, based on the evaluation principles"
    )
    assumptions: str = Field(
        ...,
        title="A brief list of assumptions made by the LLM to reach the decision, and the assumption confidence levels. Limited to 5 sentences.",
    )


class GoalMatchResult(BaseModel):
    """
    Represents match details between a task and a goal.
    """

    goal: AugmentedGoal = Field(..., title="The goal being evaluated against the task")
    llmaaj_result: LLMaaJResponse = Field(
        ..., title="The result of the LLM judging the match between the task and the goal"
    )


class AugmentedTask(Task):
    """
    Represents a task with additional attributes for matching.
    """

    keywords: Optional[str] = Field(..., title="The keywords of the task")
    entities: Optional[str] = Field(..., title="The entities of the task")
    keywords_embeddings: Optional[list[float]] = Field(..., title="The embeddings of the keywords of the task")
    entities_embeddings: Optional[list[float]] = Field(..., title="The embeddings of the entities of the task")
    embedding_filtered_goals: Optional[list[AugmentedGoal]] = Field(
        None, title="The filtered goals based on embeddings cosine similarity evaluation"
    )
    llmaaj_analysis: Optional[list[LLMaaJAnalysisResult]] = Field(
        None, title="The analysis of the match between the task and the filtered goals"
    )
    llmaaj_match_results: Optional[list[GoalMatchResult]] = Field(
        None, title="The match results of the LLM judging the match between the task and the filtered goals"
    )
    suggested_goals: Optional[list[AugmentedGoal]] = Field(
        None, title="The suggested goals for the task based on the LLMaaJ evaluations"
    )


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
        ..., description="The linked goals associated with the task, as a newline separated string"
    )
    task_keywords: str = Field(..., description="The LLM extracted keywords associated with the task")
    task_entities: str = Field(..., description="The LLM extracted entities associated with the task")
    num_embedding_filtered_goals: int = Field(
        ..., description="The number of goals that passed the cosine similarity filter for the task"
    )
    embedding_filtered_goals: str = Field(
        ...,
        description="The filtered goals based on embeddings cosine similarity evaluation, as a newline separated string",
    )
    llmaaj_analysis: str = Field(
        ...,
        description="The analysis of the match between the task and the filtered goals, as a newline separated string",
    )
    llmaaj_match_results: str = Field(
        ...,
        description="The match results of the LLM judging the match between the task and the filtered goals, as a newline separated string",
    )
    suggested_goals: str = Field(
        ...,
        description="The suggested goals for the task based on the LLMaaJ evaluations, as a newline separated string",
    )


async def approach_10(goals: list[Goal], tasks: list[Task], load_embeddings_from_cache: bool):
    """
    Approach 10 for matching platform tasks to goals.

    This approach uses a multi-step process to match tasks to goals.
        - 1. Extract keywords using an LLM for tasks and goals.
        - 2. Extract entities using an LLM for tasks and goals.
        - 3. Get embeddings for keywords and entities using OpenAI's embedding model.
        - 4. Calculate cosine similarity between the embeddings of tasks and goals.
        - 5. Filter goals for each task based on a threshold for the cosine similarity scores.
        - 6. Give the remaining goals for each task to an LLM to judge the final match.
             This is done in 2 steps:
                - First, do a LLMaaJ analysis, and list all assumptions in the analysis
                - Then, take the LLMaaJ analysis and assumptions, and make a judgement using only
                  High confidence assumptions.

    This approach is designed to be more robust and accurate than previous approaches,
    and is heavily based on Approach 9.
    """
    print("Starting matching platform tasks to goals using Approach 10...")

    if load_embeddings_from_cache:
        print("Loading embeddings from cache...")
        goal_keywords: list[GoalKeywordsResponse] = load_pickle_file(GOAL_KEYWORDS_OUTPUT_PATH)
        task_keywords: list[TaskKeywordsResponse] = load_pickle_file(TASK_KEYWORDS_OUTPUT_PATH)
        goal_entities: list[GoalEntitiesResponse] = load_pickle_file(GOAL_ENTITIES_OUTPUT_PATH)
        task_entities: list[TaskEntitiesResponse] = load_pickle_file(TASK_ENTITIES_OUTPUT_PATH)
        goal_keywords_embeddings: list[list[float]] = load_pickle_file(GOAL_KEYWORDS_EMBEDDINGS_OUTPUT_PATH)
        task_keywords_embeddings: list[list[float]] = load_pickle_file(TASK_KEYWORDS_EMBEDDINGS_OUTPUT_PATH)
        goal_entities_embeddings: list[list[float]] = load_pickle_file(GOAL_ENTITIES_EMBEDDINGS_OUTPUT_PATH)
        task_entities_embeddings: list[list[float]] = load_pickle_file(TASK_ENTITIES_EMBEDDINGS_OUTPUT_PATH)
    else:
        print("Loading embeddings using OpenAI API...")

        # Get goal and task keywords
        goal_keywords: list[GoalKeywordsResponse]
        task_keywords: list[TaskKeywordsResponse]
        goal_keywords, task_keywords = await get_keywords(goals, tasks)

        # Get goal and task entities
        goal_entities: list[GoalEntitiesResponse]
        task_entities: list[TaskEntitiesResponse]
        goal_entities, task_entities = await get_entities(goals, tasks)

        # Get embeddings for keywords and entities
        goal_keywords_embeddings: list[list[float]]
        task_keywords_embeddings: list[list[float]]
        goal_entities_embeddings: list[list[float]]
        task_entities_embeddings: list[list[float]]
        (
            goal_keywords_embeddings,
            task_keywords_embeddings,
            goal_entities_embeddings,
            task_entities_embeddings,
        ) = await get_embeddings(
            goal_keywords,
            task_keywords,
            goal_entities,
            task_entities,
        )

    print(f"Goal keywords: {len(goal_keywords)}")
    print(f"Task keywords: {len(task_keywords)}")
    print(f"Goal entities: {len(goal_entities)}")
    print(f"Task entities: {len(task_entities)}")
    print(f"Goal keywords embeddings: {len(goal_keywords_embeddings)}")
    print(f"Task keywords embeddings: {len(task_keywords_embeddings)}")
    print(f"Goal entities embeddings: {len(goal_entities_embeddings)}")
    print(f"Task entities embeddings: {len(task_entities_embeddings)}")

    # Transform goals and tasks to AugmentedGoal and AugmentedTask
    augmented_goals: list[AugmentedGoal] = [
        AugmentedGoal(
            **goal.model_dump(),  # Unpack the original goal object
            keywords=goal_keywords[i].keywords,
            entities=goal_entities[i].entities,
            keywords_embeddings=goal_keywords_embeddings[i],
            entities_embeddings=goal_entities_embeddings[i],
        )
        for i, goal in enumerate(goals)
    ]

    augmented_tasks: list[AugmentedTask] = [
        AugmentedTask(
            **task.model_dump(),  # Unpack the original task object
            keywords=task_keywords[i].keywords,
            entities=task_entities[i].entities,
            keywords_embeddings=task_keywords_embeddings[i],
            entities_embeddings=task_entities_embeddings[i],
        )
        for i, task in enumerate(tasks)
    ]

    # Calculate cosine similarity between the embeddings of tasks and goals
    keywords_cosine_similarities: list[list[float]]
    entities_cosine_similarities: list[list[float]]
    keywords_cosine_similarities, entities_cosine_similarities = compute_cosine_similarities(
        augmented_tasks, augmented_goals
    )

    # Match tasks to goals based on cosine similarity
    augmented_tasks = match_tasks_to_goals_using_cosine_similarity(
        augmented_tasks,
        augmented_goals,
        keywords_cosine_similarities,
        entities_cosine_similarities,
    )

    # Do LLM as a judge evaluations for the goals that passed the cosine similarity filter
    augmented_tasks = await do_llm_as_a_judge_evaluations(augmented_tasks)

    # Get suggested goals for each task
    augmented_tasks = get_suggested_goals(augmented_tasks)

    # Create CSV results rows
    tasks_csv_rows: list[TaskMatchResultsCSVRow] = []
    for i, task in enumerate(augmented_tasks):
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
                task_keywords=task.keywords,
                task_entities=task.entities,
                num_embedding_filtered_goals=len(task.embedding_filtered_goals),
                embedding_filtered_goals="\n\n".join(
                    [
                        f"Goal ID: {goal.id}\nTitle: {goal.title}\nCreated at: {goal.created_at.isoformat()}\nGoal URL: https://app.example.com/goals/{goal.id}\nKeywords Cosine Similarity Score: {goal.keywords_cosine_similarity_score}\nEntities Cosine Similarity Score: {goal.entities_cosine_similarity_score}\nKeywords: {goal.keywords}\nEntities: {goal.entities}"
                        for goal in task.embedding_filtered_goals
                    ]
                ),
                llmaaj_analysis="\n\n".join(
                    [
                        f"Goal Title: {result.goal.title}\nAnalysis: {result.llmaaj_analysis.analysis}\nAssumptions:\n {"\n".join([f"Assumption: {assumption.assumption}\nConfidence: {assumption.confidence}" for assumption in result.llmaaj_analysis.assumptions])}"
                        for result in task.llmaaj_analysis
                    ]
                ),
                llmaaj_match_results="\n\n".join(
                    [
                        f"Goal title: {result.goal.title}\nDecision: {result.llmaaj_result.decision}\nConfidence: {result.llmaaj_result.confidence}\nReasoning: {result.llmaaj_result.reasoning}\nAssumptions: {result.llmaaj_result.assumptions}"
                        for result in task.llmaaj_match_results
                    ]
                ),
                suggested_goals="\n\n".join(
                    [
                        f"Goal ID: {goal.id}\nTitle: {goal.title}\nCreated at: {goal.created_at.isoformat()}\nGoal URL: https://app.example.com/goals/{goal.id}"
                        for goal in task.suggested_goals
                    ]
                ),
            )
        )

    # Dump the task match results to a CSV file
    dump_list_of_objects_to_csv(tasks_csv_rows, TASK_MATCH_RESULTS_CSV_OUTPUT_PATH)


def match_tasks_to_goals_using_cosine_similarity(
    augmented_tasks: list[AugmentedTask],
    augmented_goals: list[AugmentedGoal],
    keywords_cosine_similarities: list[list[float]],
    entities_cosine_similarities: list[list[float]],
) -> list[AugmentedTask]:
    """
    Match tasks to goals based on cosine similarity scores.
    """
    print("Matching tasks to goals using cosine similarity...")

    for i, task in enumerate(augmented_tasks):
        # Get the cosine similarity scores for the task
        keywords_similarities: list[float] = keywords_cosine_similarities[i]
        entities_similarities: list[float] = entities_cosine_similarities[i]

        # Filter goals based on the cosine similarity scores and add cosine similarity score to the goal
        # For a match both keywords and entities cosine similarity scores need to be above the threshold
        filtered_goals: list[AugmentedGoal] = []
        for j, goal in enumerate(augmented_goals):
            keywords_similarity = keywords_similarities[j]
            entities_similarity = entities_similarities[j]
            # Check if the cosine similarity scores are above the threshold
            if (
                keywords_similarity >= KEYWORDS_SIMILARITY_LOWER_THRESHOLD
                and entities_similarity >= ENTITIES_SIMILARITY_LOWER_THRESHOLD
            ):
                goal.keywords_cosine_similarity_score = keywords_similarity
                goal.entities_cosine_similarity_score = entities_similarity
                # Add the goal to the filtered goals list
                filtered_goals.append(goal)
        task.embedding_filtered_goals = filtered_goals

    return augmented_tasks


def compute_cosine_similarities(
    augmented_tasks: list[AugmentedTask], augmented_goals: list[AugmentedGoal]
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Compute cosine similarities between tasks and goals for the extracted texts embeddings.
    """

    # Calculate cosine similarity between the keywords embeddings of tasks and goals
    print("Calculating cosine similarity between tasks and goals keywords embeddings...")
    keywords_cosine_similarities: list[list[float]] = []
    for task_embedding in [task.keywords_embeddings for task in augmented_tasks]:
        similarities = [
            cosine_similarity(task_embedding, goal_embedding)
            for goal_embedding in [goal.keywords_embeddings for goal in augmented_goals]
        ]
        keywords_cosine_similarities.append(similarities)

    # Calculate cosine similarity between the entities embeddings of tasks and goals
    print("Calculating cosine similarity between tasks and goals entities embeddings...")
    entities_cosine_similarities: list[list[float]] = []
    for task_embedding in [task.entities_embeddings for task in augmented_tasks]:
        similarities = [
            cosine_similarity(task_embedding, goal_embedding)
            for goal_embedding in [goal.entities_embeddings for goal in augmented_goals]
        ]
        entities_cosine_similarities.append(similarities)

    return keywords_cosine_similarities, entities_cosine_similarities


async def get_keywords(
    goals: list[Goal], tasks: list[Task]
) -> tuple[list[GoalKeywordsResponse], list[TaskKeywordsResponse]]:
    """
    Get keywords for tasks and goals using an LLM.
    """
    print("Getting keywords for tasks and goals...")

    # Goal keywords
    print("Getting keywords for goals...")
    system_prompt = build_prompt("approach_10_goal_keywords_system.txt.jinja")
    user_prompts = [build_prompt("approach_10_goal_keywords_user.txt.jinja", goal=goal) for goal in goals]
    goal_keywords: list[GoalKeywordsResponse] = await query_llm(system_prompt, user_prompts, GoalKeywordsResponse)

    # Task keywords
    print("Getting keywords for tasks...")
    system_prompt = build_prompt("approach_10_task_keywords_system.txt.jinja")
    user_prompts = [build_prompt("approach_10_task_keywords_user.txt.jinja", task=task) for task in tasks]
    task_keywords: list[TaskKeywordsResponse] = await query_llm(system_prompt, user_prompts, TaskKeywordsResponse)

    # Save keywords to cache
    print("Saving keywords to cache...")
    # Because of some pickling error, we need to recreate the same objects
    # Can't be bothered to dig into it right now, so just doign this workaround
    goal_keywords = [GoalKeywordsResponse(keywords=keywords.keywords) for keywords in goal_keywords]
    task_keywords = [TaskKeywordsResponse(keywords=keywords.keywords) for keywords in task_keywords]
    dump_to_pickle_file(goal_keywords, GOAL_KEYWORDS_OUTPUT_PATH)
    dump_to_pickle_file(task_keywords, TASK_KEYWORDS_OUTPUT_PATH)

    return goal_keywords, task_keywords


async def get_entities(
    goals: list[Goal], tasks: list[Task]
) -> tuple[list[GoalEntitiesResponse], list[TaskEntitiesResponse]]:
    """
    Get entities for tasks and goals using an LLM.
    """
    print("Getting entities for tasks and goals...")

    # Goal entities
    print("Getting entities for goals...")
    system_prompt = build_prompt("approach_10_goal_entities_system.txt.jinja")
    user_prompts = [build_prompt("approach_10_goal_entities_user.txt.jinja", goal=goal) for goal in goals]
    goal_entities: list[GoalEntitiesResponse] = await query_llm(system_prompt, user_prompts, GoalEntitiesResponse)

    # Task entities
    print("Getting entities for tasks...")
    system_prompt = build_prompt("approach_10_task_entities_system.txt.jinja")
    user_prompts = [build_prompt("approach_10_task_entities_user.txt.jinja", task=task) for task in tasks]
    task_entities: list[TaskEntitiesResponse] = await query_llm(system_prompt, user_prompts, TaskEntitiesResponse)

    # Save entities to cache
    print("Saving entities to cache...")
    # Because of some pickling error, we need to recreate the same objects
    # Can't be bothered to dig into it right now, so just doign this workaround
    goal_entities = [GoalEntitiesResponse(entities=entities.entities) for entities in goal_entities]
    task_entities = [TaskEntitiesResponse(entities=entities.entities) for entities in task_entities]
    dump_to_pickle_file(goal_entities, GOAL_ENTITIES_OUTPUT_PATH)
    dump_to_pickle_file(task_entities, TASK_ENTITIES_OUTPUT_PATH)

    return goal_entities, task_entities


async def query_llm(
    system_prompt: str,
    user_prompts: list[str],
    response_model: type[BaseModel],
    temperature: float = LLM_TEMPERATURE,
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> list[type[BaseModel]]:
    """
    Get keywords for the given user prompts using the specified system prompt.
    """

    set_async_instructor_client(llm_model=settings.llm_model, api_key=settings.anthropic_api_key)

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        limited_task(
            ainstruct_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                llm_model=settings.llm_model,
                temperature=temperature,
            ),
            semaphore,
            delay_between_tasks,
        )
        for user_prompt in user_prompts
    ]

    result: list[type[BaseModel]] = await execute_tasks_with_manual_pbar(tasks)

    return result


async def get_embeddings(
    goal_keywords: list[GoalKeywordsResponse],
    task_keywords: list[TaskKeywordsResponse],
    goal_entities: list[GoalEntitiesResponse],
    task_entities: list[TaskEntitiesResponse],
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    """
    Get embeddings for extracted text using OpenAI's embedding model.
    """
    print("Getting embeddings for extracted text...")

    # Get embeddings for goal keywords
    print("Embedding goal keywords...")
    goal_texts_to_embed = [keywords.keywords for keywords in goal_keywords]
    goal_keywords_embeddings: list[list[float]] = await embed_list_of_texts(goal_texts_to_embed)

    # Get embeddings for task keywords
    print("Embedding task keywords...")
    task_texts_to_embed = [keywords.keywords for keywords in task_keywords]
    task_keywords_embeddings: list[list[float]] = await embed_list_of_texts(task_texts_to_embed)

    # Get embeddings for goal entities
    print("Embedding goal entities...")
    goal_texts_to_embed = [entities.entities for entities in goal_entities]
    goal_entities_embeddings: list[list[float]] = await embed_list_of_texts(goal_texts_to_embed)

    # Get embeddings for task entities
    print("Embedding task entities...")
    task_texts_to_embed = [entities.entities for entities in task_entities]
    task_entities_embeddings: list[list[float]] = await embed_list_of_texts(task_texts_to_embed)

    # Save embeddings to cache
    print("Saving embeddings to cache...")
    dump_to_pickle_file(goal_keywords_embeddings, GOAL_KEYWORDS_EMBEDDINGS_OUTPUT_PATH)
    dump_to_pickle_file(task_keywords_embeddings, TASK_KEYWORDS_EMBEDDINGS_OUTPUT_PATH)
    dump_to_pickle_file(goal_entities_embeddings, GOAL_ENTITIES_EMBEDDINGS_OUTPUT_PATH)
    dump_to_pickle_file(task_entities_embeddings, TASK_ENTITIES_EMBEDDINGS_OUTPUT_PATH)

    return (
        goal_keywords_embeddings,
        task_keywords_embeddings,
        goal_entities_embeddings,
        task_entities_embeddings,
    )


async def embed_list_of_texts(
    list_of_texts: list[str],
    max_concurrent_tasks: int = 30,  # Max number of concurrent tasks
    delay_between_tasks: float = 0.1,  # Delay in seconds between task starts
) -> list[list[float]]:
    """
    Get embeddings for the given list of texts using OpenAI's embedding model.
    """

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


async def do_llm_as_a_judge_evaluations(augmented_tasks: list[AugmentedTask]) -> list[AugmentedTask]:
    """
    Do LLM as a judge evaluations per tasks for the goals that passed the cosine similarity filter.
    """
    print("Doing LLMaaJ evaluations per task for the goals that passed the cosine similarity filter...")

    print(f"There are {len(augmented_tasks)} tasks to evaluate...")

    for i, task in enumerate(augmented_tasks):
        print(f"Doing LLMaaJ evaluations for task {i + 1}...")

        embedding_filtered_goals: list[AugmentedGoal] = task.embedding_filtered_goals
        print(f"Number of goals to evaulate: {len(embedding_filtered_goals)}")

        # Step 1, do LLMaaJ analysis
        print("Doing LLMaaJ analysis...")
        system_prompt = build_prompt("approach_10_llmaaj_analysis_system.txt.jinja")
        user_prompts = [
            build_prompt("approach_10_llmaaj_analysis_user.txt.jinja", task=task, goal=goal)
            for goal in embedding_filtered_goals
        ]
        analysis_responses: list[LLMaaJAnalysisResponse] = await query_llm(
            system_prompt, user_prompts, LLMaaJAnalysisResponse
        )

        task.llmaaj_analysis = [
            LLMaaJAnalysisResult(goal=goal, llmaaj_analysis=analysis_response)
            for goal, analysis_response in zip(embedding_filtered_goals, analysis_responses)
        ]

        # Step 2, do LLMaaJ match judgement
        print("Doing LLMaaJ match judgement...")
        system_prompt = build_prompt("approach_10_llmaaj_judgement_system.txt.jinja")
        user_prompts = [
            build_prompt("approach_10_llmaaj_judgement_user.txt.jinja", task=task, goal=goal, llmaaj_analysis=analysis)
            for goal, analysis in zip(embedding_filtered_goals, analysis_responses)
        ]
        judgement_responses: list[LLMaaJResponse] = await query_llm(system_prompt, user_prompts, LLMaaJResponse)

        task.llmaaj_match_results = [
            GoalMatchResult(goal=goal, llmaaj_result=judgement_response)
            for goal, judgement_response in zip(embedding_filtered_goals, judgement_responses)
        ]

    return augmented_tasks


def get_suggested_goals(augmented_tasks: list[AugmentedTask]) -> list[AugmentedTask]:
    """
    Get suggested goals for each task based on the LLMaaJ evaluations.
    That is, the goals that were judged as a match by the LLM.
    """
    print("Getting suggested goals for each task based on the LLMaaJ evaluations...")

    for task in augmented_tasks:
        task.suggested_goals = [
            result.goal for result in task.llmaaj_match_results if result.llmaaj_result.decision == "yes"
        ]

    return augmented_tasks
