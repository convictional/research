import asyncio
import pandas as pd
from pydantic import BaseModel, Field

from common.async_helper import limited_task, execute_tasks_with_manual_pbar
from common.instruct_llm import ainstruct_llm, set_async_instructor_client
from common.prompt_template_engine import build_prompt
from .raw_data_processing import PROCESSED_DATA_PATH
from .settings import settings

LLM_TEMPERATURE = 0.0

INDIVIDUAL_GOALS_DIR = settings.data_path / "individual_goals"
STATED_GOALS_PATH = INDIVIDUAL_GOALS_DIR / "pre_expert_rating_stated_goals.csv"
UNSTATED_GOALS_PATH = INDIVIDUAL_GOALS_DIR / "pre_expert_rating_unstated_goals.csv"


class Goal(BaseModel):
    """Individual business goal extracted from activity response."""
    goal_title: str = Field(..., description="Title for the business goal")
    goal_description: str = Field(..., description="Description of the business goal")


class ExtractedGoals(BaseModel):
    """Hierarchical structure for extracted stated and unstated goals."""
    stated_goals: list[Goal] = Field(..., description="List of stated business goals")
    unstated_goals: list[Goal] = Field(..., description="List of unstated business goals")


def load_processed_data() -> pd.DataFrame | None:
    """Load the processed goal mining data from CSV."""
    print("Loading processed data...")

    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: Processed data file not found: {PROCESSED_DATA_PATH}")
        print("Please run data processing first.")
        return None

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Column types:\n{df.dtypes}")

    return df


async def extract_goals_from_activity_response(title: str, activity_response: str) -> tuple[str, ExtractedGoals | None]:
    """Extract individual goals from a single activity response using LLM."""
    try:
        system_prompt = build_prompt("goal_extraction_system.txt.jinja")
        user_prompt = build_prompt("goal_extraction_user.txt.jinja", activity_response=activity_response)

        response = await ainstruct_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ExtractedGoals,
            llm_model=settings.llm_model,
            temperature=LLM_TEMPERATURE
        )

        return title, response
    except Exception as e:
        print(f"Error extracting goals for '{title}': {str(e)}")
        return title, None


async def process_all_goal_extractions(
        df: pd.DataFrame,
        max_concurrent_tasks: int = 30,
        delay_between_tasks: float = 0.1
    ) -> list[tuple[str, ExtractedGoals | None]]:
    """Process all DataFrame rows to extract goals asynchronously."""
    print("Starting async goal extraction from activity responses...")

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = [
        limited_task(
            extract_goals_from_activity_response(row['title'], row['activity_response']),
            semaphore,
            delay_between_tasks
        )
        for _, row in df.iterrows()
    ]

    results = await execute_tasks_with_manual_pbar(tasks)

    # Print results for each title
    print("\nGoal extraction summary by title:")
    for title, extracted_goals in results:
        stated_count = len(extracted_goals.stated_goals)
        unstated_count = len(extracted_goals.unstated_goals)
        print(f"'{title}': {stated_count} stated goals, {unstated_count} unstated goals")

    return results


def process_and_save_extraction_results(results: list[tuple[str, ExtractedGoals | None]]) -> None:
    """Process extraction results and save goals to CSV files."""
    print("Processing and saving extraction results...")

    # Create individual_goals directory if it doesn't exist
    INDIVIDUAL_GOALS_DIR.mkdir(exist_ok=True)

    # Collect all stated and unstated goals from all rows
    stated_goals_data = []
    unstated_goals_data = []

    # Skip None results from failed LLM extractions
    for title, extracted_goals in results:
        if extracted_goals is None:
            continue

        # Extract stated goals with title reference
        for goal in extracted_goals.stated_goals:
            stated_goals_data.append({
                'title': title,
                'goal_title': goal.goal_title,
                'goal_description': goal.goal_description
            })

        # Extract unstated goals with title reference
        for goal in extracted_goals.unstated_goals:
            unstated_goals_data.append({
                'title': title,
                'goal_title': goal.goal_title,
                'goal_description': goal.goal_description
            })

    # Create pandas DataFrames with columns: title, goal_title, goal_description
    stated_df = pd.DataFrame(stated_goals_data)
    unstated_df = pd.DataFrame(unstated_goals_data)

    # Save DataFrames to CSV files
    stated_df.to_csv(STATED_GOALS_PATH, index=False)
    unstated_df.to_csv(UNSTATED_GOALS_PATH, index=False)

    # Print counts of goals saved to each file
    print(f"Stated goals saved: {len(stated_goals_data)} goals to {STATED_GOALS_PATH}")
    print(f"Unstated goals saved: {len(unstated_goals_data)} goals to {UNSTATED_GOALS_PATH}")


async def extract_individual_goals() -> None:
    """Extract individual goals from processed goal mining data."""
    print("Extracting individual goals...")

    # Check if output files already exist
    if STATED_GOALS_PATH.exists() or UNSTATED_GOALS_PATH.exists():
        print("Goal extraction output files already exist. Skipping extraction.")
        return

    df = load_processed_data()
    if df is None:
        return

    # Initialize instructor client
    set_async_instructor_client(settings.llm_model, settings.anthropic_api_key)

    # Process all goal extractions
    results = await process_all_goal_extractions(df)
    print(f"Goal extraction completed. Processed {len(results)} responses.")

    # Process extraction results and save to CSV files
    process_and_save_extraction_results(results)
