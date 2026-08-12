import pandas as pd
from pydantic import BaseModel, Field

from common.async_helper import execute_tasks_with_manual_pbar
from common.instruct_llm import ainstruct_llm, set_async_instructor_client
from common.prompt_template_engine import build_prompt
from .settings import settings

RATER_FEEDBACK_DIR = settings.data_path / "rater_feedback_analysis"
STATED_GOALS_FEEDBACK_PATH = RATER_FEEDBACK_DIR / "rated_stated_goals_with_feedback.csv"
UNSTATED_GOALS_FEEDBACK_PATH = RATER_FEEDBACK_DIR / "rated_unstated_goals_with_feedback.csv"
CEO_UNSTATED_ANALYSIS_PATH = RATER_FEEDBACK_DIR / "rater_feedback_ceo_unstated_goals_analysis.md"
CEO_STATED_ANALYSIS_PATH = RATER_FEEDBACK_DIR / "rater_feedback_ceo_stated_goals_analysis.md"
OTHER_UNSTATED_ANALYSIS_PATH = RATER_FEEDBACK_DIR / "rater_feedback_other_unstated_goals_analysis.md"
OTHER_STATED_ANALYSIS_PATH = RATER_FEEDBACK_DIR / "rater_feedback_other_stated_goals_analysis.md"

LLM_TEMPERATURE = 0.0


class LLMResponse(BaseModel):
    """Response model for LLM analysis of feedback."""
    analysis: str = Field(..., description="Comprehensive analysis of rating patterns")


def load_rater_feedback_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both stated and unstated goals with rater feedback."""
    print("Loading rater feedback data...")

    if not STATED_GOALS_FEEDBACK_PATH.exists():
        raise FileNotFoundError(f"Stated goals feedback file not found: {STATED_GOALS_FEEDBACK_PATH}")

    if not UNSTATED_GOALS_FEEDBACK_PATH.exists():
        raise FileNotFoundError(f"Unstated goals feedback file not found: {UNSTATED_GOALS_FEEDBACK_PATH}")

    stated_df = pd.read_csv(STATED_GOALS_FEEDBACK_PATH)
    unstated_df = pd.read_csv(UNSTATED_GOALS_FEEDBACK_PATH)

    print(f"Stated goals feedback loaded: {stated_df.shape[0]} rows, {stated_df.shape[1]} columns")
    print(f"Unstated goals feedback loaded: {unstated_df.shape[0]} rows, {unstated_df.shape[1]} columns")

    return stated_df, unstated_df


async def analyze_ceo_feedback_unstated_goals(unstated_df: pd.DataFrame) -> None:
    """Analyze the CEO rater feedback specifically for unstated goals."""
    print("Analyzing CEO feedback for unstated goals...")

    # Check if analysis already exists
    if CEO_UNSTATED_ANALYSIS_PATH.exists():
        print(f"CEO unstated goals analysis already exists at: {CEO_UNSTATED_ANALYSIS_PATH}")
        print("Skipping CEO unstated goals analysis.")
        return

    print(f"Analyzing {len(unstated_df)} unstated goals for CEO feedback")

    # Prepare data for LLM analysis
    ceo_goals_data = []
    for _, row in unstated_df.iterrows():
        ceo_goals_data.append({
            'goal_title': row['goal_title'],
            'goal_description': row['goal_description'],
            'rater1_rating': row['rater1_rating'],
            'rater1_rating_why': row['rater1_rating_why']
        })

    # Build prompts
    system_prompt = build_prompt("rater_feedback_ceo_unstated_goals_system.txt.jinja")
    user_prompt = build_prompt(
        "rater_feedback_ceo_unstated_goals_user.txt.jinja",
        ceo_goals_data=ceo_goals_data
    )

    # Call LLM with progress bar
    llm_task = ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=LLMResponse,
        llm_model=settings.llm_model,
        temperature=LLM_TEMPERATURE
    )

    results = await execute_tasks_with_manual_pbar([llm_task])
    response = results[0]

    # Save analysis to markdown file
    with open(CEO_UNSTATED_ANALYSIS_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# CEO Feedback Analysis - Unstated Goals\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write(response.analysis)

    print(f"CEO unstated goals analysis saved to: {CEO_UNSTATED_ANALYSIS_PATH}")


async def analyze_ceo_feedback_stated_goals(stated_df: pd.DataFrame) -> None:
    """Analyze the CEO rater feedback specifically for stated goals."""
    print("Analyzing CEO feedback for stated goals...")

    # Check if analysis already exists
    if CEO_STATED_ANALYSIS_PATH.exists():
        print(f"CEO stated goals analysis already exists at: {CEO_STATED_ANALYSIS_PATH}")
        print("Skipping CEO stated goals analysis.")
        return

    print(f"Analyzing {len(stated_df)} stated goals for CEO feedback")

    # Prepare data for LLM analysis
    ceo_goals_data = []
    for _, row in stated_df.iterrows():
        ceo_goals_data.append({
            'goal_title': row['goal_title'],
            'goal_description': row['goal_description'],
            'rater1_rating': row['rater1_rating'],
            'rater1_rating_why': row['rater1_rating_why']
        })

    # Build prompts
    system_prompt = build_prompt("rater_feedback_ceo_stated_goals_system.txt.jinja")
    user_prompt = build_prompt(
        "rater_feedback_ceo_stated_goals_user.txt.jinja",
        ceo_goals_data=ceo_goals_data
    )

    # Call LLM with progress bar
    llm_task = ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=LLMResponse,
        llm_model=settings.llm_model,
        temperature=LLM_TEMPERATURE
    )

    results = await execute_tasks_with_manual_pbar([llm_task])
    response = results[0]

    # Save analysis to markdown file
    with open(CEO_STATED_ANALYSIS_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# CEO Feedback Analysis - Stated Goals\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write(response.analysis)

    print(f"CEO stated goals analysis saved to: {CEO_STATED_ANALYSIS_PATH}")


async def analyze_other_feedback_unstated_goals(unstated_df: pd.DataFrame) -> None:
    """Analyze other raters' feedback specifically for unstated goals."""
    print("Analyzing other raters' feedback for unstated goals...")

    # Check if analysis already exists
    if OTHER_UNSTATED_ANALYSIS_PATH.exists():
        print(f"Other raters' unstated goals analysis already exists at: {OTHER_UNSTATED_ANALYSIS_PATH}")
        print("Skipping other raters' unstated goals analysis.")
        return

    print(f"Analyzing {len(unstated_df)} unstated goals for other raters' feedback")

    # Prepare data for LLM analysis - include all goals regardless of completeness
    other_goals_data = []
    for _, row in unstated_df.iterrows():
        other_goals_data.append({
            'goal_title': row['goal_title'],
            'goal_description': row['goal_description'],
            'rater2_rating': row['rater2_rating'],
            'rater2_rating_why': row['rater2_rating_why'],
            'rater3_rating': row['rater3_rating'],
            'rater3_rating_why': row['rater3_rating_why']
        })

    # Build prompts
    system_prompt = build_prompt("rater_feedback_other_unstated_goals_system.txt.jinja")
    user_prompt = build_prompt(
        "rater_feedback_other_unstated_goals_user.txt.jinja",
        other_goals_data=other_goals_data
    )

    # Call LLM with progress bar
    llm_task = ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=LLMResponse,
        llm_model=settings.llm_model,
        temperature=LLM_TEMPERATURE
    )

    results = await execute_tasks_with_manual_pbar([llm_task])
    response = results[0]

    # Save analysis to markdown file
    with open(OTHER_UNSTATED_ANALYSIS_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# Other Raters' Feedback Analysis - Unstated Goals\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write(response.analysis)

    print(f"Other raters' unstated goals analysis saved to: {OTHER_UNSTATED_ANALYSIS_PATH}")


async def analyze_other_feedback_stated_goals(stated_df: pd.DataFrame) -> None:
    """Analyze other raters' feedback specifically for stated goals."""
    print("Analyzing other raters' feedback for stated goals...")

    if OTHER_STATED_ANALYSIS_PATH.exists():
        print(f"Other raters' stated goals analysis already exists at: {OTHER_STATED_ANALYSIS_PATH}")
        print("Skipping other raters' stated goals analysis.")
        return

    print(f"Analyzing {len(stated_df)} stated goals for other raters' feedback")

    other_goals_data = []
    for _, row in stated_df.iterrows():
        other_goals_data.append({
            'goal_title': row['goal_title'],
            'goal_description': row['goal_description'],
            'rater2_rating': row['rater2_rating'],
            'rater2_rating_why': row['rater2_rating_why'],
            'rater3_rating': row['rater3_rating'],
            'rater3_rating_why': row['rater3_rating_why']
        })

    system_prompt = build_prompt("rater_feedback_other_stated_goals_system.txt.jinja")
    user_prompt = build_prompt(
        "rater_feedback_other_stated_goals_user.txt.jinja",
        other_goals_data=other_goals_data
    )

    llm_task = ainstruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=LLMResponse,
        llm_model=settings.llm_model,
        temperature=LLM_TEMPERATURE
    )

    results = await execute_tasks_with_manual_pbar([llm_task])
    response = results[0]

    with open(OTHER_STATED_ANALYSIS_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# Other Raters' Feedback Analysis - Stated Goals\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write(response.analysis)

    print(f"Other raters' stated goals analysis saved to: {OTHER_STATED_ANALYSIS_PATH}")


async def analyze_other_feedback(stated_df: pd.DataFrame, unstated_df: pd.DataFrame) -> None:
    """Analyze feedback patterns from other raters (Rater 2 and Rater 3)."""
    print("Analyzing other raters' feedback...")

    await analyze_other_feedback_unstated_goals(unstated_df)
    await analyze_other_feedback_stated_goals(stated_df)


async def analyze_ceo_feedback(stated_df: pd.DataFrame, unstated_df: pd.DataFrame) -> None:
    """Main function to analyze the CEO rater feedback patterns."""
    print("Starting CEO feedback analysis...")

    await analyze_ceo_feedback_unstated_goals(unstated_df)
    await analyze_ceo_feedback_stated_goals(stated_df)


async def rater_feedback_analysis() -> None:
    """Analyze rater feedback data."""
    print("Starting rater feedback analysis...")

    # Initialize instructor client
    set_async_instructor_client(settings.llm_model, settings.anthropic_api_key)

    # Load the data
    stated_df, unstated_df = load_rater_feedback_data()

    # Run CEO-specific analysis
    await analyze_ceo_feedback(stated_df, unstated_df)

    # Run other raters analysis
    await analyze_other_feedback(stated_df, unstated_df)
