import pandas as pd
import tiktoken

from .settings import settings

# NOTE: The input and output data for this experiment was a production export and was removed
# before open-sourcing. These paths describe the layout the code expects; supply your own CSV.
RAW_DATA_PATH = settings.data_path / "raw_data" / "raw_goal_mining_dump.csv"
PROCESSED_DATA_DIR = settings.data_path / "processed_data"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_goal_mining_dump.csv"


def count_tokens(text: str) -> int:
    """Count tokens in text using GPT-4 encoding."""
    if pd.isna(text) or not text:
        return 0

    encoding = tiktoken.encoding_for_model("gpt-4")

    return len(encoding.encode(text))


def add_token_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Add token count columns for activity_user_prompt."""
    print("Adding token counts to data...")

    df['num_gpt_4_tokens_user_prompt'] = df['activity_user_prompt'].apply(count_tokens)
    df['num_sonnet_4_tokens_user_prompt'] = (df['num_gpt_4_tokens_user_prompt'] * 1.12).round().astype(int)
    df = df.drop('num_gpt_4_tokens_user_prompt', axis=1)

    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """Save processed DataFrame to CSV in processed_data directory."""
    print("Dumping processed data...")

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include relevant columns for analysis."""
    print("Filtering data columns...")

    columns_to_keep = [
        'title',
        'activity_user_prompt',
        'activity_system_prompt',
        'activity_response',
        'activity_seed_data_created_date',
        'activity_weekly_updates_lookback_days',
        'activity_discussions_lookback_days',
        'activity_meetings_lookback_days',
        'activity_decisions_lookback_days',
        'activity_tasks_lookback_days',
        'activity_weekly_updates_total_count',
        'activity_discussions_total_count',
        'activity_meetings_total_count',
        'activity_decisions_total_count',
        'activity_tasks_total_count'
    ]

    return df[columns_to_keep]


def process_raw_data() -> None:
    """Load the raw goal mining data CSV file into a DataFrame."""
    print("Processing raw goal extraction data...")

    if PROCESSED_DATA_PATH.exists():
        print(f"Processed data already exists at: {PROCESSED_DATA_PATH}")
        print("Skipping data processing.")
        return

    if not RAW_DATA_PATH.exists():
        print(f"Error: Raw data file not found: {RAW_DATA_PATH}")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    df = filter_columns(df)
    print(f"After filtering: {df.shape[0]} rows, {df.shape[1]} columns")

    df = add_token_counts(df)
    print(f"After adding token counts: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Column types after processing:\n{df.dtypes}")

    save_processed_data(df)
