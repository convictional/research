import logging
import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("goal_alignment_judge")
logger.setLevel(logging.INFO)

current_env = (os.environ.get("ENV") or "development").lower()

DATABASE = "decide_development_decide__inter_rater_goal_align"
MIN_CONTENT_ITEMS_PER_GOAL = 10
EXPERIMENT_NAME = "goal_alignment_judge"

CLAUDE_OPUS = "claude-opus-4-6"
CLAUDE_SONNET = "claude-sonnet-4-6"
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.secrets"), extra="ignore")
    gcp_project: str = ""
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    rubric_model: str = Field(default=CLAUDE_OPUS)
    scorer_model: str = Field(default=CLAUDE_SONNET)
    ranking_model: str = Field(default=CLAUDE_HAIKU)

    # Paths
    data_path: Path = Path(__file__).parent / "data"
    ratings_dir: Path = Path(__file__).parent / "data" / "ratings"
    cases_csv_path: Path = Path(__file__).parent / "data" / "cases.csv"
    processed_data_path: Path = Path(__file__).parent / "data" / "processed"
    output_path: Path = Path(__file__).resolve().parent.parent / "output"

    # Data processing
    win_margin: float = 0.25
    train_ratio: float = 0.50
    dev_ratio: float = 0.25

    # LLM scoring
    temperature: float = 0.0
    max_tokens: int = 16384
    max_concurrency: int = 10
    delay_between_tasks: float = 0.1

    # Rubric discovery
    rubric_batch_size: int = 10
    rubric_sample_size: int = 120

    # Ensemble
    ensemble_n: int = 1
    ensemble_temperature: float = 0.1

    # Iteration
    max_iteration_rounds: int = 3
    target_pairwise_accuracy: float = 0.90

    # Pointwise pipeline
    pointwise_input_csv: Path = Path(__file__).resolve().parent.parent / "input" / "goal_alignments_rated.csv"
    pointwise_processed_path: Path = Path(__file__).parent / "data" / "processed" / "pointwise"
    negative_similarity_min: float = 0.15
    negative_similarity_max: float = 0.45
    negatives_per_goal: int = 5
    target_pointwise_f1: float = 0.70
    pointwise_rubric_sample_size: int = 40
    pointwise_rubric_batch_size: int = 8

    # Generalization outer loop
    generalization_gap_threshold: float = 0.15
    generalization_guard_rail: float = 0.75
    generalization_min_rounds: int = 4

    # Goal ID -> owner's rater name (for preferring owner ratings over majority vote).
    # Rater names are anonymised here; goals with no matching rater fall back to
    # majority vote. The underlying rating data is not included in this repository.
    goal_owner_rater: dict[str, str] = {
        "c67bd8d1-a48c-44b5-bd8c-ea456c9debd9": "rater1",  # AI Moat
        "765ec5fc-2d8b-4779-9a3e-df356ee791f1": "rater2",  # Capital Allocation
        "85ca9d5f-8eb6-4444-b496-c760942eb0e6": "rater3",  # Activation
        "6be147a5-1d85-4f36-91f2-8ca648561088": "rater3",  # Integration
        "3b5f551e-0e33-4824-a629-1d613f40a887": "rater3",  # Momentum
        "f15d2519-6312-4f43-aec0-ed28f4bdeb6f": "rater3",  # Retention
        "6c307179-1409-4a62-9b33-cae9bd705728": "rater3",  # Validation
    }

    # Few-shot example (content_id, goal_id) pairs — all Conversion goal (excluded from splits)
    fewshot_pairs: list[tuple[str, str]] = [
        (
            "c5fe324c-de32-45cb-a382-079faf89cd31",
            "09bc1c9e-c50d-4346-aeeb-55fe57f67566",
        ),  # pinned: a customer onboarding thread
        (
            "378ba13d-566c-4594-af6d-5f2a337f32ee",
            "09bc1c9e-c50d-4346-aeeb-55fe57f67566",
        ),  # deleted: Powered by Stripe?
        ("6b7a8319-f6ac-4ec7-904c-8a691b6bae47", "09bc1c9e-c50d-4346-aeeb-55fe57f67566"),  # neutral: a 1:1 thread
    ]

    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def rubric_path(self) -> Path:
        return self.output_path / "rubric"

    @property
    def results_path(self) -> Path:
        return self.output_path / "results"

    @property
    def eda_path(self) -> Path:
        return self.output_path / "eda"

    @property
    def ranking_path(self) -> Path:
        return self.output_path / "ranking"

    @property
    def dspy_path(self) -> Path:
        return self.output_path / "dspy"


settings = Settings()

for path in [
    settings.data_path,
    settings.processed_data_path,
    settings.results_path,
    settings.eda_path,
    settings.pointwise_processed_path,
    settings.dspy_path,
]:
    path.mkdir(parents=True, exist_ok=True)
