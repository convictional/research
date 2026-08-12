import os
import logging
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("train_research_report_judge")
logger.setLevel(logging.INFO)

CLAUDE_SONNET = "claude-sonnet-4-5-20250929"
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.secrets"), extra="ignore")

    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    llm_model: str = Field(default=CLAUDE_SONNET)

    raw_data_path: Path = Path(__file__).resolve().parent.parent.parent.parent / "tmp" / "research_output_evals.csv"
    data_path: Path = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_path: Path = Path(__file__).resolve().parent.parent / "output"

    train_ratio: float = 0.6
    dev_ratio: float = 0.2
    test_ratio: float = 0.2

    max_concurrency: int = 10
    delay_between_tasks: float = 0.1
    temperature: float = 0.1
    max_tokens: int = 4096

    rubric_batch_size: int = 12
    rubric_sample_size: int = 48
    calibration_example_count: int = 5
    calibration_max_chars: int = 2000

    # Ensemble scoring (Trial 8)
    ensemble_n: int = 5
    ensemble_temperature: float = 0.5

    # Claim analysis (Trial 9)
    claim_analysis_enabled: bool = True

    # Ceiling assessment metadata (Trial 10)
    include_metadata: bool = True

    # RAG verification scorer (Trial 11)
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    openai_organization: str | None = None
    organization_id: str = ""
    haiku_model: str = Field(default=CLAUDE_HAIKU)
    content_search_limit: int = 5
    content_table: str = "content"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    postgres_url: str = "asyncpg://decide:@localhost:5432/decide_development"
    rag_verification_enabled: bool = False

    max_iteration_rounds: int = 4
    target_spearman: float = 0.7
    target_mae: float = 0.5
    target_adjacent_match: float = 0.85

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
    def service_path(self) -> Path:
        return self.output_path / "service"


settings = Settings()

for path in [settings.data_path, settings.rubric_path, settings.results_path, settings.service_path]:
    path.mkdir(parents=True, exist_ok=True)
