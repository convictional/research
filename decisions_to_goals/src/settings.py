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

logger = logging.getLogger("decisions_to_goals")
logger.setLevel(logging.INFO)

current_env = (os.environ.get("ENV") or "development").lower()

CLAUDE_OPUS = "claude-opus-4-7"
CLAUDE_SONNET = "claude-sonnet-4-6"
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(_PROJECT_ROOT / ".env"),
            str(_PROJECT_ROOT / ".env.secrets"),
        ),
        extra="ignore",
    )
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    openai_api_key: SecretStr = Field(default=SecretStr(""))

    # Model IDs — stamped into every output artifact's metadata so reruns months later remain comparable
    step1_model: str = Field(default=CLAUDE_OPUS)
    step2_model: str = Field(default=CLAUDE_SONNET)
    step4_model: str = Field(default=CLAUDE_SONNET)
    step5_model: str = Field(default=CLAUDE_OPUS)
    # Phase 2 — mapping
    mapping_analysis_model: str = Field(default=CLAUDE_SONNET)
    mapping_judgement_model: str = Field(default=CLAUDE_SONNET)
    # Phase 3 — summarization + judging
    # Sonnet is used (not Opus) because Opus rejects the temperature param
    # (see instruct_helper._NO_TEMPERATURE_MODELS). Determinism at temp=0 is
    # essential for the obfuscation layer to be reproducible across reruns.
    summarizer_model: str = Field(default=CLAUDE_SONNET)
    summary_word_target: int = Field(default=525)   # midpoint of the 450–600 band
    summary_word_min: int = Field(default=450)
    summary_word_max: int = Field(default=600)

    # LLM parameters
    temperature: float = 0.0
    max_tokens: int = 8192
    max_concurrency: int = 5
    mapping_max_concurrency: int = 8
    delay_between_tasks: float = 0.2

    org_id: str = Field(default="00000000-0000-0000-0000-000000000000")
    dataset_cutoff_date: str = Field(default="2025-04-29")

    # Data ingress: default to the local Postgres DB that `make research_load`
    # (in app/web) populates — host 127.0.0.1:5432, DB decide_{env}_decide, OS
    # user via trust auth (no password). asyncpg defaults the user to the OS
    # login, so no credentials are needed locally. Override with POSTGRES_DSN in
    # .env to point at a different database.
    postgres_dsn: str = Field(default=f"postgresql://127.0.0.1:5432/decide_{current_env}_decide")

    # Embedding settings
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dim: int = 1536
    # Cosine similarity above this threshold → treat as duplicate in step 3
    consolidation_similarity_threshold: float = 0.85
    dsm_score_threshold: float = Field(default=0.20)

    # Step 1 sampling limits
    max_activity_events_for_extraction: int = 150
    num_unstated_goals_to_extract: int = 12

    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def output_path(self) -> Path:
        return self.root / "output"

    @property
    def shared_output_path(self) -> Path:
        return self.output_path / "shared"

    @property
    def prompts_path(self) -> Path:
        return self.root / "src" / "prompts"

    def init_output_dirs(self) -> None:
        for path in [
            self.shared_output_path,
            self.output_path / "unstated",
            self.output_path / "stated",
            self.output_path / "mixed",
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def condition_output_path(self, condition: str) -> Path:
        valid = {"unstated", "stated", "mixed"}
        if condition not in valid:
            raise ValueError(f"Unknown condition '{condition}'. Must be one of {valid}.")
        return self.output_path / condition

    @property
    def model_ids(self) -> dict:
        return {
            "step1": self.step1_model,
            "step2": self.step2_model,
            "step4": self.step4_model,
            "step5": self.step5_model,
            "mapping_analysis": self.mapping_analysis_model,
            "mapping_judgement": self.mapping_judgement_model,
            "summarizer": self.summarizer_model,
        }


settings = Settings()
