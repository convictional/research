import os
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.secrets"), extra="ignore")

    # Database
    postgres_url: str = "postgresql://decide:@localhost:5432/decide_development_decide__gemma_4_vs_claude_for_"

    # Anthropic (Sonnet, Haiku)
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    sonnet_model: str = "claude-sonnet-4-6"
    haiku_model: str = "claude-haiku-4-5-20251001"

    # Google Vertex (Gemma)
    google_vertex_project: str = Field(
        default="",
        validation_alias=AliasChoices("google_vertex_project", "GCP_PROJECT"),
    )
    google_vertex_location: str = "us-central1"
    gemma_model: str = "google/gemma-4-26b-a4b-it-maas"

    # Extraction
    max_concurrent: int = 5
    max_results_per_query: int = 10
    max_learnings_per_query: int = 12
    max_tokens_per_result: int = 8500

    @property
    def root(self) -> Path:
        parent_directory_path = os.path.dirname(os.path.abspath(__file__))
        return Path(os.path.abspath(os.path.join(parent_directory_path, "..")))

    @property
    def output_path(self) -> Path:
        return self.root / "output"


settings = Settings()

settings.output_path.mkdir(parents=True, exist_ok=True)


MODEL_PRICING_USD_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "google/gemma-4-26b-a4b-it-maas": {"input": 0.15, "output": 0.60},
}


def cost_usd(usage: dict, model_name: str) -> float | None:
    prices = MODEL_PRICING_USD_PER_M_TOKENS.get(model_name)
    if not prices or not usage:
        return None
    return (usage.get("input_tokens", 0) * prices["input"] + usage.get("output_tokens", 0) * prices["output"]) / 1_000_000
