import os
import logging

from pydantic import Field, SecretStr, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Set our package's logger to INFO
logger = logging.getLogger("humans_and_llms")
logger.setLevel(logging.INFO)

current_env = (os.environ.get("ENV") or "development").lower()

# Removed redundant output_dir variable and its directory creation logic.


CLAUDE_HAIKU = "claude-3-haiku-20240307"
CLAUDE_SONNET = "claude-3-7-sonnet-20250219"
CLAUDE_OPUS = "claude-3-opus-20240229"

OPENAI_GPT4O = "gpt-4o"
OPENAI_GPT4O_MINI = "gpt-4o-mini"
OPENAI_GPT4 = "gpt-4-turbo-2024-04-09"
OPENAI_GPT35 = "gpt-3.5-turbo-1106"
OPENAI_O1 = "o1-2024-12-17"
OPENAI_O1_MINI = "o1-mini-2024-09-12"
OPENAI_O3_MINI = "o3-mini-2025-01-31"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.secrets"), extra="ignore")
    base_url: HttpUrl = Field(default=HttpUrl("http://localhost:8000"))
    gcp_project: str = ""  # This is found in .env file, which is included in the model_config, but there still has to be an attribute here for gcp_project
    output_path: Path = Path(__file__).parent / "output"
    input_path: Path = Path(__file__).parent / "input"
    llm_model: str = Field(default=CLAUDE_SONNET)
    openai_organization: str = ""
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    embedding_model: str = Field(default="text-embedding-3-small")  # same as vectors.py in the app
    embedding_dimension: int = 1536  # same as vectors.py in the app
    faiss_embedding_dimension: int = 3072
    faiss_embedding_model: str = Field(default="text-embedding-3-large")
    organization_id: str = "00000000-0000-0000-0000-000000000000"
    recall_ai_api_key: SecretStr = Field(default=SecretStr(""))
    local_postgres_dbname: str = "prod_db_dump"
    local_postgres_user: str = "postgres"
    local_postgres_password: str = ""
    local_postgres_host: str = "127.0.0.1"
    local_postgres_port: str = "5432"
    # BigQuery settings
    bigquery_datasets: str = Field(
        default="${GCP_PROJECT}.prod_mart_reporting,${GCP_PROJECT}.prod_core,${GCP_PROJECT}.prod_mart_finance",
        description="Comma-separated list of BigQuery datasets to search",
    )
    bigquery_use_authentication_default: bool = Field(
        default=True, description="Whether to use application default credentials"
    )
    # Query refinement settings
    max_query_attempts: int = Field(
        default=3, description="Maximum number of query refinement attempts through self-reflection"
    )

    # LLM thinking settings
    thinking_budget_tokens: int = Field(
        default=1024, description="Number of tokens allocated for LLM extended thinking budget"
    )

    @property
    def root(self):
        parent_directory_path = os.path.dirname(os.path.abspath(__file__))
        return Path(os.path.abspath(os.path.join(parent_directory_path, "..")))


settings = Settings()

# ensure input and output paths exist
settings.output_path.mkdir(parents=True, exist_ok=True)
