import os
from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

current_env = (os.environ.get("ENV") or "development").lower()

CLAUDE_HAIKU = "claude-3-haiku-20240307"
CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
CLAUDE_OPUS = "claude-3-opus-20240229"

OPENAI_GPT4O = "gpt-4o"
OPENAI_GPT4O_MINI = "gpt-4o-mini"
OPENAI_GPT4 = "gpt-4-turbo-2024-04-09"
OPENAI_GPT35 = "gpt-3.5-turbo-1106"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.secrets"), extra="ignore")
    env: str = Field(default=current_env)
    log_level: str = Field(default="info")
    gcp_project: str = ""
    gcp_location: str = ""
    gcs_bucket: str = ""
    openai_organization: str = ""
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    neo4j_dsn: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: SecretStr = Field(default=SecretStr(""))
    output_path: Path = Path(__file__).parent / "output"
    input_path: Path = Path(__file__).parent / "input"
    llm_model: str = Field(default=CLAUDE_SONNET)
    embedding_model: str = Field(default="text-embedding-3-large")
    # Note that larger dimensions for embeddings will chew up neo4j memory and you'll have a bad time
    # we may be able to increase this in production, but keeping it small for now to avoid memory issues
    # while we use the free tier of neo4j
    embedding_dimension: int = 256
    # Determines the number of guru chunks to retrieve and include in manual graph generation
    embedded_guru_top_k: int = 2
    faiss_embedding_dimension: int = 3072
    faiss_embedding_model: str = Field(default="text-embedding-3-large")
    local_postgres_dbname: str = "prod_db_dump"
    local_postgres_user: str = "postgres"
    local_postgres_password: str = ""
    local_postgres_host: str = "127.0.0.1"
    local_postgres_port: str = "5432"
    organization_id: str = "00000000-0000-0000-0000-000000000000"

    def is_env(self, *environments: str) -> bool:
        return self.env in environments


settings = Settings()

# ensure input and output paths exist
settings.input_path.mkdir(parents=True, exist_ok=True)
settings.output_path.mkdir(parents=True, exist_ok=True)
