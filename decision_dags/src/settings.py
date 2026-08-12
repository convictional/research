import os
from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

current_env = (os.environ.get("ENV") or "development").lower()

# Model constants
CLAUDE_SONNET = "claude-sonnet-4-20250514"
CLAUDE_OPUS = "claude-opus-4-20250514"

GPT_4O_MINI = "gpt-4o-mini"
GPT_4O = "gpt-4o"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.secrets"), extra="ignore")

    # API Keys
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    openai_api_key: SecretStr = Field(default=SecretStr(""))

    # Database Configuration
    local_postgres_user: str = Field(default=os.environ.get("POSTGRES_USER", "postgres"))
    local_postgres_password: str = Field(default="")
    local_postgres_host: str = Field(default="localhost")
    local_postgres_port: int = Field(default=5432)
    local_postgres_db: str = Field(default="decision_dags_experiment")
    embedding_dimension: int = Field(default=1536)

    # Paths
    output_path: Path = Path(__file__).parent / "output"
    input_path: Path = Path(__file__).parent / "input"

    # LLM Configuration
    llm_model: str = Field(default=CLAUDE_SONNET)
    embedding_model: str = Field(default="text-embedding-3-small")

    # DAG Builder Configuration
    max_concurrent_agents: int = Field(default=10)
    agent_timeout: float = Field(default=30.0)
    max_layers: int = Field(default=6)
    max_children_per_node: int = Field(default=5)
    min_children_per_node: int = Field(default=2)

    # Deduplication Configuration
    similarity_threshold: float = Field(default=0.8)
    weak_similarity_threshold: float = Field(default=0.6)

    # LLM Settings
    generation_temperature: float = Field(default=0.7)
    assessment_temperature: float = Field(default=0.3)
    max_retries: int = Field(default=3)

    # Evolution Configuration
    max_concurrent_evolutions: int = Field(default=4)
    max_iterations_per_path: int = Field(default=3)
    min_improvement_threshold: float = Field(default=0.1)

    # Environment
    environment: str = Field(default=current_env)

    # Organization Configuration
    organization_id: str = Field(default="00000000-0000-0000-0000-000000000000")

    @property
    def root(self):
        parent_directory_path = os.path.dirname(os.path.abspath(__file__))
        return Path(os.path.abspath(os.path.join(parent_directory_path, "..")))

    def is_env(self, env_name: str) -> bool:
        return self.environment == env_name.lower()


settings = Settings()

# Ensure input and output paths exist
settings.output_path.mkdir(parents=True, exist_ok=True)
settings.input_path.mkdir(parents=True, exist_ok=True)
