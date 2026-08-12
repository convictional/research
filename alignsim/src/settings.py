import getpass
import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CLAUDE_SONNET = "claude-sonnet-4-6"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.secrets"), extra="ignore")

    # API Keys
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))

    # LLM Configuration
    llm_model: str = Field(default=CLAUDE_SONNET)
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.3)

    # Game Configuration
    default_seed: int = Field(default=42)
    context_window_turns: int = Field(default=5)

    # Database Configuration
    local_postgres_user: str = Field(default=os.environ.get("POSTGRES_USER", getpass.getuser()))
    local_postgres_password: str = Field(default="")
    local_postgres_host: str = Field(default="localhost")
    local_postgres_port: int = Field(default=5432)
    local_postgres_db: str = Field(default="alignsim")

    # Paths
    output_path: Path = Path(__file__).parent.parent / "output"

    @property
    def root(self) -> Path:
        return Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))


settings = Settings()
