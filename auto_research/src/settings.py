import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


CLAUDE_OPUS = "claude-opus-4-6"
CLAUDE_SONNET = "claude-sonnet-4-6"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.secrets"), extra="ignore")
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    llm_model: str = Field(default=CLAUDE_OPUS)
    filter_model: str = Field(default=CLAUDE_SONNET)
    arxiv_categories: list[str] = Field(default=["cs.IR", "cs.AI", "cs.HC", "cs.GT", "cs.CL"])
    output_path: Path = Path(__file__).parent.parent / "reports"
    email_enabled: bool = Field(default=False)
    resend_api_key: SecretStr = Field(default=SecretStr(""))
    email_from: str = Field(default="onboarding@resend.dev")
    email_to: str = Field(default="")

    @property
    def root(self) -> Path:
        parent_directory_path = os.path.dirname(os.path.abspath(__file__))
        return Path(os.path.abspath(os.path.join(parent_directory_path, "..")))

    @property
    def repo_root(self) -> Path:
        return self.root.parent.parent


settings = Settings()

settings.output_path.mkdir(parents=True, exist_ok=True)
