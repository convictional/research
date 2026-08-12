import os
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

current_env = (os.environ.get("ENV") or "development").lower()


CLAUDE_HAIKU = "claude-3-haiku-20240307"
CLAUDE_SONNET_35 = "claude-3-5-sonnet-20241022"
CLAUDE_SONNET_37 = "claude-3-7-sonnet-20250219"
CLAUDE_OPUS = "claude-3-opus-20240229"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.secrets"), extra="ignore")
    gcp_project: str = ""  # This is found in .env file, which is included in the model_config, but there still has to be an attribute here for gcp_project
    output_path: Path = Path(__file__).parent / "output"
    input_path: Path = Path(__file__).parent / "input"
    llm_model: str = Field(default=CLAUDE_SONNET_37)
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))

    @property
    def root(self):
        parent_directory_path = os.path.dirname(os.path.abspath(__file__))
        return Path(os.path.abspath(os.path.join(parent_directory_path, "..")))


settings = Settings()

# ensure input and output paths exist
settings.output_path.mkdir(parents=True, exist_ok=True)
settings.input_path.mkdir(parents=True, exist_ok=True)
