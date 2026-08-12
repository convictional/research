import os
import logging
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Set default level to WARNING for all loggers
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set our package's logger to INFO
logger = logging.getLogger("named_entity_knowledge_store")
logger.setLevel(logging.INFO)

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

    @property
    def root(self):
        parent_directory_path = os.path.dirname(os.path.abspath(__file__))
        return Path(os.path.abspath(os.path.join(parent_directory_path, "..")))


settings = Settings()

# ensure input and output paths exist
settings.output_path.mkdir(parents=True, exist_ok=True)
