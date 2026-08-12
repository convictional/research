"""Settings for LLM-based baseline evaluation."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Settings for LLM baseline evaluation."""

    model_config = SettingsConfigDict(env_file=("../.env", "../.env.secrets"), extra="ignore")

    # API Configuration
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))

    # Rate Limiting & Concurrency
    max_concurrent_requests: int = Field(default=10)
    rate_limit_per_minute: int = Field(default=100)
    llm_timeout: float = Field(default=30.0)

    # Model Configuration - Using latest models from Anthropic API
    default_llm_models: list[str] = Field(
        default=["claude-3-5-haiku-20241022", "claude-sonnet-4-20250514", "claude-opus-4-1-20250805"]
    )

    # Retry Configuration
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    backoff_factor: float = Field(default=2.0)

    # Temperature and generation settings
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1000)


# Global settings instance
llm_settings = LLMSettings()
