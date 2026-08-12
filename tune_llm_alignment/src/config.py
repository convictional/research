"""Configuration loader for the experiment."""

import os
import tomllib
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv


class Config:
    """Load and access experiment configuration."""

    def __init__(self, config_path: str = "config.toml"):
        # Load secrets from .env.secrets
        load_dotenv(".env.secrets")

        # Load config from TOML
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "rb") as f:
            self._config: Dict[str, Any] = tomllib.load(f)

    def __getattr__(self, name: str) -> Any:
        """Allow dot notation access to config sections."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return self._config.get(name, {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'models.generator_model')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def gemini_api_key(self) -> str:
        """Get Gemini API key from environment."""
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please set it in .env.secrets"
            )
        return key

    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic API key from environment."""
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Please set it in .env.secrets"
            )
        return key

    @property
    def seed_db_path(self) -> Path:
        """Get path to seed database."""
        return Path(self.get("database.seed_db_path"))

    @property
    def user_id(self) -> str:
        """Get user ID for filtering data."""
        user_id = self.get("user.user_id")
        if not user_id:
            raise ValueError("user_id not set in config.toml")
        return user_id


# Global config instance
config = Config()
