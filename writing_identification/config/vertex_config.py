"""Vertex AI and environment-based configuration."""

import os
import json
import logging
from typing import Any, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .config import ExperimentConfig, config as base_config

logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()


class VertexConfig(BaseModel):
    """Configuration for Vertex AI training."""

    # GCP Settings
    gcp_project: str = Field(default="", description="GCP Project ID")
    gcs_bucket: str = Field(default="", description="GCS Bucket for artifacts")
    vertex_ai_enabled: bool = Field(default=False, description="Whether running on Vertex AI")

    # Experiment Settings
    experiment_name: str = Field(default="default", description="Experiment name for tracking")
    run_id: Optional[str] = Field(default=None, description="Unique run identifier")

    # Training Settings (can override base config)
    config_overrides: dict[str, Any] = Field(default_factory=dict, description="Config overrides")

    # Vertex AI specific
    job_name: Optional[str] = Field(default=None, description="Vertex AI job name")
    machine_type: str = Field(default="n1-standard-8", description="Machine type")
    accelerator_type: Optional[str] = Field(default="NVIDIA_TESLA_T4", description="GPU type")
    accelerator_count: int = Field(default=1, description="Number of GPUs")

    # Hyperparameter search
    is_hyperparam_search: bool = Field(default=False, description="Whether this is a hyperparam search")
    hyperparam_config: Optional[dict[str, Any]] = Field(default=None, description="Hyperparam search config")

    # Monitoring
    use_tensorboard: bool = Field(default=True, description="Enable TensorBoard logging")
    tensorboard_resource: Optional[str] = Field(default=None, description="Vertex AI TensorBoard resource")
    tensorboard_log_dir: Optional[str] = Field(default=None, description="Custom TensorBoard log directory")
    use_wandb: bool = Field(default=False, description="Enable Weights & Biases logging")

    @classmethod
    def from_environment(cls) -> "VertexConfig":
        """Create configuration from environment variables."""
        vertex_config = cls(
            gcp_project=os.getenv("GCP_PROJECT", ""),
            gcs_bucket=os.getenv("GCS_BUCKET", ""),
            vertex_ai_enabled=os.getenv("VERTEX_AI_ENABLED", "false").lower() == "true",
            experiment_name=os.getenv("EXPERIMENT_NAME", "default"),
            run_id=os.getenv("RUN_ID", None),
            job_name=os.getenv("CLOUD_ML_JOB_ID", None),  # Vertex AI sets this
            machine_type=os.getenv("MACHINE_TYPE", "n1-standard-8"),
            accelerator_type=os.getenv("ACCELERATOR_TYPE", "NVIDIA_TESLA_T4"),
            accelerator_count=int(os.getenv("ACCELERATOR_COUNT", "1")),
            is_hyperparam_search=os.getenv("IS_HYPERPARAM_SEARCH", "false").lower() == "true",
            use_tensorboard=os.getenv("USE_TENSORBOARD", "true").lower() == "true",
            tensorboard_resource=os.getenv("TENSORBOARD_RESOURCE", None),
            tensorboard_log_dir=os.getenv("TENSORBOARD_LOG_DIR", None),
            use_wandb=os.getenv("USE_WANDB", "false").lower() == "true",
        )

        # Parse config overrides from JSON string
        config_overrides_str = os.getenv("CONFIG_OVERRIDES", "{}")
        try:
            vertex_config.config_overrides = json.loads(config_overrides_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse CONFIG_OVERRIDES: {config_overrides_str}")
            vertex_config.config_overrides = {}

        # Parse hyperparameter config if present
        hyperparam_config_str = os.getenv("HYPERPARAM_CONFIG", None)
        if hyperparam_config_str:
            try:
                vertex_config.hyperparam_config = json.loads(hyperparam_config_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse HYPERPARAM_CONFIG: {hyperparam_config_str}")

        return vertex_config

    def apply_overrides(self, base_config: ExperimentConfig) -> ExperimentConfig:
        """
        Apply configuration overrides to base config.

        Args:
            base_config: Base experiment configuration

        Returns:
            Updated configuration with overrides applied
        """
        if not self.config_overrides:
            return base_config

        # Create a copy of the base config
        config_dict = base_config.model_dump()

        # Apply overrides recursively
        def apply_recursive(target: dict, overrides: dict, path: str = ""):
            for key, value in overrides.items():
                full_key = f"{path}.{key}" if path else key
                if key in target:
                    if isinstance(value, dict) and isinstance(target[key], dict):
                        apply_recursive(target[key], value, full_key)
                    else:
                        target[key] = value
                        logger.info(f"Config override: {full_key} = {value}")
                else:
                    # Allow new fields in nested configs (like model.encoder_type)
                    # Only warn for top-level unknown fields
                    if "." in path:
                        target[key] = value
                        logger.info(f"Config override (new field): {full_key} = {value}")
                    else:
                        logger.warning(f"Unknown top-level config key: {key}")

        apply_recursive(config_dict, self.config_overrides)

        # Recreate the config object
        return ExperimentConfig(**config_dict)


class HyperparameterSpace(BaseModel):
    """Defines hyperparameter search space."""

    learning_rates: list[float] = Field(default=[1e-5, 5e-5, 1e-4], description="Learning rates to try")
    batch_sizes: list[int] = Field(default=[16, 32, 64], description="Batch sizes to try")
    margins: list[float] = Field(default=[0.1, 0.2, 0.3], description="Contrastive loss margins to try")
    dropout_rates: list[float] = Field(default=[0.1, 0.2, 0.3], description="Dropout rates to try")
    encoder_types: list[str] = Field(default=["fusion", "attention"], description="Encoder architectures to try")
    hidden_dims: list[int] = Field(default=[256, 512, 768], description="Hidden dimensions to try")

    def generate_configs(self) -> list[dict[str, Any]]:
        """
        Generate all configuration combinations for grid search.

        Returns:
            List of configuration override dictionaries
        """
        configs = []

        # Simple grid search (can be extended to random or Bayesian)
        for lr in self.learning_rates:
            for batch_size in self.batch_sizes:
                for margin in self.margins:
                    for dropout in self.dropout_rates:
                        for encoder in self.encoder_types:
                            for hidden_dim in self.hidden_dims:
                                config = {
                                    "training": {"learning_rate": lr, "batch_size": batch_size, "margin": margin},
                                    "model": {
                                        "dropout_rate": dropout,
                                        "encoder_type": encoder,
                                        "hidden_dim": hidden_dim,
                                    },
                                }
                                configs.append(config)

        logger.info(f"Generated {len(configs)} hyperparameter configurations")
        return configs


def get_merged_config() -> tuple[ExperimentConfig, VertexConfig]:
    """
    Get merged configuration from base config and environment overrides.

    Returns:
        Tuple of (merged experiment config, vertex config)
    """
    # Get Vertex configuration from environment
    vertex_config = VertexConfig.from_environment()

    # Apply overrides to base config
    merged_config = vertex_config.apply_overrides(base_config)

    # Update device based on environment
    if vertex_config.vertex_ai_enabled:
        # Use CUDA on Vertex AI
        import torch

        if torch.cuda.is_available():
            merged_config.device = "cuda"
            logger.info(f"Running on Vertex AI with {torch.cuda.device_count()} GPU(s)")
        else:
            logger.warning("Vertex AI enabled but no CUDA available, using CPU")
            merged_config.device = "cpu"

    return merged_config, vertex_config


def save_config_for_vertex(config: ExperimentConfig, output_path: str = "vertex_config.json"):
    """
    Save configuration in a format suitable for Vertex AI.

    Args:
        config: Experiment configuration
        output_path: Path to save the config file
    """
    config_dict = config.dict()

    # Add metadata for Vertex AI
    config_dict["metadata"] = {"created_at": str(Path.cwd()), "config_version": "1.0.0"}

    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Config saved to {output_path}")
