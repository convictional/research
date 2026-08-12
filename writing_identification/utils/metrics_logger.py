"""Unified metrics logging for training experiments."""

import json
import logging
from typing import Any, Optional, Union
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Vertex AI
try:
    from google.cloud import aiplatform

    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Unified metrics logger supporting TensorBoard, Vertex AI, and JSON."""

    def __init__(
        self,
        experiment_name: str,
        run_id: str,
        config: dict[str, Any],
        enable_tensorboard: bool = True,
        enable_vertex_ai: bool = False,
        enable_json: bool = True,
        log_dir: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """
        Initialize metrics logger.

        Args:
            experiment_name: Name of the experiment
            run_id: Unique run identifier
            config: Configuration dict to log
            enable_tensorboard: Enable TensorBoard logging
            enable_vertex_ai: Enable Vertex AI experiment tracking
            enable_json: Enable JSON file logging
            log_dir: Local directory for logs (defaults to runs/)
            gcs_bucket: GCS bucket name for TensorBoard logs
            project_id: GCP project ID for Vertex AI
        """
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.config = config
        self.project_id = project_id

        # Create log directory
        if log_dir is None:
            log_dir = f"runs/{experiment_name}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backends
        self.tensorboard_writer = None
        self.vertex_ai_enabled = False
        self.json_enabled = enable_json

        # JSON logging setup
        if self.json_enabled:
            self.json_file = self.log_dir / f"metrics_{run_id}.json"
            self.metrics_history = []

        # Initialize TensorBoard
        if enable_tensorboard and TENSORBOARD_AVAILABLE:
            self._init_tensorboard(gcs_bucket)

        # Initialize Vertex AI
        if enable_vertex_ai and VERTEX_AI_AVAILABLE:
            self._init_vertex_ai()

        # Log initial configuration
        self.log_config(config)

    def _init_tensorboard(self, gcs_bucket: Optional[str] = None):
        """Initialize TensorBoard logging."""
        try:
            if gcs_bucket:
                # Use GCS path for Vertex AI TensorBoard integration
                log_dir = f"gs://{gcs_bucket}/tensorboard/{self.experiment_name}/{self.run_id}"
            else:
                # Use local directory
                log_dir = self.log_dir / "tensorboard"

            self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard initialized: {log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.tensorboard_writer = None

    def _init_vertex_ai(self):
        """Initialize Vertex AI experiment tracking."""
        try:
            if self.project_id:
                aiplatform.init(project=self.project_id, location="us-central1", experiment=self.experiment_name)
                aiplatform.start_run(run=self.run_id)
                self.vertex_ai_enabled = True
                logger.info(f"Vertex AI initialized: {self.experiment_name}/{self.run_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize Vertex AI: {e}")
            self.vertex_ai_enabled = False

    def log_config(self, config: dict[str, Any]):
        """Log experiment configuration to all backends."""
        # Flatten config for logging
        flat_config = self._flatten_dict(config)

        # TensorBoard hyperparameters
        if self.tensorboard_writer:
            try:
                # Convert config to hparams format
                hparams = {}
                for k, v in flat_config.items():
                    if isinstance(v, (int, float, str, bool)):
                        hparams[k] = v
                    else:
                        hparams[k] = str(v)

                # Log as hyperparameters table
                self.tensorboard_writer.add_hparams(hparams, {})
            except Exception as e:
                logger.warning(f"Failed to log config to TensorBoard: {e}")

        # Vertex AI parameters
        if self.vertex_ai_enabled:
            try:
                aiplatform.log_params(flat_config)
            except Exception as e:
                logger.warning(f"Failed to log config to Vertex AI: {e}")

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None):
        """Log metrics to all enabled backends."""
        # Sanitize metrics
        clean_metrics = self._sanitize_metrics(metrics)

        # Add timestamp
        timestamp = datetime.now().isoformat()
        if step is not None:
            clean_metrics["step"] = step

        # JSON logging
        if self.json_enabled:
            log_entry = {"timestamp": timestamp, "step": step, **clean_metrics}
            self.metrics_history.append(log_entry)
            self._save_json()

        # TensorBoard logging
        if self.tensorboard_writer:
            try:
                for key, value in clean_metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(key, value, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log metrics to TensorBoard: {e}")

        # Vertex AI logging
        if self.vertex_ai_enabled:
            try:
                # Use time series metrics for proper step tracking
                if step is not None:
                    time_series_metrics = {}
                    for key, value in clean_metrics.items():
                        if isinstance(value, (int, float)):
                            time_series_metrics[key] = value
                    aiplatform.log_time_series_metrics(time_series_metrics)
                else:
                    aiplatform.log_metrics(clean_metrics)
            except Exception as e:
                logger.warning(f"Failed to log metrics to Vertex AI: {e}")

    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """Log histogram data (mainly for TensorBoard)."""
        if self.tensorboard_writer:
            try:
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().numpy()
                self.tensorboard_writer.add_histogram(name, values, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log histogram {name}: {e}")

    def log_embeddings(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None, step: Optional[int] = None
    ):
        """Log embedding visualization."""
        if self.tensorboard_writer:
            try:
                # Sample embeddings if too large
                if embeddings.size(0) > 1000:
                    indices = torch.randperm(embeddings.size(0))[:1000]
                    embeddings = embeddings[indices]
                    if labels is not None:
                        labels = labels[indices]

                self.tensorboard_writer.add_embedding(
                    embeddings.detach().cpu(),
                    metadata=labels.detach().cpu().tolist() if labels is not None else None,
                    global_step=step,
                    tag="embeddings",
                )
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log embeddings: {e}")

    def log_model_graph(self, model: torch.nn.Module, input_batch: tuple):
        """Log model graph to TensorBoard."""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_graph(model, input_batch)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")

    def close(self):
        """Close all logging backends."""
        # Save final JSON
        if self.json_enabled:
            self._save_json()

        # Close TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        # End Vertex AI run
        if self.vertex_ai_enabled:
            try:
                aiplatform.end_run()
            except Exception as e:
                logger.warning(f"Failed to end Vertex AI run: {e}")

    def _sanitize_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Convert metrics to loggable format."""
        sanitized = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):  # NumPy/PyTorch scalar
                sanitized[k] = v.item()
            elif isinstance(v, (int, float, str, bool)):
                sanitized[k] = v
            elif v is None:
                continue
            else:
                # Convert complex types to string
                sanitized[k] = str(v)
        return sanitized

    def _flatten_dict(self, d: dict[str, Any], parent_key: str = "", sep: str = "_") -> dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            elif v is not None:
                items.append((new_key, v))
        return dict(items)

    def _save_json(self):
        """Save metrics to JSON file."""
        try:
            with open(self.json_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save JSON metrics: {e}")
