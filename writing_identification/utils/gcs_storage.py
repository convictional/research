"""Google Cloud Storage adapter for model artifacts and data."""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
import tempfile
import shutil

from google.cloud import storage
import torch

logger = logging.getLogger(__name__)


class GCSStorage:
    """Handles all GCS operations for training artifacts."""

    def __init__(self, bucket_name: str, project_id: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Initialize GCS storage handler.

        Args:
            bucket_name: GCS bucket name
            project_id: GCP project ID
            experiment_name: Name of the current experiment for organization
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.experiment_name = experiment_name or "default"

        # Initialize GCS client
        if project_id:
            self.client = storage.Client(project=project_id)
        else:
            self.client = storage.Client()

        self.bucket = self.client.bucket(bucket_name)

        # Define path structure
        self.base_path = f"writing-identification/{self.experiment_name}"

        logger.info(f"GCS Storage initialized: gs://{bucket_name}/{self.base_path}")

    def _get_blob_path(self, artifact_type: str, filename: str) -> str:
        """Generate consistent blob paths."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.base_path}/{artifact_type}/{timestamp}/{filename}"

    def upload_checkpoint(self, checkpoint: dict[str, Any], epoch: int, is_best: bool = False) -> str:
        """
        Upload model checkpoint to GCS.

        Args:
            checkpoint: PyTorch checkpoint dict
            epoch: Training epoch number
            is_best: Whether this is the best model so far

        Returns:
            GCS path of uploaded checkpoint
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            torch.save(checkpoint, tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Upload checkpoint
            filename = f"checkpoint_epoch_{epoch}.pt"
            blob_path = self._get_blob_path("models", filename)
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(tmp_path)

            logger.info(f"Checkpoint uploaded to gs://{self.bucket_name}/{blob_path}")

            # Also save as best model if specified
            if is_best:
                best_path = f"{self.base_path}/models/best_model.pt"
                best_blob = self.bucket.blob(best_path)
                best_blob.upload_from_filename(tmp_path)
                logger.info(f"Best model updated at gs://{self.bucket_name}/{best_path}")

            return f"gs://{self.bucket_name}/{blob_path}"

        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    def download_checkpoint(self, gcs_path: str, local_path: str) -> None:
        """
        Download checkpoint from GCS.

        Args:
            gcs_path: Full GCS path (gs://bucket/path) or blob path
            local_path: Local file path to save checkpoint
        """
        # Parse GCS path
        if gcs_path.startswith("gs://"):
            parts = gcs_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ""

            if bucket_name != self.bucket_name:
                # Different bucket, create new client
                bucket = self.client.bucket(bucket_name)
            else:
                bucket = self.bucket
        else:
            # Assume it's a blob path in current bucket
            blob_path = gcs_path
            bucket = self.bucket

        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        logger.info(f"Checkpoint downloaded from {gcs_path} to {local_path}")

    def upload_training_history(self, history: list, run_id: str) -> str:
        """
        Upload training history to GCS.

        Args:
            history: Training history as list of dicts
            run_id: Unique identifier for this training run

        Returns:
            GCS path of uploaded history
        """
        filename = f"training_history_{run_id}.json"
        blob_path = self._get_blob_path("results", filename)
        blob = self.bucket.blob(blob_path)

        # Upload as JSON
        blob.upload_from_string(json.dumps(history, indent=2), content_type="application/json")

        logger.info(f"Training history uploaded to gs://{self.bucket_name}/{blob_path}")
        return f"gs://{self.bucket_name}/{blob_path}"

    def upload_config(self, config: dict[str, Any], run_id: str) -> str:
        """
        Upload experiment configuration to GCS.

        Args:
            config: Configuration dictionary
            run_id: Unique identifier for this training run

        Returns:
            GCS path of uploaded config
        """
        filename = f"config_{run_id}.json"
        blob_path = self._get_blob_path("configs", filename)
        blob = self.bucket.blob(blob_path)

        # Upload as JSON
        blob.upload_from_string(json.dumps(config, indent=2), content_type="application/json")

        logger.info(f"Config uploaded to gs://{self.bucket_name}/{blob_path}")
        return f"gs://{self.bucket_name}/{blob_path}"

    def list_checkpoints(self) -> list:
        """
        list all available checkpoints in GCS.

        Returns:
            list of checkpoint GCS paths
        """
        prefix = f"{self.base_path}/models/"
        blobs = self.bucket.list_blobs(prefix=prefix)

        checkpoints = []
        for blob in blobs:
            if blob.name.endswith(".pt"):
                checkpoints.append(f"gs://{self.bucket_name}/{blob.name}")

        return sorted(checkpoints)

    def upload_feature_cache(self, cache_dir: str) -> str:
        """
        Upload feature cache directory to GCS.

        Args:
            cache_dir: Local directory containing cached features

        Returns:
            GCS prefix of uploaded cache
        """
        cache_prefix = f"{self.base_path}/feature_cache"

        # Upload all files in cache directory
        cache_path = Path(cache_dir)
        for file_path in cache_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(cache_path)
                blob_path = f"{cache_prefix}/{relative_path}"
                blob = self.bucket.blob(blob_path)
                blob.upload_from_filename(str(file_path))

        logger.info(f"Feature cache uploaded to gs://{self.bucket_name}/{cache_prefix}")
        return f"gs://{self.bucket_name}/{cache_prefix}"

    def download_feature_cache(self, local_dir: str) -> None:
        """
        Download feature cache from GCS to local directory.

        Args:
            local_dir: Local directory to save cache files
        """
        cache_prefix = f"{self.base_path}/feature_cache/"
        blobs = self.bucket.list_blobs(prefix=cache_prefix)

        # Create local directory
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # Download all cache files
        for blob in blobs:
            relative_path = blob.name[len(cache_prefix) :]
            local_file = local_path / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))

        logger.info(f"Feature cache downloaded to {local_dir}")

    def sync_local_to_gcs(self, local_dir: str, gcs_prefix: str) -> None:
        """
        Sync a local directory to GCS (one-way upload).

        Args:
            local_dir: Local directory to sync
            gcs_prefix: GCS prefix to sync to
        """
        local_path = Path(local_dir)

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                blob_path = f"{gcs_prefix}/{relative_path}"
                blob = self.bucket.blob(blob_path)
                blob.upload_from_filename(str(file_path))

        logger.info(f"Synced {local_dir} to gs://{self.bucket_name}/{gcs_prefix}")

    def upload_precomputed_features(self, h5_file_path: str, feature_version: str = "1.0") -> str:
        """
        Upload pre-computed features HDF5 file to GCS.

        Args:
            h5_file_path: Local path to HDF5 file
            feature_version: Version string for the features

        Returns:
            Full GCS path to uploaded file
        """
        h5_path = Path(h5_file_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")

        # Create versioned filename
        filename = h5_path.name
        if not filename.endswith(".h5"):
            filename = f"{filename}.h5"

        # Upload to versioned path
        blob_path = f"precomputed-features/v{feature_version}/{filename}"
        blob = self.bucket.blob(blob_path)

        logger.info(f"Uploading {h5_path.name} to GCS...")
        blob.upload_from_filename(str(h5_path))

        full_gcs_path = f"gs://{self.bucket_name}/{blob_path}"

        # Log file size for monitoring
        file_size_mb = h5_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Uploaded {h5_path.name} ({file_size_mb:.1f} MB) to {full_gcs_path}")

        return full_gcs_path

    def download_precomputed_features(self, gcs_path: str, local_path: str) -> None:
        """
        Download pre-computed features from GCS.

        Args:
            gcs_path: Full GCS path (gs://bucket/path) or blob path
            local_path: Local file path to save HDF5 file
        """
        # Parse GCS path
        if gcs_path.startswith("gs://"):
            parts = gcs_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ""

            if bucket_name != self.bucket_name:
                # Different bucket, create new client
                bucket = self.client.bucket(bucket_name)
            else:
                bucket = self.bucket
        else:
            # Assume it's a blob path in current bucket
            blob_path = gcs_path
            bucket = self.bucket

        # Ensure local directory exists
        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        logger.info(f"Downloading pre-computed features from {gcs_path}...")
        blob = bucket.blob(blob_path)

        if not blob.exists():
            raise FileNotFoundError(f"Pre-computed features not found in GCS: {gcs_path}")

        blob.download_to_filename(local_path)

        # Log download info
        file_size_mb = local_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded {local_path_obj.name} ({file_size_mb:.1f} MB) from GCS")

    def list_precomputed_features(self, feature_version: Optional[str] = None) -> list[str]:
        """
        list available pre-computed feature files in GCS.

        Args:
            feature_version: Optional version filter

        Returns:
            list of GCS paths to pre-computed feature files
        """
        prefix = "precomputed-features/"
        if feature_version:
            prefix += f"v{feature_version}/"

        blobs = self.bucket.list_blobs(prefix=prefix)
        gcs_paths = []

        for blob in blobs:
            if blob.name.endswith(".h5"):
                full_path = f"gs://{self.bucket_name}/{blob.name}"
                gcs_paths.append(full_path)

        logger.info(f"Found {len(gcs_paths)} pre-computed feature files in GCS")
        return gcs_paths

    def upload_precomputed_metadata(
        self, metadata: dict[str, Any], config_name: str, feature_version: str = "1.0"
    ) -> str:
        """
        Upload metadata about pre-computed features.

        Args:
            metadata: dictionary containing feature metadata
            config_name: Configuration name used for features
            feature_version: Feature version string

        Returns:
            GCS path to metadata file
        """
        # Create metadata filename
        filename = f"metadata_{config_name}_v{feature_version}.json"
        blob_path = f"precomputed-features/metadata/{filename}"

        # Upload metadata as JSON
        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(json.dumps(metadata, indent=2), content_type="application/json")

        full_gcs_path = f"gs://{self.bucket_name}/{blob_path}"
        logger.info(f"✓ Uploaded feature metadata to {full_gcs_path}")

        return full_gcs_path

    def download_precomputed_metadata(self, gcs_path: str) -> dict[str, Any]:
        """
        Download and parse pre-computed feature metadata.

        Args:
            gcs_path: GCS path to metadata JSON file

        Returns:
            dictionary containing metadata
        """
        # Parse GCS path
        if gcs_path.startswith("gs://"):
            parts = gcs_path[5:].split("/", 1)
            blob_path = parts[1] if len(parts) > 1 else ""
        else:
            blob_path = gcs_path

        blob = self.bucket.blob(blob_path)

        if not blob.exists():
            raise FileNotFoundError(f"Metadata file not found in GCS: {gcs_path}")

        # Download and parse JSON
        metadata_json = blob.download_as_text()
        metadata = json.loads(metadata_json)

        logger.info(f"✓ Downloaded feature metadata from {gcs_path}")
        return metadata


class HybridStorage:
    """
    Hybrid storage that can work both locally and with GCS.
    Useful for development and production compatibility.
    """

    def __init__(
        self,
        local_base_path: str = "./",
        gcs_bucket: Optional[str] = None,
        gcs_project: Optional[str] = None,
        experiment_name: Optional[str] = None,
        use_gcs: bool = True,
    ):
        """
        Initialize hybrid storage.

        Args:
            local_base_path: Base path for local storage
            gcs_bucket: GCS bucket name (optional)
            gcs_project: GCP project ID (optional)
            experiment_name: Experiment name for organization
            use_gcs: Whether to use GCS (if False, only local storage)
        """
        self.local_base_path = Path(local_base_path)
        self.use_gcs = use_gcs and gcs_bucket is not None

        if self.use_gcs:
            self.gcs = GCSStorage(gcs_bucket, gcs_project, experiment_name)
        else:
            self.gcs = None
            logger.info("Running in local-only mode (no GCS)")

    def save_checkpoint(self, checkpoint: dict[str, Any], epoch: int, is_best: bool = False) -> str:
        """
        Save checkpoint to both local and GCS.

        Returns:
            Path to saved checkpoint (GCS if enabled, otherwise local)
        """
        # Always save locally first
        local_dir = self.local_base_path / "models" / "checkpoints"
        local_dir.mkdir(parents=True, exist_ok=True)

        local_path = local_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, local_path)

        if is_best:
            best_path = local_dir / "best_model.pt"
            shutil.copy(local_path, best_path)

        # Upload to GCS if enabled
        if self.use_gcs:
            gcs_path = self.gcs.upload_checkpoint(checkpoint, epoch, is_best)
            return gcs_path

        return str(local_path)

    def save_training_history(self, history: list, run_id: str) -> str:
        """
        Save training history to both local and GCS.

        Returns:
            Path to saved history (GCS if enabled, otherwise local)
        """
        # Save locally
        local_dir = self.local_base_path / "results"
        local_dir.mkdir(parents=True, exist_ok=True)

        local_path = local_dir / f"training_history_{run_id}.json"
        with open(local_path, "w") as f:
            json.dump(history, f, indent=2)

        # Upload to GCS if enabled
        if self.use_gcs:
            gcs_path = self.gcs.upload_training_history(history, run_id)
            return gcs_path

        return str(local_path)
