#!/usr/bin/env python3
"""Pre-compute and cache features for authorship verification training.

This script extracts features for all samples in the external dataset and saves them
to HDF5 files for efficient training. Features are uploaded to GCS for use in Vertex AI.

Usage:
    # Local development - extract features and save locally
    poetry run python scripts/precompute_features.py --config external_contrastive

    # For Vertex AI - extract features and upload to GCS
    poetry run python scripts/precompute_features.py --config external_contrastive --upload-to-gcs

    # Resume from interruption
    poetry run python scripts/precompute_features.py --config external_contrastive --resume
"""

import argparse
import logging
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import json
import time
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import h5py
import torch
import numpy as np
from tqdm import tqdm

from config.config import config as base_config
from config.experiment_configs import get_experiment_config
from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
from features.email_patterns import EmailPatternExtractor
from data.external_datasets import ExternalDatasetLoader, ExternalSample
from utils.gcs_storage import GCSStorage

logger = logging.getLogger(__name__)


class FeaturePrecomputer:
    """Handles pre-computation and caching of features for training."""

    def __init__(self, config: dict[str, Any], cache_dir: str = "cache/precomputed", feature_version: str = "1.0"):
        """
        Initialize the feature pre-computer.

        Args:
            config: Experiment configuration
            cache_dir: Directory to save pre-computed features
            feature_version: Version string for features (for cache invalidation)
        """

        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_version = feature_version

        # Initialize extractors
        self.semantic_extractor = None
        self.style_extractor = None
        self.email_extractor = None

        # GCS storage (optional)
        self.gcs_storage = None
        if config.get("gcs_bucket"):
            self.gcs_storage = GCSStorage(
                bucket_name=config["gcs_bucket"],
                project_id=config.get("project_id"),
                experiment_name="precomputed-features",
            )

        logger.info(f"FeaturePrecomputer initialized with version {feature_version}")
        logger.info(f"Cache directory: {self.cache_dir}")

    def _get_cache_filename(self, dataset_name: str, split: str) -> str:
        """Generate simple cache filename without hash."""
        # Simple, predictable naming for easier management
        filename = f"features_{dataset_name}_{split}.h5"
        return filename

    def _get_version_dir(self) -> str:
        """Get version directory name (vYYYYMMDD)."""
        return f"v{datetime.now().strftime('%Y%m%d')}"

    def _initialize_extractors(self, all_texts: list[str]) -> None:
        """Initialize and fit feature extractors."""
        logger.info("Initializing feature extractors...")

        # Semantic extractor (no fitting required)
        self.semantic_extractor = SemanticFeatureExtractor()
        logger.info("✓ Semantic extractor initialized")

        # Email pattern extractor (no fitting required)
        self.email_extractor = EmailPatternExtractor()
        logger.info("✓ Email pattern extractor initialized")

        # Style extractor (requires fitting on all texts)
        logger.info("Fitting style feature extractor on training data...")
        max_features = self.config.get("data", {}).get("max_features", 1000)
        self.style_extractor = StyleFeatureExtractor(max_features=max_features)

        # Fit on sample of texts to avoid memory issues
        sample_size = min(50000, len(all_texts))  # Use up to 50k texts for fitting
        sample_texts = all_texts[:sample_size]
        logger.info(f"Fitting style extractor on {len(sample_texts)} texts...")

        start_time = time.time()
        self.style_extractor.fit(sample_texts)
        fit_time = time.time() - start_time

        logger.info(f"✓ Style extractor fitted in {fit_time:.1f}s")

        # Track what we fitted on for reporting
        self._fitted_texts_count = len(sample_texts)

    def save_extractors_only(self, dataset_name: str, split: str = "train") -> Path:
        """Save fitted extractors without pre-computing features.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split name

        Returns:
            Path to saved extractors pickle file
        """
        # Generate filename matching what full pre-computation would create
        h5_filename = self._get_cache_filename(dataset_name.replace("/", "_"), split)
        extractor_path = self.cache_dir / h5_filename.replace(".h5", ".extractors.pkl")

        logger.info(f"Saving fitted extractors to {extractor_path}")

        # Move semantic extractor model to CPU before saving to ensure cross-platform compatibility
        # This prevents issues when loading on different devices (MPS vs CUDA vs CPU)
        if hasattr(self.semantic_extractor, 'model') and hasattr(self.semantic_extractor.model, 'to'):
            logger.info("Moving semantic extractor model to CPU for device-agnostic saving")
            self.semantic_extractor.model = self.semantic_extractor.model.to('cpu')

        with open(extractor_path, "wb") as f:
            pickle.dump(
                {
                    "style_extractor": self.style_extractor,
                    "semantic_extractor": self.semantic_extractor,
                    "email_extractor": self.email_extractor,
                    "feature_version": self.feature_version,
                    "extraction_date": datetime.now().isoformat(),
                    "fitted_on_samples": getattr(self, "_fitted_texts_count", "unknown"),
                },
                f,
            )

        logger.info(f"✓ Extractors saved to {extractor_path}")
        return extractor_path

    def _extract_sample_features(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract semantic and style features from a single text sample.

        Returns:
            tuple: (semantic_features, style_features) where:
                - semantic_features: 768D semantic embeddings
                - style_features: Combined style+email features (~1207D)
        """
        # Semantic features (768D)
        semantic_features = self.semantic_extractor.extract_features(text)
        if isinstance(semantic_features, torch.Tensor):
            semantic_features = semantic_features.cpu().numpy()

        # Style features (~1169D)
        style_features = self.style_extractor.extract_features(text)
        if isinstance(style_features, torch.Tensor):
            style_features = style_features.cpu().numpy()

        # Email features (38D)
        email_features = self.email_extractor.extract_features(text)
        if isinstance(email_features, torch.Tensor):
            email_features = email_features.cpu().numpy()

        # Keep semantic separate, combine style and email
        semantic_final = semantic_features.flatten().astype(np.float32)
        style_final = np.concatenate([style_features.flatten(), email_features.flatten()]).astype(np.float32)

        return semantic_final, style_final

    def _save_progress(self, progress_file: Path, processed_count: int, total_count: int) -> None:
        """Save progress for resumption."""
        progress_data = {
            "processed_count": processed_count,
            "total_count": total_count,
            "timestamp": datetime.now().isoformat(),
            "feature_version": self.feature_version,
        }

        with open(progress_file, "w") as f:
            json.dump(progress_data, f)

    def _load_progress(self, progress_file: Path) -> Optional[dict[str, Any]]:
        """Load progress for resumption."""
        if not progress_file.exists():
            return None

        try:
            with open(progress_file, "r") as f:
                progress_data = json.load(f)

            # Validate version compatibility
            if progress_data.get("feature_version") != self.feature_version:
                logger.warning("Progress file has different feature version, starting fresh")
                return None

            return progress_data
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
            return None

    def precompute_dataset_features(
        self,
        dataset_name: str,
        samples: list[ExternalSample],
        split: str = "train",
        batch_size: int = 1000,
        resume: bool = False,
    ) -> Path:
        """
        Pre-compute features for a dataset split.

        Args:
            dataset_name: Name of the dataset
            samples: List of ExternalSample objects
            split: Dataset split name (train/validation)
            batch_size: Number of samples to process at once
            resume: Whether to resume from previous progress

        Returns:
            Path to the saved HDF5 file
        """
        logger.info(f"Pre-computing features for {dataset_name} {split} split")
        logger.info(f"Processing {len(samples)} samples")

        # Generate output filename
        h5_filename = self._get_cache_filename(dataset_name.replace("/", "_"), split)
        h5_path = self.cache_dir / h5_filename
        progress_path = self.cache_dir / f"{h5_filename}.progress"

        # Check for resumption
        start_idx = 0
        if resume and h5_path.exists():
            progress_data = self._load_progress(progress_path)
            if progress_data:
                start_idx = progress_data["processed_count"]
                logger.info(f"Resuming from sample {start_idx}")

        if start_idx == 0:
            # Initialize extractors if starting fresh
            all_texts = []
            for sample in samples:
                all_texts.extend([sample.text1, sample.text2])
            self._initialize_extractors(all_texts)

            # Get feature dimensions by processing one sample
            sample_semantic, sample_style = self._extract_sample_features(samples[0].text1)
            semantic_dim = len(sample_semantic)
            style_dim = len(sample_style)
            logger.info(f"Semantic feature dimension: {semantic_dim}")
            logger.info(f"Style feature dimension: {style_dim}")

            # Create HDF5 file
            with h5py.File(h5_path, "w") as f:
                # Create datasets with chunking and compression
                chunk_size = min(batch_size, len(samples))

                # Metadata
                metadata_group = f.create_group("metadata")
                metadata_group.attrs["feature_version"] = self.feature_version
                metadata_group.attrs["dataset_name"] = dataset_name
                metadata_group.attrs["split"] = split
                metadata_group.attrs["extraction_date"] = datetime.now().isoformat()
                metadata_group.attrs["semantic_dim"] = semantic_dim
                metadata_group.attrs["style_dim"] = style_dim
                metadata_group.attrs["n_samples"] = len(samples)

                # Create datasets for separate semantic and style features
                # Semantic features (text1 and text2)
                f.create_dataset(
                    "semantic_features_text1",
                    shape=(len(samples), semantic_dim),
                    dtype=np.float32,
                    chunks=(chunk_size, semantic_dim),
                    compression="gzip",
                    compression_opts=6,
                )
                f.create_dataset(
                    "semantic_features_text2",
                    shape=(len(samples), semantic_dim),
                    dtype=np.float32,
                    chunks=(chunk_size, semantic_dim),
                    compression="gzip",
                    compression_opts=6,
                )

                # Style features (text1 and text2)
                f.create_dataset(
                    "style_features_text1",
                    shape=(len(samples), style_dim),
                    dtype=np.float32,
                    chunks=(chunk_size, style_dim),
                    compression="gzip",
                    compression_opts=6,
                )
                f.create_dataset(
                    "style_features_text2",
                    shape=(len(samples), style_dim),
                    dtype=np.float32,
                    chunks=(chunk_size, style_dim),
                    compression="gzip",
                    compression_opts=6,
                )

                # Labels
                f.create_dataset(
                    "labels", shape=(len(samples),), dtype=np.int8, chunks=(chunk_size,), compression="gzip"
                )

                # Save sample IDs for debugging
                sample_ids = [i for i in range(len(samples))]
                f.create_dataset("sample_ids", data=sample_ids)
        else:
            # Resuming - re-initialize extractors
            logger.info("Re-initializing extractors for resumption...")
            all_texts = []
            for sample in samples[:1000]:  # Use subset for re-fitting
                all_texts.extend([sample.text1, sample.text2])
            self._initialize_extractors(all_texts)

        # Process samples in batches
        with h5py.File(h5_path, "a") as f:
            semantic_text1_dataset = f["semantic_features_text1"]
            semantic_text2_dataset = f["semantic_features_text2"]
            style_text1_dataset = f["style_features_text1"]
            style_text2_dataset = f["style_features_text2"]
            labels_dataset = f["labels"]

            # Progress bar
            progress_bar = tqdm(
                range(start_idx, len(samples)),
                desc=f"Extracting {split} features",
                initial=start_idx,
                total=len(samples),
            )

            batch_start = start_idx
            while batch_start < len(samples):
                batch_end = min(batch_start + batch_size, len(samples))
                batch_samples = samples[batch_start:batch_end]

                # Extract features for batch
                batch_semantic_text1 = []
                batch_style_text1 = []
                batch_semantic_text2 = []
                batch_style_text2 = []
                batch_labels = []

                for sample in batch_samples:
                    try:
                        semantic1, style1 = self._extract_sample_features(sample.text1)
                        semantic2, style2 = self._extract_sample_features(sample.text2)

                        batch_semantic_text1.append(semantic1)
                        batch_style_text1.append(style1)
                        batch_semantic_text2.append(semantic2)
                        batch_style_text2.append(style2)
                        batch_labels.append(sample.label)

                    except Exception as e:
                        logger.error(f"Failed to process sample {batch_start}: {e}")
                        # Use zero features as fallback
                        zero_semantic = np.zeros(semantic_dim, dtype=np.float32)
                        zero_style = np.zeros(style_dim, dtype=np.float32)
                        batch_semantic_text1.append(zero_semantic)
                        batch_style_text1.append(zero_style)
                        batch_semantic_text2.append(zero_semantic)
                        batch_style_text2.append(zero_style)
                        batch_labels.append(sample.label)

                # Save batch to HDF5
                batch_semantic_text1 = np.array(batch_semantic_text1)
                batch_style_text1 = np.array(batch_style_text1)
                batch_semantic_text2 = np.array(batch_semantic_text2)
                batch_style_text2 = np.array(batch_style_text2)
                batch_labels = np.array(batch_labels, dtype=np.int8)

                semantic_text1_dataset[batch_start:batch_end] = batch_semantic_text1
                style_text1_dataset[batch_start:batch_end] = batch_style_text1
                semantic_text2_dataset[batch_start:batch_end] = batch_semantic_text2
                style_text2_dataset[batch_start:batch_end] = batch_style_text2
                labels_dataset[batch_start:batch_end] = batch_labels

                # Update progress
                progress_bar.update(len(batch_samples))

                # Save progress periodically
                if (batch_end - start_idx) % (batch_size * 10) == 0:
                    self._save_progress(progress_path, batch_end, len(samples))

                batch_start = batch_end

            progress_bar.close()

        # Save fitted extractors as companion pickle file
        extractor_path = h5_path.with_suffix(".extractors.pkl")
        logger.info(f"Saving fitted extractors to {extractor_path}")

        # Move semantic extractor model to CPU before saving to ensure cross-platform compatibility
        # This prevents issues when loading on different devices (MPS vs CUDA vs CPU)
        if hasattr(self.semantic_extractor, 'model') and hasattr(self.semantic_extractor.model, 'to'):
            logger.info("Moving semantic extractor model to CPU for device-agnostic saving")
            self.semantic_extractor.model = self.semantic_extractor.model.to('cpu')

        with open(extractor_path, "wb") as f:
            pickle.dump(
                {
                    "style_extractor": self.style_extractor,
                    "semantic_extractor": self.semantic_extractor,
                    "email_extractor": self.email_extractor,
                    "feature_version": self.feature_version,
                    "extraction_date": datetime.now().isoformat(),
                },
                f,
            )
        logger.info(f"✓ Extractors saved to {extractor_path}")

        # Clean up progress file
        if progress_path.exists():
            progress_path.unlink()

        # Log statistics
        file_size_mb = h5_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Features saved to {h5_path}")
        logger.info(f"File size: {file_size_mb:.1f} MB")

        return h5_path

    def upload_to_gcs(self, h5_path: Path, version_dir: str = None) -> str:
        """Upload pre-computed features and extractors to GCS with versioned directories."""
        if not self.gcs_storage:
            raise RuntimeError("GCS storage not configured")

        # Use provided version directory or generate new one
        if version_dir is None:
            version_dir = self._get_version_dir()

        logger.info(f"Uploading {h5_path} to GCS in {version_dir}/...")

        # Upload HDF5 with versioned path
        gcs_path = f"precomputed-features/{version_dir}/{h5_path.name}"
        blob = self.gcs_storage.bucket.blob(gcs_path)
        blob.upload_from_filename(str(h5_path))

        full_gcs_path = f"gs://{self.gcs_storage.bucket_name}/{gcs_path}"
        logger.info(f"✓ Uploaded to {full_gcs_path}")

        # Upload companion pickle file if it exists
        extractor_path = h5_path.with_suffix(".extractors.pkl")
        if extractor_path.exists():
            logger.info(f"Uploading extractors {extractor_path.name} to GCS...")
            extractor_gcs_path = f"precomputed-features/{version_dir}/{extractor_path.name}"
            extractor_blob = self.gcs_storage.bucket.blob(extractor_gcs_path)
            extractor_blob.upload_from_filename(str(extractor_path))
            logger.info(f"✓ Uploaded extractors to gs://{self.gcs_storage.bucket_name}/{extractor_gcs_path}")

        return full_gcs_path

    def validate_features(self, h5_path: Path) -> bool:
        """Validate the integrity of pre-computed features."""
        logger.info(f"Validating features in {h5_path}")

        try:
            with h5py.File(h5_path, "r") as f:
                # Check metadata
                metadata = f["metadata"]
                feature_version = metadata.attrs.get("feature_version")
                semantic_dim = metadata.attrs.get("semantic_dim")
                style_dim = metadata.attrs.get("style_dim")
                n_samples = metadata.attrs.get("n_samples")

                logger.info(f"Feature version: {feature_version}")
                logger.info(f"Semantic dimension: {semantic_dim}")
                logger.info(f"Style dimension: {style_dim}")
                logger.info(f"Number of samples: {n_samples}")

                # Check dataset shapes
                semantic_text1_shape = f["semantic_features_text1"].shape
                semantic_text2_shape = f["semantic_features_text2"].shape
                style_text1_shape = f["style_features_text1"].shape
                style_text2_shape = f["style_features_text2"].shape
                labels_shape = f["labels"].shape

                logger.info(f"Semantic text1 shape: {semantic_text1_shape}")
                logger.info(f"Semantic text2 shape: {semantic_text2_shape}")
                logger.info(f"Style text1 shape: {style_text1_shape}")
                logger.info(f"Style text2 shape: {style_text2_shape}")
                logger.info(f"Labels shape: {labels_shape}")

                # Validate shapes
                if semantic_text1_shape[0] != n_samples or semantic_text2_shape[0] != n_samples:
                    logger.error("Semantic features sample count mismatch")
                    return False

                if style_text1_shape[0] != n_samples or style_text2_shape[0] != n_samples:
                    logger.error("Style features sample count mismatch")
                    return False

                if labels_shape[0] != n_samples:
                    logger.error("Labels shape mismatch: {labels_shape[0]} != {n_samples}")
                    return False

                if semantic_text1_shape[1] != semantic_dim or semantic_text2_shape[1] != semantic_dim:
                    logger.error("Semantic dimension mismatch")
                    return False

                if style_text1_shape[1] != style_dim or style_text2_shape[1] != style_dim:
                    logger.error("Style dimension mismatch")
                    return False

                # Check for NaN/Inf values in sample features
                sample_semantic1 = f["semantic_features_text1"][0]
                sample_semantic2 = f["semantic_features_text2"][0]
                sample_style1 = f["style_features_text1"][0]
                sample_style2 = f["style_features_text2"][0]

                for name, features in [
                    ("semantic1", sample_semantic1),
                    ("semantic2", sample_semantic2),
                    ("style1", sample_style1),
                    ("style2", sample_style2),
                ]:
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        logger.error(f"Found NaN/Inf values in {name} features")
                        return False

                logger.info("✓ All validation checks passed")
                return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


def load_reddit_data_for_precomputation(config: dict, task: str, max_samples: Optional[int] = None):
    """
    Load Reddit data and generate pairs/samples for precomputation.

    Args:
        config: Full configuration dictionary
        task: "classification" or "verification"
        max_samples: Maximum samples to generate (for testing)

    Returns:
        Tuple of (train_samples, val_samples) as ExternalSample objects
    """
    from data.reddit_dataset import load_reddit_dataset_for_training
    from data.reddit_pairs import RedditPairGenerator
    from data.external_datasets import ExternalSample

    logger.info(f"Loading Reddit data for {task} task...")

    # Load Reddit dataset
    from data.reddit_dataset import RedditDatasetConfig

    reddit_config_dict = config.get("external_data", {}).get("reddit_config", {})
    reddit_config = RedditDatasetConfig(**reddit_config_dict)
    train_texts_by_author, val_texts_by_author, author_metadata = load_reddit_dataset_for_training(reddit_config)

    logger.info(
        f"Loaded Reddit data: {len(train_texts_by_author)} train authors, {len(val_texts_by_author)} val authors"
    )

    if task == "classification":
        # Generate individual text samples for classification
        train_samples = []
        val_samples = []

        # Convert texts to ExternalSample format for classification
        for author, texts in train_texts_by_author.items():
            for text in texts:
                # For classification, we use the text twice (text1 == text2) with same author label
                sample = ExternalSample(
                    text1=text,
                    text2=text,  # Same text for classification
                    label=1,  # Same author (always true for classification samples)
                    source="reddit_classification",
                )
                train_samples.append(sample)
                if max_samples and len(train_samples) >= max_samples:
                    break
            if max_samples and len(train_samples) >= max_samples:
                break

        for author, texts in val_texts_by_author.items():
            for text in texts:
                sample = ExternalSample(text1=text, text2=text, label=1, source="reddit_classification")
                val_samples.append(sample)
                if max_samples and len(val_samples) >= max_samples:
                    break
            if max_samples and len(val_samples) >= max_samples:
                break

        logger.info(f"Generated {len(train_samples)} classification train samples, {len(val_samples)} val samples")

    else:  # verification task
        # Generate pairs using the Reddit pair generator
        reddit_pairs_config = config.get("external_data", {}).get("reddit_pairs_config", {})

        # Set defaults if not provided
        if not reddit_pairs_config:
            reddit_pairs_config = {
                "cache_dir": "cache/reddit_pairs",
                "positive_ratio": 0.5,
                "max_pairs_per_author": 100,
                "min_samples_per_author": 2,
                "seed": 42,
                "force_regenerate": False,
            }

        pair_generator = RedditPairGenerator(
            cache_dir=reddit_pairs_config["cache_dir"],
            positive_ratio=reddit_pairs_config["positive_ratio"],
            max_pairs_per_author=reddit_pairs_config["max_pairs_per_author"],
            min_samples_per_author=reddit_pairs_config["min_samples_per_author"],
            seed=reddit_pairs_config["seed"],
        )

        # Generate pairs
        train_pairs = pair_generator.generate_pairs(
            train_texts_by_author, "train", reddit_pairs_config.get("force_regenerate", False)
        )
        val_pairs = pair_generator.generate_pairs(
            val_texts_by_author, "val", reddit_pairs_config.get("force_regenerate", False)
        )

        # Convert pairs to ExternalSample format
        train_samples = []
        for text1, text2, label in train_pairs:
            sample = ExternalSample(text1=text1, text2=text2, label=label, source="reddit_verification")
            train_samples.append(sample)
            if max_samples and len(train_samples) >= max_samples:
                break

        val_samples = []
        for text1, text2, label in val_pairs:
            sample = ExternalSample(text1=text1, text2=text2, label=label, source="reddit_verification")
            val_samples.append(sample)
            if max_samples and len(val_samples) >= max_samples:
                break

        logger.info(f"Generated {len(train_samples)} verification train pairs, {len(val_samples)} val pairs")

    return train_samples, val_samples


def main():
    """Main function to run feature pre-computation."""
    parser = argparse.ArgumentParser(description="Pre-compute features for authorship verification")
    parser.add_argument("--config", type=str, required=True, help="Experiment configuration name")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "verification"],
        default="verification",
        help="Task type for Reddit data: classification (individual texts) or verification (pairs)",
    )
    parser.add_argument("--upload-to-gcs", action="store_true", help="Upload pre-computed features to GCS")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for feature extraction")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--extractors-only",
        action="store_true",
        help="Only fit and save extractors without pre-computing features (fast mode for testing)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"precompute_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ],
    )

    logger.info("=" * 80)
    logger.info("FEATURE PRE-COMPUTATION" if not args.extractors_only else "EXTRACTOR FITTING (FAST MODE)")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Extractors only mode: {args.extractors_only}")
    logger.info(f"Upload to GCS: {args.upload_to_gcs}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max samples: {args.max_samples}")

    try:
        # Load configuration
        experiment_config = get_experiment_config(args.config)

        # Merge configurations
        config = {
            **base_config.model_dump(),
            **experiment_config,
            "gcs_bucket": "your-gcs-bucket",
            "project_id": "${GCP_PROJECT}",
        }

        # Initialize pre-computer
        precomputer = FeaturePrecomputer(config)

        # Load dataset based on configuration
        external_config = experiment_config.get("external_data", {})
        if not external_config or not external_config.get("use_external_data"):
            logger.error("This script requires external data configuration")
            return

        dataset_name = external_config["dataset_name"]
        logger.info(f"Dataset: {dataset_name}, Task: {args.task}")

        if dataset_name == "reddit":
            logger.info("Loading Reddit data and generating pairs for precomputation...")
            # Load Reddit data and generate pairs
            train_samples, val_samples = load_reddit_data_for_precomputation(config, args.task, args.max_samples)
        else:
            logger.info(f"Loading external dataset: {dataset_name}")
            # Load external dataset (Swan07 or others)
            external_loader = ExternalDatasetLoader(cache_dir="cache/external")

            # Load training samples
            logger.info("Loading training samples...")
            if external_config.get("use_gcs_cache") and external_config.get("gcs_train_metadata"):
                logger.info(f"Loading training data from GCS cache: {external_config['gcs_train_metadata']}")
                train_samples = external_loader.load_from_gcs_cache(
                    metadata_file_path=external_config["gcs_train_metadata"], max_samples=args.max_samples
                )
            else:
                logger.info(f"Loading training data from HuggingFace dataset: {dataset_name}")
                train_samples = external_loader.load_huggingface_authorship_dataset(
                    dataset_name=dataset_name,
                    split=external_config.get("external_train_split", "train"),
                    max_samples=args.max_samples,
                    min_text_length=config.get("data", {}).get("min_text_length", 100),
                )

            # Load validation samples
            logger.info("Loading validation samples...")
            if external_config.get("use_gcs_cache") and external_config.get("gcs_val_metadata"):
                logger.info(f"Loading validation data from GCS cache: {external_config['gcs_val_metadata']}")
                val_samples = external_loader.load_from_gcs_cache(
                    metadata_file_path=external_config["gcs_val_metadata"], max_samples=args.max_samples
                )
            else:
                logger.info(f"Loading validation data from HuggingFace dataset: {dataset_name}")
                val_samples = external_loader.load_huggingface_authorship_dataset(
                    dataset_name=dataset_name,
                    split=external_config.get("external_val_split", "validation"),
                    max_samples=args.max_samples,
                    min_text_length=config.get("data", {}).get("min_text_length", 100),
                )

        # Check if we're in extractors-only mode
        if args.extractors_only:
            logger.info("=" * 80)
            logger.info("EXTRACTORS-ONLY MODE")
            logger.info("=" * 80)
            logger.info("Fitting extractors without full feature pre-computation...")

            # Gather all texts for fitting
            all_texts = []
            for sample in train_samples:
                all_texts.extend([sample.text1, sample.text2])
            logger.info(f"Collected {len(all_texts)} texts from {len(train_samples)} training samples")

            # Initialize and fit extractors
            precomputer._initialize_extractors(all_texts)

            # Save extractors for both train and validation (using same fitted extractors)
            train_extractor_path = precomputer.save_extractors_only(dataset_name=dataset_name, split="train")

            val_extractor_path = precomputer.save_extractors_only(dataset_name=dataset_name, split="validation")

            logger.info("=" * 80)
            logger.info("✓ Extractors saved successfully!")
            logger.info(f"Train extractors: {train_extractor_path}")
            logger.info(f"Val extractors: {val_extractor_path}")

            # Upload to GCS if requested
            if args.upload_to_gcs:
                logger.info("Uploading extractors to GCS...")

                # Get version directory for consistent upload
                version_dir = precomputer._get_version_dir()
                logger.info(f"Using version directory: {version_dir}")

                # Upload train extractors
                train_gcs_path = f"precomputed-features/{version_dir}/{train_extractor_path.name}"
                train_blob = precomputer.gcs_storage.bucket.blob(train_gcs_path)
                train_blob.upload_from_filename(str(train_extractor_path))
                train_full_gcs = f"gs://{precomputer.gcs_storage.bucket_name}/{train_gcs_path}"
                logger.info(f"✓ Uploaded train extractors to {train_full_gcs}")

                # Upload validation extractors
                val_gcs_path = f"precomputed-features/{version_dir}/{val_extractor_path.name}"
                val_blob = precomputer.gcs_storage.bucket.blob(val_gcs_path)
                val_blob.upload_from_filename(str(val_extractor_path))
                val_full_gcs = f"gs://{precomputer.gcs_storage.bucket_name}/{val_gcs_path}"
                logger.info(f"✓ Uploaded val extractors to {val_full_gcs}")

            logger.info("=" * 80)
            logger.info("You can now test baseline evaluation with these extractors")
            logger.info("Note: For production, run full pre-computation to ensure exact match")

            return  # Exit without doing full pre-computation

        # Pre-compute features for training set
        logger.info("Pre-computing training features...")
        train_h5_path = precomputer.precompute_dataset_features(
            dataset_name=dataset_name,
            samples=train_samples,
            split="train",
            batch_size=args.batch_size,
            resume=args.resume,
        )

        # Validate training features
        if not precomputer.validate_features(train_h5_path):
            logger.error("Training feature validation failed")
            return

        # Pre-compute features for validation set
        logger.info("Pre-computing validation features...")
        val_h5_path = precomputer.precompute_dataset_features(
            dataset_name=dataset_name,
            samples=val_samples,
            split="validation",
            batch_size=args.batch_size,
            resume=args.resume,
        )

        # Validate validation features
        if not precomputer.validate_features(val_h5_path):
            logger.error("Validation feature validation failed")
            return

        # Upload to GCS if requested
        if args.upload_to_gcs:
            logger.info("Uploading to GCS...")
            # Use same version directory for both train and validation
            version_dir = precomputer._get_version_dir()
            logger.info(f"Using version directory: {version_dir}")
            train_gcs_path = precomputer.upload_to_gcs(train_h5_path, version_dir=version_dir)
            val_gcs_path = precomputer.upload_to_gcs(val_h5_path, version_dir=version_dir)

            # Save GCS paths for easy reference
            gcs_paths = {
                "train_features": train_gcs_path,
                "validation_features": val_gcs_path,
                "feature_version": precomputer.feature_version,
                "config": args.config,
                "creation_date": datetime.now().isoformat(),
            }

            gcs_info_path = precomputer.cache_dir / f"gcs_paths_{args.config}.json"
            with open(gcs_info_path, "w") as f:
                json.dump(gcs_paths, f, indent=2)

            logger.info(f"GCS paths saved to {gcs_info_path}")
            logger.info(f"Training features: {train_gcs_path}")
            logger.info(f"Validation features: {val_gcs_path}")

        logger.info("=" * 80)
        logger.info("FEATURE PRE-COMPUTATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Training features: {train_h5_path}")
        logger.info(f"Validation features: {val_h5_path}")

        if args.upload_to_gcs:
            logger.info(f"Training GCS: {train_gcs_path}")
            logger.info(f"Validation GCS: {val_gcs_path}")

    except Exception as e:
        logger.error(f"Feature pre-computation failed: {e}")
        raise


if __name__ == "__main__":
    main()
