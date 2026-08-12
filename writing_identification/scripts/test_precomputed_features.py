#!/usr/bin/env python3
"""Test script to validate pre-computed features match on-the-fly extraction.

This script ensures that:
1. Pre-computed features exactly match on-the-fly extraction
2. Training produces identical loss values with both approaches
3. Pre-computation provides significant speedup
4. HDF5 files are correctly formatted and accessible

Usage:
    poetry run python scripts/test_precomputed_features.py
    poetry run python scripts/test_precomputed_features.py --full  # Test all samples
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging
import time
import tempfile
from typing import Optional
import numpy as np
import torch
import h5py

from config.config import config as base_config
from config.experiment_configs import get_experiment_config
from data.external_datasets import ExternalDatasetLoader, ExternalSample
from data.precomputed_dataset import PrecomputedAuthorshipDataset
from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
from features.email_patterns import EmailPatternExtractor
from models.siamese import SiameseNetwork, AuthorshipVerifier
from scripts.precompute_features import FeaturePrecomputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PrecomputedFeaturesValidator:
    """Validates pre-computed features against on-the-fly extraction."""

    def __init__(self, config_name: str = "external_contrastive"):
        """Initialize validator with configuration."""
        self.config_name = config_name
        self.experiment_config = get_experiment_config(config_name)
        self.config = {**base_config.model_dump(), **self.experiment_config}

        # Initialize extractors
        self.semantic_extractor = None
        self.style_extractor = None
        self.email_extractor = None

    def _initialize_extractors(self, sample_texts: list[str]) -> None:
        """Initialize and fit feature extractors."""
        logger.info("Initializing feature extractors...")

        # Semantic extractor (no fitting required)
        self.semantic_extractor = SemanticFeatureExtractor()

        # Email pattern extractor (no fitting required)
        self.email_extractor = EmailPatternExtractor()

        # Style extractor (requires fitting)
        max_features = self.config.get("data", {}).get("max_features", 1000)
        self.style_extractor = StyleFeatureExtractor(max_features=max_features)

        # Fit style extractor
        logger.info(f"Fitting style extractor on {len(sample_texts)} texts...")
        self.style_extractor.fit(sample_texts)

    def extract_features_on_the_fly(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract semantic and style features separately (matching pre-computation format).

        Returns:
            tuple: (semantic_features, style_features) where:
                - semantic_features: 768D semantic embeddings
                - style_features: Combined style+email features (~1207D)
        """
        # Semantic features
        semantic_features = self.semantic_extractor.extract_features(text)
        if isinstance(semantic_features, torch.Tensor):
            semantic_features = semantic_features.cpu().numpy()

        # Style features
        style_features = self.style_extractor.extract_features(text)
        if isinstance(style_features, torch.Tensor):
            style_features = style_features.cpu().numpy()

        # Email features
        email_features = self.email_extractor.extract_features(text)
        if isinstance(email_features, torch.Tensor):
            email_features = email_features.cpu().numpy()

        # Return separated: semantic, combined style+email (matching pre-computation)
        semantic_final = semantic_features.flatten().astype(np.float32)
        style_final = np.concatenate([style_features.flatten(), email_features.flatten()]).astype(np.float32)

        return semantic_final, style_final

    def test_feature_consistency(
        self, samples: list[ExternalSample], num_samples_to_test: int = 100
    ) -> tuple[bool, dict]:
        """Test that pre-computed features match on-the-fly extraction."""
        logger.info("=" * 80)
        logger.info("TEST 1: Feature Consistency")
        logger.info("=" * 80)

        results = {"passed": False, "num_tested": 0, "max_diff": 0.0, "mean_diff": 0.0, "failed_samples": []}

        # Create temporary HDF5 file for pre-computed features
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Pre-compute features for test samples
            logger.info(f"Pre-computing features for {num_samples_to_test} samples...")
            precomputer = FeaturePrecomputer(self.config, cache_dir=tmp_path.parent)

            test_samples = samples[:num_samples_to_test]

            # Initialize extractors with all texts
            all_texts = []
            for sample in test_samples:
                all_texts.extend([sample.text1, sample.text2])
            self._initialize_extractors(all_texts)

            # Pre-compute features
            h5_path = precomputer.precompute_dataset_features(
                dataset_name="test", samples=test_samples, split="test", batch_size=100
            )

            # Load pre-computed dataset
            precomputed_dataset = PrecomputedAuthorshipDataset(h5_path, validate_on_load=True)

            # Test each sample
            logger.info("Comparing pre-computed vs on-the-fly features...")
            differences = []

            for i in range(len(test_samples)):
                sample = test_samples[i]

                # Get pre-computed features (now properly separated)
                precomputed_data = precomputed_dataset[i]
                precomputed_semantic1 = precomputed_data["semantic_features1"].numpy()
                precomputed_style1 = precomputed_data["style_features1"].numpy()
                precomputed_semantic2 = precomputed_data["semantic_features2"].numpy()
                precomputed_style2 = precomputed_data["style_features2"].numpy()

                # Extract on-the-fly (now also separated)
                on_the_fly_semantic1, on_the_fly_style1 = self.extract_features_on_the_fly(sample.text1)
                on_the_fly_semantic2, on_the_fly_style2 = self.extract_features_on_the_fly(sample.text2)

                # Compare semantic and style features separately
                semantic_diff1 = np.abs(precomputed_semantic1 - on_the_fly_semantic1)
                style_diff1 = np.abs(precomputed_style1 - on_the_fly_style1)
                semantic_diff2 = np.abs(precomputed_semantic2 - on_the_fly_semantic2)
                style_diff2 = np.abs(precomputed_style2 - on_the_fly_style2)

                # Get max differences across all feature types
                max_semantic_diff1 = np.max(semantic_diff1)
                max_style_diff1 = np.max(style_diff1)
                max_semantic_diff2 = np.max(semantic_diff2)
                max_style_diff2 = np.max(style_diff2)
                max_diff = max(max_semantic_diff1, max_style_diff1, max_semantic_diff2, max_style_diff2)

                differences.append(max_diff)

                # Check if within tolerance
                rtol = 1e-5
                atol = 1e-6

                semantic_match1 = np.allclose(precomputed_semantic1, on_the_fly_semantic1, rtol=rtol, atol=atol)
                style_match1 = np.allclose(precomputed_style1, on_the_fly_style1, rtol=rtol, atol=atol)
                semantic_match2 = np.allclose(precomputed_semantic2, on_the_fly_semantic2, rtol=rtol, atol=atol)
                style_match2 = np.allclose(precomputed_style2, on_the_fly_style2, rtol=rtol, atol=atol)

                all_match = semantic_match1 and style_match1 and semantic_match2 and style_match2

                if not all_match:
                    results["failed_samples"].append(
                        {
                            "index": i,
                            "max_diff": float(max_diff),
                            "semantic1_match": semantic_match1,
                            "style1_match": style_match1,
                            "semantic2_match": semantic_match2,
                            "style2_match": style_match2,
                            "max_semantic_diff": float(max(max_semantic_diff1, max_semantic_diff2)),
                            "max_style_diff": float(max(max_style_diff1, max_style_diff2)),
                        }
                    )
                    logger.warning(
                        f"Sample {i}: max_diff={max_diff:.2e} (semantic: {max(max_semantic_diff1, max_semantic_diff2):.2e}, style: {max(max_style_diff1, max_style_diff2):.2e})"
                    )

                if (i + 1) % 20 == 0:
                    logger.info(f"Tested {i + 1}/{len(test_samples)} samples")

            # Calculate statistics
            results["num_tested"] = len(test_samples)
            results["max_diff"] = float(np.max(differences))
            results["mean_diff"] = float(np.mean(differences))
            results["passed"] = len(results["failed_samples"]) == 0

            # Log results
            if results["passed"]:
                logger.info(f"✅ All {results['num_tested']} samples passed consistency check")
                logger.info(f"   Max difference: {results['max_diff']:.2e}")
                logger.info(f"   Mean difference: {results['mean_diff']:.2e}")
            else:
                logger.error(f"❌ {len(results['failed_samples'])} samples failed consistency check")
                logger.error(f"   Max difference: {results['max_diff']:.2e}")
                for failed in results["failed_samples"][:5]:  # Show first 5 failures
                    logger.error(f"   Sample {failed['index']}: max_diff={failed['max_diff']:.2e}")

        finally:
            # Cleanup temporary files
            if tmp_path.exists():
                tmp_path.unlink()
            h5_path.unlink() if h5_path.exists() else None

        return results["passed"], results

    def test_training_consistency(
        self, samples: list[ExternalSample], num_batches: int = 10, batch_size: int = 32
    ) -> tuple[bool, dict]:
        """Test that training produces identical losses with both methods."""
        logger.info("=" * 80)
        logger.info("TEST 2: Training Consistency")
        logger.info("=" * 80)

        results = {
            "passed": False,
            "precomputed_losses": [],
            "on_the_fly_losses": [],
            "max_loss_diff": 0.0,
            "mean_loss_diff": 0.0,
        }

        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Prepare test samples
            num_samples = min(num_batches * batch_size, len(samples))
            test_samples = samples[:num_samples]

            # Initialize extractors
            all_texts = []
            for sample in test_samples:
                all_texts.extend([sample.text1, sample.text2])
            self._initialize_extractors(all_texts)

            # Pre-compute features
            logger.info(f"Pre-computing features for {num_samples} samples...")
            precomputer = FeaturePrecomputer(self.config, cache_dir=tmp_path.parent)
            h5_path = precomputer.precompute_dataset_features(
                dataset_name="test", samples=test_samples, split="test", batch_size=100
            )

            # Initialize model
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )

            # Get feature dimensions for fusion encoder
            sample_semantic = self.semantic_extractor.extract_features(test_samples[0].text1)
            sample_style = self.style_extractor.extract_features(test_samples[0].text1)
            sample_email = self.email_extractor.extract_features(test_samples[0].text1)

            semantic_dim = len(sample_semantic.flatten())
            style_dim = len(sample_style.flatten()) + len(sample_email.flatten())

            # Use fusion encoder (production architecture)
            siamese_network = SiameseNetwork(
                encoder_type="fusion",
                semantic_dim=semantic_dim,  # 768D semantic features
                style_dim=style_dim,  # 1207D combined style + email features
                hidden_dim=self.config.get("model", {}).get("hidden_dim", 512),
                output_dim=self.config.get("model", {}).get("final_embedding_dim", 256),
            ).to(device)

            model = AuthorshipVerifier(
                siamese_network=siamese_network,
                loss_type="contrastive",
                margin=self.config.get("training", {}).get("margin", 0.5),
            ).to(device)

            # Test with pre-computed features
            logger.info("Testing with pre-computed features...")
            precomputed_dataset = PrecomputedAuthorshipDataset(h5_path)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(test_samples))

                # Prepare batch
                batch_semantic1 = []
                batch_style1 = []
                batch_semantic2 = []
                batch_style2 = []
                batch_labels = []

                for i in range(batch_start, batch_end):
                    data = precomputed_dataset[i]
                    batch_semantic1.append(data["semantic_features1"])
                    batch_style1.append(data["style_features1"])
                    batch_semantic2.append(data["semantic_features2"])
                    batch_style2.append(data["style_features2"])
                    batch_labels.append(data["label"])

                # Stack tensors
                semantic_features1 = torch.stack(batch_semantic1).to(device)
                style_features1 = torch.stack(batch_style1).to(device)
                semantic_features2 = torch.stack(batch_semantic2).to(device)
                style_features2 = torch.stack(batch_style2).to(device)
                labels = torch.stack(batch_labels).to(device)

                # Forward pass
                with torch.no_grad():
                    output = model(semantic_features1, style_features1, semantic_features2, style_features2, labels)
                    results["precomputed_losses"].append(output["loss"].item())

            # Test with on-the-fly extraction
            logger.info("Testing with on-the-fly extraction...")

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(test_samples))

                # Extract features on-the-fly with proper separation
                batch_semantic1 = []
                batch_style1 = []
                batch_semantic2 = []
                batch_style2 = []
                batch_labels = []

                for i in range(batch_start, batch_end):
                    sample = test_samples[i]

                    # Text 1
                    sem1 = self.semantic_extractor.extract_features(sample.text1)
                    style1 = self.style_extractor.extract_features(sample.text1)
                    email1 = self.email_extractor.extract_features(sample.text1)

                    # Text 2
                    sem2 = self.semantic_extractor.extract_features(sample.text2)
                    style2 = self.style_extractor.extract_features(sample.text2)
                    email2 = self.email_extractor.extract_features(sample.text2)

                    # Combine style and email features (matching pre-computation)
                    combined_style1 = torch.cat([style1.flatten(), email1.flatten()])
                    combined_style2 = torch.cat([style2.flatten(), email2.flatten()])

                    batch_semantic1.append(sem1.flatten())
                    batch_style1.append(combined_style1)
                    batch_semantic2.append(sem2.flatten())
                    batch_style2.append(combined_style2)
                    batch_labels.append(torch.tensor(float(sample.label)))

                # Stack tensors
                semantic_features1 = torch.stack(batch_semantic1).to(device)
                style_features1 = torch.stack(batch_style1).to(device)
                semantic_features2 = torch.stack(batch_semantic2).to(device)
                style_features2 = torch.stack(batch_style2).to(device)
                labels = torch.stack(batch_labels).to(device)

                # Forward pass
                with torch.no_grad():
                    output = model(semantic_features1, style_features1, semantic_features2, style_features2, labels)
                    results["on_the_fly_losses"].append(output["loss"].item())

            # Compare losses
            loss_diffs = [abs(p - o) for p, o in zip(results["precomputed_losses"], results["on_the_fly_losses"])]
            results["max_loss_diff"] = max(loss_diffs)
            results["mean_loss_diff"] = np.mean(loss_diffs)

            # Check if losses match within tolerance
            results["passed"] = all(diff < 1e-4 for diff in loss_diffs)

            # Log results
            if results["passed"]:
                logger.info(f"✅ Training losses match for {num_batches} batches")
                logger.info(f"   Max loss difference: {results['max_loss_diff']:.2e}")
                logger.info(f"   Mean loss difference: {results['mean_loss_diff']:.2e}")
            else:
                logger.error("❌ Training losses do not match")
                logger.error(f"   Max loss difference: {results['max_loss_diff']:.2e}")
                for i, (p, o) in enumerate(zip(results["precomputed_losses"], results["on_the_fly_losses"])):
                    if abs(p - o) > 1e-4:
                        logger.error(f"   Batch {i}: precomputed={p:.6f}, on_the_fly={o:.6f}, diff={abs(p - o):.2e}")

        finally:
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
            if h5_path.exists():
                h5_path.unlink()

        return results["passed"], results

    def test_speedup(self, samples: list[ExternalSample], num_samples: int = 1000) -> tuple[bool, dict]:
        """Test speedup from pre-computation."""
        logger.info("=" * 80)
        logger.info("TEST 3: Performance Speedup")
        logger.info("=" * 80)

        results = {"passed": False, "precomputed_time": 0.0, "on_the_fly_time": 0.0, "speedup": 0.0, "num_samples": 0}

        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Prepare test samples
            test_samples = samples[: min(num_samples, len(samples))]
            results["num_samples"] = len(test_samples)

            # Initialize extractors
            all_texts = []
            for sample in test_samples[:500]:  # Use subset for fitting
                all_texts.extend([sample.text1, sample.text2])
            self._initialize_extractors(all_texts)

            # Pre-compute features
            logger.info(f"Pre-computing features for {len(test_samples)} samples...")
            precomputer = FeaturePrecomputer(self.config, cache_dir=tmp_path.parent)
            h5_path = precomputer.precompute_dataset_features(
                dataset_name="test", samples=test_samples, split="test", batch_size=100
            )

            # Benchmark pre-computed loading
            logger.info("Benchmarking pre-computed feature loading...")
            precomputed_dataset = PrecomputedAuthorshipDataset(h5_path)

            start_time = time.time()
            for i in range(len(test_samples)):
                _ = precomputed_dataset[i]
            results["precomputed_time"] = time.time() - start_time

            # Benchmark on-the-fly extraction (sample subset to avoid timeout)
            logger.info("Benchmarking on-the-fly extraction (sample)...")
            sample_size = min(100, len(test_samples))  # Test subset for speed

            start_time = time.time()
            for i in range(sample_size):
                sample = test_samples[i]
                _ = self.extract_features_on_the_fly(sample.text1)
                _ = self.extract_features_on_the_fly(sample.text2)
            on_the_fly_sample_time = time.time() - start_time

            # Extrapolate to full dataset
            results["on_the_fly_time"] = on_the_fly_sample_time * (len(test_samples) / sample_size)

            # Calculate speedup
            results["speedup"] = results["on_the_fly_time"] / results["precomputed_time"]
            results["passed"] = results["speedup"] > 10  # Expect at least 10x speedup

            # Log results
            logger.info(f"Pre-computed time: {results['precomputed_time']:.2f}s for {len(test_samples)} samples")
            logger.info(
                f"On-the-fly time (estimated): {results['on_the_fly_time']:.2f}s for {len(test_samples)} samples"
            )
            logger.info(f"Speedup: {results['speedup']:.1f}x")

            if results["passed"]:
                logger.info(f"✅ Achieved {results['speedup']:.1f}x speedup (>10x expected)")
            else:
                logger.warning(f"⚠️  Only achieved {results['speedup']:.1f}x speedup (<10x expected)")

        finally:
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
            if h5_path.exists():
                h5_path.unlink()

        return results["passed"], results

    def test_hdf5_integrity(self, h5_path: Optional[Path] = None) -> tuple[bool, dict]:
        """Test HDF5 file integrity and format."""
        logger.info("=" * 80)
        logger.info("TEST 4: HDF5 File Integrity")
        logger.info("=" * 80)

        results = {
            "passed": False,
            "file_exists": False,
            "required_datasets": [],
            "missing_datasets": [],
            "metadata_valid": False,
            "data_quality": {},
        }

        # Use provided path or create a test file
        if h5_path is None:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            # Create a small test file
            logger.info("Creating test HDF5 file...")
            external_loader = ExternalDatasetLoader(cache_dir="cache/external_test")
            samples = external_loader.load_huggingface_authorship_dataset(
                dataset_name="swan07/authorship-verification", split="validation", max_samples=50
            )

            precomputer = FeaturePrecomputer(self.config, cache_dir=tmp_path.parent)
            h5_path = precomputer.precompute_dataset_features(dataset_name="test", samples=samples, split="test")
            cleanup_needed = True
        else:
            cleanup_needed = False

        try:
            # Check file exists
            results["file_exists"] = h5_path.exists()
            if not results["file_exists"]:
                logger.error(f"❌ File does not exist: {h5_path}")
                return False, results

            # Open and validate file
            with h5py.File(h5_path, "r") as f:
                # Check required datasets
                required = [
                    "semantic_features_text1",
                    "semantic_features_text2",
                    "style_features_text1",
                    "style_features_text2",
                    "labels",
                    "metadata",
                ]
                results["required_datasets"] = required

                for dataset_name in required:
                    if dataset_name not in f:
                        results["missing_datasets"].append(dataset_name)

                if results["missing_datasets"]:
                    logger.error(f"❌ Missing datasets: {results['missing_datasets']}")
                    return False, results

                # Validate metadata
                metadata = f["metadata"]
                required_attrs = ["feature_version", "dataset_name", "split", "semantic_dim", "style_dim", "n_samples"]
                metadata_valid = all(attr in metadata.attrs for attr in required_attrs)
                results["metadata_valid"] = metadata_valid

                if not metadata_valid:
                    logger.error("❌ Metadata incomplete")
                    return False, results

                # Check data quality
                n_samples = metadata.attrs["n_samples"]
                semantic_dim = metadata.attrs["semantic_dim"]
                style_dim = metadata.attrs["style_dim"]

                # Validate shapes
                shape_checks = {
                    "semantic_text1_shape": f["semantic_features_text1"].shape == (n_samples, semantic_dim),
                    "semantic_text2_shape": f["semantic_features_text2"].shape == (n_samples, semantic_dim),
                    "style_text1_shape": f["style_features_text1"].shape == (n_samples, style_dim),
                    "style_text2_shape": f["style_features_text2"].shape == (n_samples, style_dim),
                    "labels_shape": f["labels"].shape == (n_samples,),
                }

                # Sample data quality check
                sample_semantic1 = f["semantic_features_text1"][: min(100, n_samples)]
                sample_semantic2 = f["semantic_features_text2"][: min(100, n_samples)]
                sample_style1 = f["style_features_text1"][: min(100, n_samples)]
                sample_style2 = f["style_features_text2"][: min(100, n_samples)]

                # Check for problematic values across all feature types
                all_samples = [sample_semantic1, sample_semantic2, sample_style1, sample_style2]
                results["data_quality"] = {
                    **shape_checks,
                    "has_nan": bool(any(np.any(np.isnan(sample)) for sample in all_samples)),
                    "has_inf": bool(any(np.any(np.isinf(sample)) for sample in all_samples)),
                    "all_zeros": bool(all(np.all(sample == 0) for sample in all_samples)),
                }

                # Check compression
                if hasattr(f["semantic_features_text1"], "compression"):
                    logger.info(f"Compression: {f['semantic_features_text1'].compression}")

                # File size
                file_size_mb = h5_path.stat().st_size / (1024 * 1024)
                logger.info(f"File size: {file_size_mb:.1f} MB")
                logger.info(f"Samples: {n_samples}")
                logger.info(f"Semantic dimension: {semantic_dim}")
                logger.info(f"Style dimension: {style_dim}")
                logger.info(f"Total feature dimension: {semantic_dim + style_dim}")

                # Determine if passed
                results["passed"] = (
                    not results["missing_datasets"]
                    and results["metadata_valid"]
                    and all(shape_checks.values())
                    and not results["data_quality"]["has_nan"]
                    and not results["data_quality"]["has_inf"]
                    and not results["data_quality"]["all_zeros"]
                )

                if results["passed"]:
                    logger.info("✅ HDF5 file integrity check passed")
                else:
                    logger.error("❌ HDF5 file integrity check failed")
                    for key, value in results["data_quality"].items():
                        if key.startswith("has_") or key == "all_zeros":
                            if value:
                                logger.error(f"   {key}: {value}")
                        else:
                            if not value:
                                logger.error(f"   {key}: {value}")

        finally:
            # Cleanup if needed
            if cleanup_needed and h5_path.exists():
                h5_path.unlink()

        return results["passed"], results


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test pre-computed features")
    parser.add_argument("--config", type=str, default="external_contrastive", help="Experiment configuration to test")
    parser.add_argument("--full", action="store_true", help="Run full test suite with more samples")
    parser.add_argument("--test-file", type=str, help="Path to existing HDF5 file to test")

    args = parser.parse_args()

    print("🧪 PRE-COMPUTED FEATURES VALIDATION")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Full test: {args.full}")
    print()

    # Initialize validator
    validator = PrecomputedFeaturesValidator(config_name=args.config)

    # Load test data
    if not args.test_file:
        logger.info("Loading external dataset samples...")
        external_loader = ExternalDatasetLoader(cache_dir="cache/external_test")

        # Determine sample size
        max_samples = 1000 if args.full else 100

        samples = external_loader.load_huggingface_authorship_dataset(
            dataset_name="swan07/authorship-verification",
            split="validation",
            max_samples=max_samples,
            min_text_length=100,
        )

        logger.info(f"Loaded {len(samples)} samples for testing")
    else:
        samples = None

    # Run tests
    all_passed = True
    test_results = {}

    # Test 1: Feature consistency
    if samples:
        num_consistency_samples = 500 if args.full else 50
        passed, results = validator.test_feature_consistency(samples, num_consistency_samples)
        test_results["feature_consistency"] = results
        all_passed = all_passed and passed
        print()

    # Test 2: Training consistency
    if samples:
        num_batches = 20 if args.full else 5
        passed, results = validator.test_training_consistency(samples, num_batches=num_batches)
        test_results["training_consistency"] = results
        all_passed = all_passed and passed
        print()

    # Test 3: Speedup
    if samples:
        num_speedup_samples = 5000 if args.full else 500
        passed, results = validator.test_speedup(samples, num_samples=num_speedup_samples)
        test_results["speedup"] = results
        all_passed = all_passed and passed
        print()

    # Test 4: HDF5 integrity
    if args.test_file:
        passed, results = validator.test_hdf5_integrity(Path(args.test_file))
    else:
        passed, results = validator.test_hdf5_integrity()
    test_results["hdf5_integrity"] = results
    all_passed = all_passed and passed

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, results in test_results.items():
        status = "✅ PASSED" if results.get("passed", False) else "❌ FAILED"
        print(f"{test_name}: {status}")

        # Print key metrics
        if test_name == "feature_consistency":
            print(f"  - Tested: {results.get('num_tested', 0)} samples")
            print(f"  - Max diff: {results.get('max_diff', 0):.2e}")
        elif test_name == "training_consistency":
            print(f"  - Max loss diff: {results.get('max_loss_diff', 0):.2e}")
        elif test_name == "speedup":
            print(f"  - Speedup: {results.get('speedup', 0):.1f}x")

    print()
    if all_passed:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
