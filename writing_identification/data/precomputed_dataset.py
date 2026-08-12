"""PyTorch dataset for pre-computed authorship verification features."""

import logging
import pickle
from pathlib import Path
from typing import Tuple, Any
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

logger = logging.getLogger(__name__)


class PrecomputedAuthorshipDataset(Dataset):
    """Dataset for pre-computed authorship verification features stored in HDF5."""

    def __init__(self, h5_file_path: str | Path, split: str = "train", validate_on_load: bool = True):
        """
        Initialize the pre-computed dataset.

        Args:
            h5_file_path: Path to HDF5 file with pre-computed features
            split: Dataset split ("train" or "validation")
            validate_on_load: Whether to validate the file on loading

        Raises:
            RuntimeError: If h5py is not available or file is invalid
        """

        self.h5_file_path = Path(h5_file_path)
        self.split = split

        if not self.h5_file_path.exists():
            raise FileNotFoundError(f"Pre-computed features file not found: {h5_file_path}")

        # Load and validate the file
        self._h5_file = None
        self._semantic_features_text1 = None
        self._semantic_features_text2 = None
        self._style_features_text1 = None
        self._style_features_text2 = None
        self._labels = None
        self._metadata = None

        self._load_file(validate=validate_on_load)

        logger.info(f"PrecomputedAuthorshipDataset loaded: {len(self)} samples from {self.h5_file_path}")
        logger.info(f"Feature dimension: {self.feature_dim}")
        logger.info(f"Split: {self.split}")

    def _load_file(self, validate: bool = True) -> None:
        """Load the HDF5 file and set up memory-mapped datasets."""
        try:
            # Open file with memory mapping for efficient access
            self._h5_file = h5py.File(self.h5_file_path, "r")

            # Load metadata
            if "metadata" not in self._h5_file:
                raise ValueError("HDF5 file missing metadata group")

            self._metadata = dict(self._h5_file["metadata"].attrs)

            # Check required datasets exist
            required_datasets = [
                "semantic_features_text1",
                "semantic_features_text2",
                "style_features_text1",
                "style_features_text2",
                "labels",
            ]
            for dataset_name in required_datasets:
                if dataset_name not in self._h5_file:
                    raise ValueError(f"HDF5 file missing required dataset: {dataset_name}")

            # Set up memory-mapped access to datasets
            self._semantic_features_text1 = self._h5_file["semantic_features_text1"]
            self._semantic_features_text2 = self._h5_file["semantic_features_text2"]
            self._style_features_text1 = self._h5_file["style_features_text1"]
            self._style_features_text2 = self._h5_file["style_features_text2"]
            self._labels = self._h5_file["labels"]

            # Validate if requested
            if validate:
                self._validate_dataset()

        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to load pre-computed features: {e}")

    def _validate_dataset(self) -> None:
        """Validate the integrity of the loaded dataset."""
        # Check metadata
        required_metadata = ["feature_version", "dataset_name", "split", "semantic_dim", "style_dim", "n_samples"]
        for key in required_metadata:
            if key not in self._metadata:
                raise ValueError(f"Missing metadata key: {key}")

        n_samples = self._metadata["n_samples"]
        semantic_dim = self._metadata["semantic_dim"]
        style_dim = self._metadata["style_dim"]

        # Check dataset shapes
        if self._semantic_features_text1.shape != (n_samples, semantic_dim):
            raise ValueError(
                f"Semantic text1 features shape mismatch: {self._semantic_features_text1.shape} != {(n_samples, semantic_dim)}"
            )

        if self._semantic_features_text2.shape != (n_samples, semantic_dim):
            raise ValueError(
                f"Semantic text2 features shape mismatch: {self._semantic_features_text2.shape} != {(n_samples, semantic_dim)}"
            )

        if self._style_features_text1.shape != (n_samples, style_dim):
            raise ValueError(
                f"Style text1 features shape mismatch: {self._style_features_text1.shape} != {(n_samples, style_dim)}"
            )

        if self._style_features_text2.shape != (n_samples, style_dim):
            raise ValueError(
                f"Style text2 features shape mismatch: {self._style_features_text2.shape} != {(n_samples, style_dim)}"
            )

        if self._labels.shape != (n_samples,):
            raise ValueError(f"Labels shape mismatch: {self._labels.shape} != {(n_samples,)}")

        # Check for valid label values
        unique_labels = np.unique(self._labels[: min(1000, len(self._labels))])  # Sample check
        if not all(label in [0, 1] for label in unique_labels):
            raise ValueError(f"Invalid label values found: {unique_labels}")

        logger.info("✓ Dataset validation passed")

    def _cleanup(self) -> None:
        """Clean up HDF5 file handle."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass  # File might already be closed
            self._h5_file = None
            self._semantic_features_text1 = None
            self._semantic_features_text2 = None
            self._style_features_text1 = None
            self._style_features_text2 = None
            self._labels = None

    def __del__(self):
        """Cleanup on deletion."""
        self._cleanup()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._labels is None:
            return 0
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a training sample by index.

        Args:
            idx: Sample index

        Returns:
            dictionary containing:
                - semantic_features1: Combined features for text1
                - style_features1: Dummy tensor (for compatibility)
                - semantic_features2: Combined features for text2
                - style_features2: Dummy tensor (for compatibility)
                - label: Ground truth label

        Note:
            Since features are pre-computed and combined, we return them as
            semantic_features and provide empty style_features for compatibility
            with existing training code.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Load features efficiently (HDF5 handles memory mapping)
        semantic_features1 = torch.from_numpy(self._semantic_features_text1[idx].astype(np.float32))
        semantic_features2 = torch.from_numpy(self._semantic_features_text2[idx].astype(np.float32))
        style_features1 = torch.from_numpy(self._style_features_text1[idx].astype(np.float32))
        style_features2 = torch.from_numpy(self._style_features_text2[idx].astype(np.float32))
        label = torch.tensor(float(self._labels[idx]), dtype=torch.float32)

        # Return properly separated semantic and style features for fusion encoder
        return {
            "semantic_features1": semantic_features1,
            "style_features1": style_features1,
            "semantic_features2": semantic_features2,
            "style_features2": style_features2,
            "label": label,
        }

    def get_sample_info(self, idx: int) -> dict[str, Any]:
        """
        Get metadata information about a sample.

        Args:
            idx: Sample index

        Returns:
            dictionary with sample information
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        sample_info = {
            "index": idx,
            "label": int(self._labels[idx]),
            "feature_dim": self.feature_dim,
            "feature_version": self.feature_version,
            "dataset_name": self.dataset_name,
        }

        # Add sample ID if available
        if "sample_ids" in self._h5_file:
            sample_info["sample_id"] = int(self._h5_file["sample_ids"][idx])

        return sample_info

    @property
    def semantic_dim(self) -> int:
        """Get the semantic feature dimension."""
        return self._metadata.get("semantic_dim", 0) if self._metadata else 0

    @property
    def style_dim(self) -> int:
        """Get the style feature dimension."""
        return self._metadata.get("style_dim", 0) if self._metadata else 0

    @property
    def feature_dim(self) -> int:
        """Get the total feature dimension (semantic + style)."""
        return self.semantic_dim + self.style_dim

    @property
    def feature_version(self) -> str:
        """Get the feature version."""
        return self._metadata.get("feature_version", "unknown") if self._metadata else "unknown"

    @property
    def dataset_name(self) -> str:
        """Get the dataset name."""
        return self._metadata.get("dataset_name", "unknown") if self._metadata else "unknown"

    @property
    def creation_date(self) -> str:
        """Get the dataset creation date."""
        return self._metadata.get("extraction_date", "unknown") if self._metadata else "unknown"

    def get_label_distribution(self) -> dict[int, int]:
        """Get the distribution of labels in the dataset."""
        if self._labels is None:
            return {}

        # Efficiently count labels
        unique_labels, counts = np.unique(self._labels[:], return_counts=True)
        return {int(label): int(count) for label, count in zip(unique_labels, counts)}

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive dataset statistics."""
        if self._features_text1 is None or self._labels is None:
            return {}

        # Sample a subset for statistics to avoid loading everything into memory
        sample_size = min(1000, len(self))
        sample_indices = np.random.choice(len(self), size=sample_size, replace=False)

        # Load sample features
        sample_semantic1 = self._semantic_features_text1[sample_indices]
        sample_semantic2 = self._semantic_features_text2[sample_indices]
        sample_style1 = self._style_features_text1[sample_indices]
        sample_style2 = self._style_features_text2[sample_indices]

        stats = {
            "n_samples": len(self),
            "semantic_dim": self.semantic_dim,
            "style_dim": self.style_dim,
            "feature_dim": self.feature_dim,
            "feature_version": self.feature_version,
            "dataset_name": self.dataset_name,
            "creation_date": self.creation_date,
            "label_distribution": self.get_label_distribution(),
            "feature_stats": {
                "semantic1_mean": float(np.mean(sample_semantic1)),
                "semantic1_std": float(np.std(sample_semantic1)),
                "semantic1_min": float(np.min(sample_semantic1)),
                "semantic1_max": float(np.max(sample_semantic1)),
                "semantic2_mean": float(np.mean(sample_semantic2)),
                "semantic2_std": float(np.std(sample_semantic2)),
                "semantic2_min": float(np.min(sample_semantic2)),
                "semantic2_max": float(np.max(sample_semantic2)),
                "style1_mean": float(np.mean(sample_style1)),
                "style1_std": float(np.std(sample_style1)),
                "style1_min": float(np.min(sample_style1)),
                "style1_max": float(np.max(sample_style1)),
                "style2_mean": float(np.mean(sample_style2)),
                "style2_std": float(np.std(sample_style2)),
                "style2_min": float(np.min(sample_style2)),
                "style2_max": float(np.max(sample_style2)),
            },
        }

        # Check for any problematic values
        all_features = [sample_semantic1, sample_semantic2, sample_style1, sample_style2]
        stats["data_quality"] = {
            "has_nan": bool(any(np.any(np.isnan(feat)) for feat in all_features)),
            "has_inf": bool(any(np.any(np.isinf(feat)) for feat in all_features)),
            "all_zeros": bool(all(np.all(feat == 0) for feat in all_features)),
        }

        return stats

    def create_subset(self, indices: list[int]) -> "PrecomputedAuthorshipDataset":
        """
        Create a subset of the dataset with specified indices.

        Args:
            indices: List of indices to include in subset

        Returns:
            New PrecomputedAuthorshipDataset with subset of data

        Note:
            This creates a view that references the same HDF5 file,
            so the original file must remain accessible.
        """
        # Create a new instance that shares the same file but with different indices
        subset = PrecomputedAuthorshipDataset.__new__(PrecomputedAuthorshipDataset)
        subset.h5_file_path = self.h5_file_path
        subset.split = self.split
        subset._h5_file = self._h5_file  # Share the same file handle
        subset._metadata = self._metadata

        # Create index-filtered views
        subset._semantic_features_text1 = self._semantic_features_text1[indices]
        subset._semantic_features_text2 = self._semantic_features_text2[indices]
        subset._style_features_text1 = self._style_features_text1[indices]
        subset._style_features_text2 = self._style_features_text2[indices]
        subset._labels = self._labels[indices]

        logger.info(f"Created subset with {len(indices)} samples from {len(self)} total samples")
        return subset


class GPUCachedPrecomputedDataset(Dataset):
    """GPU-cached dataset that loads all pre-computed features into VRAM at initialization."""

    def __init__(
        self,
        h5_file_path: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        validate_on_load: bool = True,
    ):
        """
        Initialize GPU-cached dataset.

        Args:
            h5_file_path: Path to HDF5 file with pre-computed features
            device: Device to cache data on (default: 'cuda')
            dtype: Data type for tensors (default: torch.float32)
            validate_on_load: Whether to validate dataset structure
        """
        self.h5_file_path = Path(h5_file_path)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        if not self.h5_file_path.exists():
            raise FileNotFoundError(f"Pre-computed features file not found: {h5_file_path}")

        logger.info(f"Loading dataset to {self.device}...")
        self._load_to_gpu(validate=validate_on_load)

        # Log memory usage
        memory_gb = self._calculate_memory_usage() / (1024**3)
        logger.info(f"✓ Loaded {len(self)} samples to {self.device} ({memory_gb:.2f} GB)")

    def _load_to_gpu(self, validate: bool = True) -> None:
        """Load entire dataset into GPU memory."""
        try:
            with h5py.File(self.h5_file_path, "r") as h5_file:
                # Validate structure if requested
                if validate:
                    self._validate_file_structure(h5_file)

                # Check GPU memory availability
                if self.device.type == "cuda":
                    self._check_gpu_memory(h5_file)

                # Load all data to GPU
                logger.info("Loading semantic features to GPU...")
                self.semantic_features_text1 = torch.from_numpy(
                    h5_file["semantic_features_text1"][:].astype(np.float32)
                ).to(device=self.device, dtype=self.dtype)

                self.semantic_features_text2 = torch.from_numpy(
                    h5_file["semantic_features_text2"][:].astype(np.float32)
                ).to(device=self.device, dtype=self.dtype)

                logger.info("Loading style features to GPU...")
                self.style_features_text1 = torch.from_numpy(h5_file["style_features_text1"][:].astype(np.float32)).to(
                    device=self.device, dtype=self.dtype
                )

                self.style_features_text2 = torch.from_numpy(h5_file["style_features_text2"][:].astype(np.float32)).to(
                    device=self.device, dtype=self.dtype
                )

                logger.info("Loading labels to GPU...")
                self.labels = torch.from_numpy(h5_file["labels"][:].astype(np.float32)).to(
                    device=self.device, dtype=self.dtype
                )

                # Store metadata
                self.metadata = dict(h5_file["metadata"].attrs)

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset to GPU: {e}")

    def _validate_file_structure(self, h5_file) -> None:
        """Validate HDF5 file structure."""
        required_datasets = [
            "semantic_features_text1",
            "semantic_features_text2",
            "style_features_text1",
            "style_features_text2",
            "labels",
        ]

        for dataset_name in required_datasets:
            if dataset_name not in h5_file:
                raise ValueError(f"HDF5 file missing required dataset: {dataset_name}")

        if "metadata" not in h5_file:
            raise ValueError("HDF5 file missing metadata group")

        # Check metadata fields
        metadata = dict(h5_file["metadata"].attrs)
        required_metadata = ["n_samples", "semantic_dim", "style_dim"]
        for key in required_metadata:
            if key not in metadata:
                raise ValueError(f"Missing metadata key: {key}")

    def _check_gpu_memory(self, h5_file) -> None:
        """Check if dataset fits in GPU memory."""
        if self.device.type != "cuda":
            return

        # Estimate memory needed
        n_samples = h5_file["metadata"].attrs["n_samples"]
        semantic_dim = h5_file["metadata"].attrs["semantic_dim"]
        style_dim = h5_file["metadata"].attrs["style_dim"]

        # Calculate bytes needed
        semantic_bytes = n_samples * semantic_dim * 2 * 4  # 2 texts, 4 bytes/float32
        style_bytes = n_samples * style_dim * 2 * 4  # 2 texts, 4 bytes/float32
        label_bytes = n_samples * 4  # 4 bytes/float32
        total_bytes = semantic_bytes + style_bytes + label_bytes

        # Check available GPU memory
        memory_available = torch.cuda.get_device_properties(0).total_memory
        memory_allocated = torch.cuda.memory_allocated()
        memory_free = memory_available - memory_allocated

        # Leave 10% buffer for other operations
        if total_bytes > memory_free * 0.9:
            raise RuntimeError(
                f"Dataset requires {total_bytes / (1024**3):.2f}GB "
                f"but only {memory_free / (1024**3):.2f}GB available on GPU. "
                f"Try reducing batch size or use CPU-based loading."
            )

        logger.info(
            f"GPU memory check: {total_bytes / (1024**3):.2f}GB needed, {memory_free / (1024**3):.2f}GB available"
        )

    def _calculate_memory_usage(self) -> int:
        """Calculate total GPU memory usage in bytes."""
        total_bytes = 0
        for tensor in [
            self.semantic_features_text1,
            self.semantic_features_text2,
            self.style_features_text1,
            self.style_features_text2,
            self.labels,
        ]:
            total_bytes += tensor.element_size() * tensor.nelement()
        return total_bytes

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a training sample by index. All data is already on GPU.

        Args:
            idx: Sample index

        Returns:
            dictionary containing GPU tensors for semantic features, style features, and label
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Direct GPU tensor indexing - extremely fast, no CPU involvement
        return {
            "semantic_features1": self.semantic_features_text1[idx],
            "style_features1": self.style_features_text1[idx],
            "semantic_features2": self.semantic_features_text2[idx],
            "style_features2": self.style_features_text2[idx],
            "label": self.labels[idx],
        }

    @property
    def semantic_dim(self) -> int:
        """Get semantic feature dimension."""
        return self.metadata.get("semantic_dim", 768)

    @property
    def style_dim(self) -> int:
        """Get style feature dimension."""
        return self.metadata.get("style_dim", 1207)

    @property
    def feature_version(self) -> str:
        """Get feature version."""
        return self.metadata.get("feature_version", "unknown")

    @property
    def dataset_name(self) -> str:
        """Get dataset name."""
        return self.metadata.get("dataset_name", "unknown")

    def get_label_distribution(self) -> dict[int, int]:
        """Get distribution of labels in the dataset."""
        # Use GPU tensors for fast computation
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        return {int(label.item()): int(count.item()) for label, count in zip(unique_labels, counts)}

    def get_memory_info(self) -> dict[str, Any]:
        """Get GPU memory usage information."""
        if self.device.type != "cuda":
            return {"device": str(self.device), "memory_usage_gb": 0}

        memory_usage = self._calculate_memory_usage()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()

        return {
            "device": str(self.device),
            "memory_usage_gb": memory_usage / (1024**3),
            "total_gpu_memory_gb": total_memory / (1024**3),
            "allocated_gpu_memory_gb": allocated_memory / (1024**3),
            "memory_utilization_percent": (memory_usage / total_memory) * 100,
        }


def create_precomputed_data_loaders(
    train_h5_path: str | Path,
    val_h5_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,  # Default to 0 for MPS compatibility
    pin_memory: bool = False,  # Default to False for MPS compatibility
    use_gpu_cache: bool = True,  # Enable GPU caching for performance
    device: str | torch.device = "cuda",  # Device for GPU caching
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for pre-computed features.

    Args:
        train_h5_path: Path to training features HDF5 file
        val_h5_path: Path to validation features HDF5 file
        batch_size: Batch size for data loading
        num_workers: Number of worker processes (0 for MPS compatibility)
        pin_memory: Whether to pin memory (False for MPS compatibility)
        use_gpu_cache: Whether to cache entire dataset in GPU memory
        device: Device for GPU caching (ignored if use_gpu_cache=False)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating data loaders for pre-computed features...")

    # Determine if GPU caching should be used
    if use_gpu_cache and torch.cuda.is_available():
        logger.info("Using GPU-cached datasets for maximum performance")
        try:
            # Create GPU-cached datasets
            train_dataset = GPUCachedPrecomputedDataset(train_h5_path, device=device)
            val_dataset = GPUCachedPrecomputedDataset(val_h5_path, device=device)

            # Log memory usage
            train_mem = train_dataset.get_memory_info()
            val_mem = val_dataset.get_memory_info()
            logger.info(
                f"GPU memory usage: Train={train_mem['memory_usage_gb']:.2f}GB, Val={val_mem['memory_usage_gb']:.2f}GB"
            )

            # Force optimal settings for GPU-cached data
            num_workers = 0  # No workers needed - data already on GPU
            pin_memory = False  # No need to pin memory

        except (RuntimeError, FileNotFoundError) as e:
            logger.warning(f"GPU caching failed ({e}), falling back to CPU-based loading")
            use_gpu_cache = False
    elif use_gpu_cache and not torch.cuda.is_available():
        logger.info("GPU not available, using CPU-based loading")
        use_gpu_cache = False

    # Fallback to original CPU-based datasets
    if not use_gpu_cache:
        logger.info("Using CPU-based datasets with disk I/O")
        train_dataset = PrecomputedAuthorshipDataset(train_h5_path, split="train")
        val_dataset = PrecomputedAuthorshipDataset(val_h5_path, split="validation")

    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Log label distributions
    train_dist = train_dataset.get_label_distribution()
    val_dist = val_dataset.get_label_distribution()
    logger.info(f"Training label distribution: {train_dist}")
    logger.info(f"Validation label distribution: {val_dist}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    cache_type = "GPU-cached" if use_gpu_cache else "CPU-based"
    logger.info(
        f"Data loaders created ({cache_type}): {len(train_loader)} training batches, "
        f"{len(val_loader)} validation batches"
    )
    logger.info(f"DataLoader settings: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")

    return train_loader, val_loader


def validate_precomputed_consistency(
    h5_path: Path, original_samples: list, feature_extractor_config: dict[str, Any], sample_size: int = 100
) -> bool:
    """
    Validate that pre-computed features match on-the-fly extraction.

    Args:
        h5_path: Path to pre-computed features file
        original_samples: List of original ExternalSample objects
        feature_extractor_config: Configuration used for feature extraction
        sample_size: Number of samples to validate

    Returns:
        True if features are consistent, False otherwise
    """
    logger.info("Validating consistency of pre-computed features against original extraction...")

    try:
        # Load pre-computed dataset
        dataset = PrecomputedAuthorshipDataset(h5_path, validate_on_load=True)

        # Initialize extractors with the same configuration
        from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
        from features.email_patterns import EmailPatternExtractor

        semantic_extractor = SemanticFeatureExtractor()
        style_extractor = StyleFeatureExtractor(
            max_features=feature_extractor_config.get("data", {}).get("max_features", 1000)
        )
        email_extractor = EmailPatternExtractor()

        # Fit style extractor on sample texts (same as pre-computation)
        sample_texts = []
        for sample in original_samples[:5000]:  # Use same subset size as pre-computation
            sample_texts.extend([sample.text1, sample.text2])
        style_extractor.fit(sample_texts)

        # Validate random samples
        sample_indices = np.random.choice(len(dataset), size=min(sample_size, len(dataset)), replace=False)

        for i, sample_idx in enumerate(sample_indices):
            if i % 20 == 0:
                logger.info(f"Validating sample {i + 1}/{len(sample_indices)}")

            # Get pre-computed features
            sample_data = dataset[sample_idx]
            precomputed_features1 = sample_data["semantic_features1"].numpy()

            # Extract features on-the-fly
            original_sample = original_samples[sample_idx]

            # Extract features for text1
            semantic_features1 = semantic_extractor.extract_features(original_sample.text1)
            style_features1 = style_extractor.extract_features(original_sample.text1)
            email_features1 = email_extractor.extract_features(original_sample.text1)

            # Convert to numpy and concatenate
            if isinstance(semantic_features1, torch.Tensor):
                semantic_features1 = semantic_features1.cpu().numpy()
            if isinstance(style_features1, torch.Tensor):
                style_features1 = style_features1.cpu().numpy()
            if isinstance(email_features1, torch.Tensor):
                email_features1 = email_features1.cpu().numpy()

            combined_features1 = np.concatenate(
                [semantic_features1.flatten(), style_features1.flatten(), email_features1.flatten()]
            ).astype(np.float32)

            # Check consistency
            if not np.allclose(precomputed_features1, combined_features1, rtol=1e-5, atol=1e-6):
                logger.error(f"Feature mismatch for sample {sample_idx}, text1")
                logger.error(f"Max difference: {np.max(np.abs(precomputed_features1 - combined_features1))}")
                return False

        logger.info("✓ Feature consistency validation passed")
        return True

    except Exception as e:
        logger.error(f"Consistency validation failed: {e}")
        return False


def load_extractors_from_h5(h5_path: Path) -> tuple:
    """Load fitted extractors from HDF5 companion pickle file.

    Args:
        h5_path: Path to HDF5 file with pre-computed features

    Returns:
        Tuple of (semantic_extractor, style_extractor, email_extractor) or (None, None, None) if not found
    """
    # Try companion pickle file
    extractor_path = h5_path.with_suffix(".extractors.pkl")
    if extractor_path.exists():
        logger.info(f"Loading fitted extractors from {extractor_path}")
        try:
            with open(extractor_path, "rb") as f:
                extractors = pickle.load(f)
                return (
                    extractors.get("semantic_extractor"),
                    extractors.get("style_extractor"),
                    extractors.get("email_extractor"),
                )
        except Exception as e:
            logger.error(f"Failed to load extractors from {extractor_path}: {e}")
    else:
        logger.warning(f"Extractor file not found: {extractor_path}")

    return None, None, None
