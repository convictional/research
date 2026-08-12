"""External dataset loaders for authorship verification."""

import logging
import pickle
import json
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from pydantic import BaseModel

try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExternalSample(BaseModel):
    """External dataset sample with text pair and label."""

    text1: str
    text2: str
    label: int  # 1 for same author, 0 for different author
    source: str = "external"


class ExternalDatasetLoader:
    """Load external authorship verification datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize external dataset loader.

        Args:
            cache_dir: Directory to cache processed datasets
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_huggingface_authorship_dataset(
        self,
        dataset_name: str = "swan07/authorship-verification",
        split: str = "train",
        max_samples: Optional[int] = None,
        min_text_length: int = 100,
    ) -> list[ExternalSample]:
        """
        Load HuggingFace authorship verification dataset.

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split (train/validation/test)
            max_samples: Maximum samples to load
            min_text_length: Minimum text length to keep

        Returns:
            list of ExternalSample objects
        """
        cache_file = self.cache_dir / f"huggingface_{dataset_name.replace('/', '_')}_{split}.pkl"

        # Load from cache if exists
        if cache_file.exists():
            logger.info(f"Loading cached dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        logger.info(f"Loading HuggingFace dataset: {dataset_name}, split: {split}")
        dataset = load_dataset(dataset_name, split=split)

        samples = []
        for i, row in enumerate(dataset):
            # Stop if we've reached max samples
            if max_samples and i >= max_samples:
                break

            text1 = row.get("text1", row.get("Text1", ""))
            text2 = row.get("text2", row.get("Text2", ""))
            # HuggingFace dataset uses 'same' field, not 'label'
            label = row.get("same", row.get("label", row.get("Label", 0)))

            # Skip if texts are too short
            if len(text1) < min_text_length or len(text2) < min_text_length:
                continue

            # Convert label to binary (some datasets use different encoding)
            if isinstance(label, str):
                label = 1 if label.lower() in ["same", "true", "1", "yes"] else 0

            samples.append(ExternalSample(text1=text1, text2=text2, label=int(label), source=f"hf_{dataset_name}"))

            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1} samples")

        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")

        # Cache the processed samples
        with open(cache_file, "wb") as f:
            pickle.dump(samples, f)
        logger.info(f"Cached dataset to {cache_file}")

        return samples

    def load_from_gcs_cache(self, metadata_file_path: str, max_samples: Optional[int] = None) -> list[ExternalSample]:
        """
        Load chunked dataset from GCS using metadata file.

        Args:
            metadata_file_path: GCS path to metadata JSON file
            max_samples: Maximum samples to load (None = all)

        Returns:
            list of ExternalSample objects from all chunks
        """
        if not GCS_AVAILABLE:
            raise RuntimeError("google-cloud-storage is required to load from GCS")

        logger.info(f"Loading chunked dataset using metadata: {metadata_file_path}")

        # Parse GCS path for metadata
        if not metadata_file_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {metadata_file_path}")

        path_parts = metadata_file_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        metadata_path = path_parts[1]

        # Load metadata
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        metadata_blob = bucket.blob(metadata_path)

        if not metadata_blob.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")

        metadata = json.loads(metadata_blob.download_as_text())
        chunks_info = metadata["chunks"]
        total_samples = metadata["total_samples"]

        logger.info(f"Found {len(chunks_info)} chunks with {total_samples:,} total samples")

        # Load chunks in order
        all_samples = []
        samples_loaded = 0

        for i, chunk_info in enumerate(chunks_info):
            if max_samples and samples_loaded >= max_samples:
                logger.info(f"Reached max_samples limit ({max_samples}), stopping at chunk {i}")
                break

            chunk_path = chunk_info["file"]
            logger.info(f"Loading chunk {i + 1}/{len(chunks_info)}: {chunk_path}")

            # Parse chunk path
            chunk_gcs_path = chunk_path[5:]  # Remove "gs://"
            chunk_path_parts = chunk_gcs_path.split("/", 1)
            chunk_bucket_name = chunk_path_parts[0]
            chunk_file_path = chunk_path_parts[1]

            # Download chunk
            chunk_blob = bucket.blob(chunk_file_path)
            chunk_json = chunk_blob.download_as_text()
            chunk_data = json.loads(chunk_json)

            # Convert to ExternalSample objects
            chunk_samples = []
            for item in chunk_data:
                if max_samples and samples_loaded >= max_samples:
                    break

                chunk_samples.append(
                    ExternalSample(
                        text1=item["text1"],
                        text2=item["text2"],
                        label=item["label"],
                        source=item.get("source", "gcs_cache_chunked"),
                    )
                )
                samples_loaded += 1

            all_samples.extend(chunk_samples)
            logger.info(f"   ✅ Loaded {len(chunk_samples)} samples from chunk {i + 1}")

        logger.info(f"✅ Total loaded from chunked GCS: {len(all_samples)} samples")
        return all_samples

    def get_dataset_statistics(self, samples: list[ExternalSample]) -> dict:
        """Get statistics about the dataset."""
        if not samples:
            return {}

        text1_lengths = [len(s.text1) for s in samples]
        text2_lengths = [len(s.text2) for s in samples]
        labels = [s.label for s in samples]

        stats = {
            "total_samples": len(samples),
            "positive_samples": sum(labels),
            "negative_samples": len(samples) - sum(labels),
            "positive_ratio": sum(labels) / len(samples),
            "text1_length": {
                "mean": sum(text1_lengths) / len(text1_lengths),
                "min": min(text1_lengths),
                "max": max(text1_lengths),
            },
            "text2_length": {
                "mean": sum(text2_lengths) / len(text2_lengths),
                "min": min(text2_lengths),
                "max": max(text2_lengths),
            },
            "sources": list(set(s.source for s in samples)),
        }

        return stats
