"""Cache external datasets to GCS in chunks for better reliability."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
import json
import asyncio
from datetime import datetime
import math

from google.cloud import storage
from data.external_datasets import ExternalDatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_dataset_chunked(
    bucket_name: str,
    dataset_samples: list,
    dataset_type: str,
    split: str,
    dataset_name: str = "swan07_authorship-verification",
    chunk_size: int = 50000,
):
    """Upload dataset in chunks for better reliability."""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    timestamp = datetime.now().strftime("%Y%m%d")

    # Upload in chunks
    num_chunks = math.ceil(len(dataset_samples) / chunk_size)
    chunk_files = []

    logger.info(f"📦 Uploading {len(dataset_samples)} samples in {num_chunks} chunks of {chunk_size}")

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(dataset_samples))
        chunk_samples = dataset_samples[start_idx:end_idx]

        logger.info(f"⬆️  Uploading chunk {chunk_idx + 1}/{num_chunks}: samples {start_idx}-{end_idx - 1}")

        # Convert chunk to serializable format
        chunk_data = []
        for sample in chunk_samples:
            chunk_data.append(
                {"text1": sample.text1, "text2": sample.text2, "label": sample.label, "source": sample.source}
            )

        # Upload chunk
        chunk_filename = f"datasets/external/{dataset_name}_{split}_{timestamp}_chunk_{chunk_idx:03d}.json"
        chunk_blob = bucket.blob(chunk_filename)

        chunk_blob.upload_from_string(json.dumps(chunk_data, indent=None), content_type="application/json")

        chunk_files.append(
            {
                "file": f"gs://{bucket_name}/{chunk_filename}",
                "start_idx": start_idx,
                "end_idx": end_idx - 1,
                "sample_count": len(chunk_data),
            }
        )

        logger.info(f"   ✅ Chunk {chunk_idx + 1} uploaded: {len(chunk_data)} samples")

    # Create metadata with chunk information
    metadata = {
        "dataset_name": dataset_name,
        "split": split,
        "dataset_type": dataset_type,
        "created_at": datetime.now().isoformat(),
        "total_samples": len(dataset_samples),
        "positive_samples": sum(1 for s in dataset_samples if s.label == 1),
        "negative_samples": sum(1 for s in dataset_samples if s.label == 0),
        "positive_ratio": sum(1 for s in dataset_samples if s.label == 1) / len(dataset_samples),
        "chunks": chunk_files,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "text1_length_stats": {
            "mean": sum(len(s.text1) for s in dataset_samples) / len(dataset_samples),
            "min": min(len(s.text1) for s in dataset_samples),
            "max": max(len(s.text1) for s in dataset_samples),
        },
        "text2_length_stats": {
            "mean": sum(len(s.text2) for s in dataset_samples) / len(dataset_samples),
            "min": min(len(s.text2) for s in dataset_samples),
            "max": max(len(s.text2) for s in dataset_samples),
        },
    }

    # Upload metadata
    metadata_filename = f"datasets/external/{dataset_name}_{split}_{timestamp}_metadata.json"
    metadata_blob = bucket.blob(metadata_filename)
    metadata_blob.upload_from_string(json.dumps(metadata, indent=2), content_type="application/json")

    logger.info(f"✅ Uploaded metadata: gs://{bucket_name}/{metadata_filename}")

    return chunk_files, f"gs://{bucket_name}/{metadata_filename}"


async def cache_huggingface_dataset_chunked(
    bucket_name: str,
    dataset_name: str = "swan07/authorship-verification",
    min_text_length: int = 100,
    chunk_size: int = 50000,
):
    """Download HuggingFace dataset and cache to GCS in chunks."""

    logger.info(f"🔄 Caching {dataset_name} dataset to GCS bucket: {bucket_name} (chunked)")

    # Initialize loader
    loader = ExternalDatasetLoader(cache_dir="cache/gcs_upload_temp")

    # Process each split
    splits_to_cache = ["train", "validation"]  # Skip test for now
    all_cached_info = {}

    for split in splits_to_cache:
        logger.info(f"\n📥 Processing {split} split...")

        try:
            # Load from cache or HuggingFace
            samples = loader.load_huggingface_authorship_dataset(
                dataset_name=dataset_name, split=split, max_samples=None, min_text_length=min_text_length
            )

            if not samples:
                logger.warning(f"⚠️  No samples found for {split} split")
                continue

            # Get statistics
            stats = loader.get_dataset_statistics(samples)
            logger.info(f"   📊 Loaded {stats['total_samples']} samples ({stats['positive_ratio']:.1%} positive)")

            # Upload in chunks
            chunk_files, metadata_file = upload_dataset_chunked(
                bucket_name=bucket_name,
                dataset_samples=samples,
                dataset_type="huggingface_authorship_verification",
                split=split,
                dataset_name=dataset_name.replace("/", "_"),
                chunk_size=chunk_size,
            )

            all_cached_info[split] = {
                "chunk_files": chunk_files,
                "metadata_file": metadata_file,
                "sample_count": len(samples),
                "positive_ratio": stats["positive_ratio"],
            }

        except Exception as e:
            logger.error(f"❌ Failed to process {split} split: {e}")
            continue

    # Create final index
    timestamp = datetime.now().strftime("%Y%m%d")
    index_filename = f"datasets/external/index_chunked_{timestamp}.json"

    index_data = {
        "dataset_name": dataset_name,
        "cached_at": datetime.now().isoformat(),
        "min_text_length": min_text_length,
        "chunk_size": chunk_size,
        "splits": all_cached_info,
        "total_samples": sum(info["sample_count"] for info in all_cached_info.values()),
        "usage_note": "This dataset is chunked. Use load_chunked_from_gcs() method to load all chunks.",
    }

    # Upload index
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    index_blob = bucket.blob(index_filename)
    index_blob.upload_from_string(json.dumps(index_data, indent=2), content_type="application/json")

    logger.info("\n✅ Chunked dataset caching complete!")
    logger.info(f"📋 Index file: gs://{bucket_name}/{index_filename}")
    logger.info(f"📊 Total samples cached: {index_data['total_samples']:,}")

    # Print chunk info
    for split, info in all_cached_info.items():
        logger.info(f"   {split}: {len(info['chunk_files'])} chunks, {info['sample_count']:,} samples")

    return index_data


async def main():
    """Main caching function."""
    import argparse

    parser = argparse.ArgumentParser(description="Cache external datasets to GCS in chunks")
    parser.add_argument("--bucket", default="your-gcs-bucket", help="GCS bucket name")
    parser.add_argument("--dataset", default="swan07/authorship-verification", help="HuggingFace dataset name")
    parser.add_argument("--min-text-length", type=int, default=100, help="Minimum text length threshold")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Samples per chunk for upload")

    args = parser.parse_args()

    await cache_huggingface_dataset_chunked(
        bucket_name=args.bucket,
        dataset_name=args.dataset,
        min_text_length=args.min_text_length,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    asyncio.run(main())
