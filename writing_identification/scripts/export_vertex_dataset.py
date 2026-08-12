"""Export training data for Vertex AI dataset."""

import asyncio
import json
import logging
from datetime import datetime

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.extract_training_data import AuthorshipDataExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def export_vertex_dataset():
    """Export training data to JSON format for Vertex AI."""

    # Extract data
    logger.info("Extracting training data...")
    extractor = AuthorshipDataExtractor()
    samples_by_author = await extractor.extract_training_samples()
    filtered_samples = extractor.filter_samples_for_training(samples_by_author)

    # Convert to simple format for Vertex AI
    export_data = []

    for author, samples in filtered_samples.items():
        logger.info(f"Processing {len(samples)} samples for {author}")

        for i, sample in enumerate(samples):
            export_data.append(
                {
                    "author": sample.author,
                    "content": sample.content,
                    "content_type": sample.content_type,
                    "char_count": sample.length,
                    "source": sample.source,
                    "title": sample.title,
                    "sample_id": f"{author}_{i:03d}",
                }
            )

    # Create output directory
    output_dir = Path("vertex_dataset")
    output_dir.mkdir(exist_ok=True)

    # Save main dataset
    dataset_file = output_dir / "authorship_dataset.jsonl"

    with open(dataset_file, "w") as f:
        for item in export_data:
            f.write(json.dumps(item) + "\n")

    # Save metadata
    metadata = {
        "dataset_name": "authorship_verification_dataset",
        "description": "Training data for authorship verification model",
        "created_at": datetime.now().isoformat(),
        "total_samples": len(export_data),
        "authors": list(filtered_samples.keys()),
        "samples_per_author": {author: len(samples) for author, samples in filtered_samples.items()},
        "content_types": list(set(item["content_type"] for item in export_data)),
        "schema": {
            "author": "string - Author identifier",
            "content": "string - Text content for training",
            "content_type": "string - Type of content (comment, issue, etc)",
            "char_count": "integer - Character count",
            "source": "string - Source of the content",
            "title": "string - Title of the content",
            "sample_id": "string - Unique identifier for the sample",
        },
    }

    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset exported to {dataset_file}")
    logger.info(f"Metadata saved to {metadata_file}")
    logger.info(f"Total samples: {len(export_data)}")
    logger.info(f"Authors: {len(filtered_samples)}")

    # Print upload instructions
    print("\n" + "=" * 60)
    print("VERTEX AI DATASET CREATION INSTRUCTIONS")
    print("=" * 60)
    print("1. Upload the dataset file to GCS:")
    print(f"   gcloud storage cp {dataset_file} gs://$GCS_BUCKET/datasets/")
    print(f"   gcloud storage cp {metadata_file} gs://$GCS_BUCKET/datasets/")
    print()
    print("2. Create Vertex AI dataset:")
    print("   gcloud ai datasets create \\")
    print("     --display-name='Authorship Verification Dataset' \\")
    print("     --schema='gs://${GCS_BUCKET}/datasets/dataset_metadata.json' \\")
    print("     --region=us-central1")
    print()
    print("3. Import data into dataset:")
    print("   # Use the Vertex AI console to import the JSONL file")
    print("   # Or use the gcloud command after creating the dataset")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(export_vertex_dataset())
