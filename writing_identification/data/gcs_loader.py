"""Load training data from GCS dataset file."""

import json
import logging
from google.cloud import storage
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GCSContentSample(BaseModel):
    """Content sample loaded from GCS dataset."""

    author: str
    content: str
    content_type: str
    char_count: int
    source: str
    title: str
    sample_id: str


class GCSDataLoader:
    """Load training data from GCS dataset file."""

    def __init__(self, gcs_bucket: str, dataset_file: str = "datasets/authorship_dataset.jsonl"):
        """
        Initialize GCS data loader.

        Args:
            gcs_bucket: GCS bucket name
            dataset_file: Path to dataset file within bucket
        """
        self.gcs_bucket = gcs_bucket
        self.dataset_file = dataset_file
        self.client = storage.Client()

    def load_training_data(self) -> dict[str, list[str]]:
        """
        Load training data from GCS and return as texts by author.

        Returns:
            dictionary mapping author names to lists of text content
        """
        logger.info(f"Loading dataset from gs://{self.gcs_bucket}/{self.dataset_file}")

        # Download the dataset file
        bucket = self.client.bucket(self.gcs_bucket)
        blob = bucket.blob(self.dataset_file)

        if not blob.exists():
            raise FileNotFoundError(f"Dataset file not found: gs://{self.gcs_bucket}/{self.dataset_file}")

        # Read JSONL content
        dataset_content = blob.download_as_text()

        # Parse JSONL
        samples = []
        for line_num, line in enumerate(dataset_content.strip().split("\n"), 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    sample = GCSContentSample(**data)
                    samples.append(sample)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Skipping invalid line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(samples)} samples from GCS")

        # Group by author
        texts_by_author = {}
        for sample in samples:
            if sample.author not in texts_by_author:
                texts_by_author[sample.author] = []
            texts_by_author[sample.author].append(sample.content)

        # Log statistics
        for author, texts in texts_by_author.items():
            logger.info(f"  {author}: {len(texts)} samples")

        return texts_by_author

    def get_sample_statistics(self) -> dict[str, any]:
        """
        Get statistics about the dataset.

        Returns:
            dictionary with dataset statistics
        """
        texts_by_author = self.load_training_data()

        total_samples = sum(len(texts) for texts in texts_by_author.values())

        stats = {
            "total_samples": total_samples,
            "num_authors": len(texts_by_author),
            "authors": list(texts_by_author.keys()),
            "samples_per_author": {author: len(texts) for author, texts in texts_by_author.items()},
            "avg_samples_per_author": total_samples / len(texts_by_author) if texts_by_author else 0,
            "content_lengths": {author: [len(text) for text in texts] for author, texts in texts_by_author.items()},
        }

        return stats
