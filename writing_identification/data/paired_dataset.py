"""PyTorch dataset for pre-paired authorship verification samples."""

import logging
from typing import Tuple
import torch
from torch.utils.data import Dataset

from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
from features.email_patterns import EmailPatternExtractor
from data.external_datasets import ExternalSample

logger = logging.getLogger(__name__)


class PairedAuthorshipDataset(Dataset):
    """Dataset for pre-paired authorship verification samples."""

    def __init__(
        self,
        samples: list[ExternalSample],
        semantic_extractor: SemanticFeatureExtractor,
        style_extractor: StyleFeatureExtractor,
        email_extractor: EmailPatternExtractor,
    ):
        """
        Initialize the paired dataset.

        Args:
            samples: list of ExternalSample objects with pre-paired texts
            semantic_extractor: Fitted semantic feature extractor
            style_extractor: Fitted style feature extractor
            email_extractor: Email pattern extractor
        """
        self.samples = samples
        self.semantic_extractor = semantic_extractor
        self.style_extractor = style_extractor
        self.email_extractor = email_extractor

        logger.info(f"Initialized PairedAuthorshipDataset with {len(samples)} pairs")

        # Log label distribution
        positive_count = sum(1 for s in samples if s.label == 1)
        negative_count = len(samples) - positive_count
        logger.info(
            f"Label distribution: {positive_count} positive, {negative_count} negative "
            f"({positive_count / len(samples):.2%} positive)"
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample pair and its features.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features1, features2, label)
        """
        sample = self.samples[idx]

        # Extract features for both texts
        features1 = self._extract_features(sample.text1)
        features2 = self._extract_features(sample.text2)
        label = torch.tensor(sample.label, dtype=torch.float32)

        return features1, features2, label

    def _extract_features(self, text: str) -> torch.Tensor:
        """Extract combined features from text."""
        # Semantic features (768D)
        semantic_features = self.semantic_extractor.extract_features(text)
        semantic_tensor = torch.tensor(semantic_features, dtype=torch.float32)

        # Style features (~1169D based on max_features=1000)
        style_features = self.style_extractor.extract_features(text)
        style_tensor = torch.tensor(style_features, dtype=torch.float32)

        # Email pattern features (38D)
        email_features = self.email_extractor.extract_features(text)
        email_tensor = torch.tensor(email_features, dtype=torch.float32)

        # Concatenate all features
        combined_features = torch.cat(
            [
                semantic_tensor,  # 768D
                style_tensor,  # ~1169D
                email_tensor,  # 38D
            ],
            dim=0,
        )

        return combined_features

    def get_sample_texts(self, idx: int) -> Tuple[str, str, int]:
        """
        Get the raw texts for a sample (useful for debugging).

        Args:
            idx: Sample index

        Returns:
            Tuple of (text1, text2, label)
        """
        sample = self.samples[idx]
        return sample.text1, sample.text2, sample.label

    def filter_by_label(self, target_label: int) -> "PairedAuthorshipDataset":
        """
        Create a new dataset with only samples of a specific label.

        Args:
            target_label: Label to filter by (0 or 1)

        Returns:
            New PairedAuthorshipDataset with filtered samples
        """
        filtered_samples = [s for s in self.samples if s.label == target_label]
        return PairedAuthorshipDataset(
            samples=filtered_samples,
            semantic_extractor=self.semantic_extractor,
            style_extractor=self.style_extractor,
            email_extractor=self.email_extractor,
        )

    def get_balanced_subset(self, max_samples: int) -> "PairedAuthorshipDataset":
        """
        Create a balanced subset with equal positive/negative samples.

        Args:
            max_samples: Maximum total samples (will be split 50/50)

        Returns:
            New balanced PairedAuthorshipDataset
        """
        positive_samples = [s for s in self.samples if s.label == 1]
        negative_samples = [s for s in self.samples if s.label == 0]

        samples_per_class = max_samples // 2

        # Take up to samples_per_class from each
        balanced_samples = []
        balanced_samples.extend(positive_samples[:samples_per_class])
        balanced_samples.extend(negative_samples[:samples_per_class])

        logger.info(
            f"Created balanced subset: {len(balanced_samples)} samples "
            f"({len([s for s in balanced_samples if s.label == 1])} positive, "
            f"{len([s for s in balanced_samples if s.label == 0])} negative)"
        )

        return PairedAuthorshipDataset(
            samples=balanced_samples,
            semantic_extractor=self.semantic_extractor,
            style_extractor=self.style_extractor,
            email_extractor=self.email_extractor,
        )
