"""Dataset for classification-based authorship training."""

import random
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset

from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
from features.email_patterns import EmailPatternExtractor


class AuthorClassificationDataset(Dataset):
    """Dataset for author classification training (single texts with author labels)."""

    def __init__(
        self,
        texts_by_author: dict[str, list[str]],
        semantic_extractor: SemanticFeatureExtractor,
        style_extractor: StyleFeatureExtractor,
        email_extractor: EmailPatternExtractor,
        min_samples_per_author: int = 2,
    ):
        """
        Initialize classification dataset.

        Args:
            texts_by_author: dictionary mapping author_id -> list of texts
            semantic_extractor: Fitted semantic feature extractor
            style_extractor: Fitted style feature extractor
            email_extractor: Email pattern extractor
            min_samples_per_author: Minimum samples per author to include
        """
        self.semantic_extractor = semantic_extractor
        self.style_extractor = style_extractor
        self.email_extractor = email_extractor

        # Filter authors with sufficient samples
        self.texts_by_author = {
            author: texts for author, texts in texts_by_author.items() if len(texts) >= min_samples_per_author
        }

        # Create author to index mapping
        self.authors = sorted(list(self.texts_by_author.keys()))
        self.author_to_idx = {author: idx for idx, author in enumerate(self.authors)}
        self.num_authors = len(self.authors)

        # Create flat list of (text, author_idx) samples
        self.samples = []
        for author in self.authors:
            author_idx = self.author_to_idx[author]
            for text in self.texts_by_author[author]:
                self.samples.append((text, author_idx))

        # Shuffle samples
        random.shuffle(self.samples)

        print(f"Classification dataset: {len(self.samples)} samples from {self.num_authors} authors")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a classification sample."""
        text, author_idx = self.samples[idx]

        # Extract features
        features = self._extract_all_features(text)

        return {
            "semantic_features": features["semantic"],
            "style_features": features["style"],
            "author_id": torch.tensor(author_idx, dtype=torch.long),
        }

    def _extract_all_features(self, text: str) -> dict[str, torch.Tensor]:
        """Extract all features from a text."""
        # Semantic features
        semantic_features = self.semantic_extractor.extract_features(text)

        # Stylometric features
        style_features = self.style_extractor.extract_features(text)

        # Email-specific features
        email_features = self.email_extractor.extract_features(text)

        # Combine style and email features
        combined_style = torch.cat([style_features, email_features])

        return {"semantic": semantic_features, "style": combined_style}


class HybridAuthorDataset(Dataset):
    """Dataset that can provide both classification and verification samples."""

    def __init__(
        self,
        texts_by_author: dict[str, list[str]],
        semantic_extractor: SemanticFeatureExtractor,
        style_extractor: StyleFeatureExtractor,
        email_extractor: EmailPatternExtractor,
        mode: str = "classification",  # "classification" or "verification"
        min_samples_per_author: int = 2,
        positive_ratio: float = 0.5,
        verification_samples_per_epoch: int = None,
    ):
        """
        Initialize hybrid dataset.

        Args:
            texts_by_author: dictionary mapping author_id -> list of texts
            semantic_extractor: Fitted semantic feature extractor
            style_extractor: Fitted style feature extractor
            email_extractor: Email pattern extractor
            mode: Dataset mode ("classification" or "verification")
            min_samples_per_author: Minimum samples per author
            positive_ratio: Ratio of positive pairs for verification mode
            verification_samples_per_epoch: Number of verification samples per epoch
        """
        self.semantic_extractor = semantic_extractor
        self.style_extractor = style_extractor
        self.email_extractor = email_extractor
        self.mode = mode
        self.positive_ratio = positive_ratio

        # Filter authors with sufficient samples
        self.texts_by_author = {
            author: texts for author, texts in texts_by_author.items() if len(texts) >= min_samples_per_author
        }

        # Create author mappings
        self.authors = sorted(list(self.texts_by_author.keys()))
        self.author_to_idx = {author: idx for idx, author in enumerate(self.authors)}
        self.num_authors = len(self.authors)

        if mode == "classification":
            self._setup_classification()
        elif mode == "verification":
            self._setup_verification(verification_samples_per_epoch)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _setup_classification(self):
        """Setup for classification mode."""
        self.samples = []
        for author in self.authors:
            author_idx = self.author_to_idx[author]
            for text in self.texts_by_author[author]:
                self.samples.append((text, author_idx))

        random.shuffle(self.samples)

    def _setup_verification(self, samples_per_epoch: Optional[int]):
        """Setup for verification mode."""
        self.samples_per_epoch = samples_per_epoch or self._estimate_verification_samples()
        self.pairs = self._generate_verification_pairs()

    def _estimate_verification_samples(self) -> int:
        """Estimate reasonable number of verification samples per epoch."""
        total_texts = sum(len(texts) for texts in self.texts_by_author.values())
        # Aim for 2-3x the number of single texts
        return min(total_texts * 3, 50000)

    def _generate_verification_pairs(self) -> list[Tuple[str, str, int, str, str]]:
        """Generate verification pairs (text1, text2, label, author1, author2)."""
        pairs = []
        positive_target = int(self.samples_per_epoch * self.positive_ratio)
        negative_target = self.samples_per_epoch - positive_target

        # Generate positive pairs
        positive_count = 0
        for author in self.authors:
            texts = self.texts_by_author[author]
            if len(texts) < 2:
                continue

            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if positive_count >= positive_target:
                        break
                    pairs.append((texts[i], texts[j], 1, author, author))
                    positive_count += 1
                if positive_count >= positive_target:
                    break

        # Generate negative pairs
        negative_count = 0
        while negative_count < negative_target:
            author1, author2 = random.sample(self.authors, 2)
            text1 = random.choice(self.texts_by_author[author1])
            text2 = random.choice(self.texts_by_author[author2])
            pairs.append((text1, text2, 0, author1, author2))
            negative_count += 1

        random.shuffle(pairs)
        return pairs

    def switch_mode(self, mode: str, verification_samples_per_epoch: int = None):
        """Switch between classification and verification modes."""
        self.mode = mode
        if mode == "classification":
            self._setup_classification()
        elif mode == "verification":
            self._setup_verification(verification_samples_per_epoch)

    def __len__(self) -> int:
        if self.mode == "classification":
            return len(self.samples)
        else:
            return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample based on current mode."""
        if self.mode == "classification":
            return self._get_classification_sample(idx)
        else:
            return self._get_verification_sample(idx)

    def _get_classification_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a classification sample."""
        text, author_idx = self.samples[idx]
        features = self._extract_all_features(text)

        return {
            "semantic_features": features["semantic"],
            "style_features": features["style"],
            "author_id": torch.tensor(author_idx, dtype=torch.long),
        }

    def _get_verification_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a verification sample."""
        text1, text2, label, author1, author2 = self.pairs[idx]

        features1 = self._extract_all_features(text1)
        features2 = self._extract_all_features(text2)

        return {
            "semantic_features1": features1["semantic"],
            "style_features1": features1["style"],
            "semantic_features2": features2["semantic"],
            "style_features2": features2["style"],
            "label": torch.tensor(label, dtype=torch.float32),
            "author_id1": torch.tensor(self.author_to_idx[author1], dtype=torch.long),
            "author_id2": torch.tensor(self.author_to_idx[author2], dtype=torch.long),
        }

    def _extract_all_features(self, text: str) -> dict[str, torch.Tensor]:
        """Extract all features from a text."""
        semantic_features = self.semantic_extractor.extract_features(text)
        style_features = self.style_extractor.extract_features(text)
        email_features = self.email_extractor.extract_features(text)
        combined_style = torch.cat([style_features, email_features])

        return {"semantic": semantic_features, "style": combined_style}
