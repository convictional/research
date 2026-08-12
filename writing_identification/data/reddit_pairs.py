"""Reddit pair generation for verification training with deterministic, reproducible pairs."""

import json
import logging
import random
from pathlib import Path
from typing import Tuple, Optional, Any
from datetime import datetime
import hashlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PairManifest:
    """Lightweight pair references - no text content stored."""

    pair_refs: list[Tuple[str, int, str, int, int]]  # (author1_id, text1_idx, author2_id, text2_idx, label)
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PairManifest":
        return cls(pair_refs=[(p[0], p[1], p[2], p[3], p[4]) for p in data["pair_refs"]], metadata=data["metadata"])

    def to_dict(self) -> dict[str, Any]:
        return {"pair_refs": [[p[0], p[1], p[2], p[3], p[4]] for p in self.pair_refs], "metadata": self.metadata}

    def reconstruct_pairs(self, texts_by_author: dict[str, list[str]]) -> list[Tuple[str, str, int]]:
        """Reconstruct full text pairs from lightweight references."""
        pairs = []
        for author1_id, text1_idx, author2_id, text2_idx, label in self.pair_refs:
            text1 = texts_by_author[author1_id][text1_idx]
            text2 = texts_by_author[author2_id][text2_idx]
            pairs.append((text1, text2, label))
        return pairs


class RedditPairGenerator:
    """Generate deterministic pairs from Reddit classification data for verification training."""

    def __init__(
        self,
        cache_dir: str = "cache/reddit_pairs",
        positive_ratio: float = 0.5,
        max_pairs_per_author: int = 100,
        min_samples_per_author: int = 2,
        seed: int = 42,
    ):
        """
        Initialize Reddit pair generator.

        Args:
            cache_dir: Directory to cache pair manifests
            positive_ratio: Ratio of positive (same author) pairs
            max_pairs_per_author: Maximum positive pairs per author
            min_samples_per_author: Minimum samples per author to include
            seed: Random seed for reproducible generation
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.positive_ratio = positive_ratio
        self.max_pairs_per_author = max_pairs_per_author
        self.min_samples_per_author = min_samples_per_author
        self.seed = seed

        logger.info(f"RedditPairGenerator initialized with seed={seed}")

    def _get_manifest_path(self, split: str, config_hash: str) -> Path:
        """Get path for pair manifest file."""
        filename = f"{split}_pairs_{config_hash}.json"
        return self.cache_dir / filename

    def _compute_config_hash(self, texts_by_author: dict[str, list[str]]) -> str:
        """Compute hash of configuration and input data for caching."""
        # Include generation parameters and author/text counts for cache invalidation
        config_data = {
            "positive_ratio": self.positive_ratio,
            "max_pairs_per_author": self.max_pairs_per_author,
            "min_samples_per_author": self.min_samples_per_author,
            "seed": self.seed,
            "authors": sorted(texts_by_author.keys()),
            "text_counts": {author: len(texts) for author, texts in texts_by_author.items()},
        }
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def generate_pairs(
        self, texts_by_author: dict[str, list[str]], split: str = "train", force_regenerate: bool = False
    ) -> list[Tuple[str, str, int]]:
        """
        Generate balanced pairs from texts by author.

        Args:
            texts_by_author: dictionary mapping author -> list of texts
            split: Data split name ("train" or "val")
            force_regenerate: Force regeneration even if cached manifest exists

        Returns:
            list of (text1, text2, label) tuples
        """
        # Check if we have a cached manifest
        config_hash = self._compute_config_hash(texts_by_author)
        manifest_path = self._get_manifest_path(split, config_hash)

        if manifest_path.exists() and not force_regenerate:
            logger.info(f"Loading existing pair manifest from {manifest_path}")
            manifest = self.load_pairs_manifest(manifest_path)
            return manifest.reconstruct_pairs(texts_by_author)

        logger.info(f"Generating new pairs for {split} split...")

        # Filter authors with sufficient samples
        filtered_authors = {
            author: texts for author, texts in texts_by_author.items() if len(texts) >= self.min_samples_per_author
        }

        logger.info(f"Filtered to {len(filtered_authors)} authors with ≥{self.min_samples_per_author} samples")

        # Guard for edge case
        if len(filtered_authors) < 2:
            logger.warning("Less than 2 authors available - cannot generate negative pairs")
            return []

        # Set seed for reproducible generation
        rng = random.Random(self.seed)

        # Generate positive pairs (same author) as references
        positive_refs = []
        for author, texts in filtered_authors.items():
            author_pairs = 0
            # Generate all possible pairs within author
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if author_pairs >= self.max_pairs_per_author:
                        break
                    positive_refs.append(
                        (author, i, author, j, 1)
                    )  # (author1_id, text1_idx, author2_id, text2_idx, label)
                    author_pairs += 1
                if author_pairs >= self.max_pairs_per_author:
                    break

        # Shuffle positive pairs
        rng.shuffle(positive_refs)

        # Calculate target counts based on actual positives available
        actual_positive_count = len(positive_refs)
        if actual_positive_count == 0:
            logger.warning("No positive pairs could be generated - each author needs at least 2 texts")
            return []

        # Calculate negative count to maintain desired ratio
        if self.positive_ratio > 0:
            total_target = int(actual_positive_count / self.positive_ratio)
            negative_target = total_target - actual_positive_count
        else:
            negative_target = actual_positive_count  # 50/50 fallback

        logger.info(f"Generated {actual_positive_count} positive pairs")

        # Generate negative pairs (different authors) as references
        authors_list = list(filtered_authors.keys())
        negative_refs = []

        for _ in range(negative_target):
            # Sample two different authors
            author1, author2 = rng.sample(authors_list, 2)
            text1_idx = rng.randint(0, len(filtered_authors[author1]) - 1)
            text2_idx = rng.randint(0, len(filtered_authors[author2]) - 1)
            negative_refs.append((author1, text1_idx, author2, text2_idx, 0))

        logger.info(f"Generated {len(negative_refs)} negative pairs (target: {negative_target})")

        # Combine and shuffle all pair references
        all_pair_refs = positive_refs + negative_refs
        rng.shuffle(all_pair_refs)

        # Save manifest with references
        self.save_pairs_manifest(all_pair_refs, split, config_hash, filtered_authors)

        logger.info(
            f"Generated {len(all_pair_refs)} total pairs ({actual_positive_count} positive, {len(negative_refs)} negative)"
        )

        # Reconstruct full text pairs for return
        pairs = []
        for author1_id, text1_idx, author2_id, text2_idx, label in all_pair_refs:
            text1 = filtered_authors[author1_id][text1_idx]
            text2 = filtered_authors[author2_id][text2_idx]
            pairs.append((text1, text2, label))

        return pairs

    def save_pairs_manifest(
        self,
        pair_refs: list[Tuple[str, int, str, int, int]],
        split: str,
        config_hash: str,
        texts_by_author: dict[str, list[str]],
    ) -> Path:
        """Save pairs manifest with lightweight references."""
        manifest_path = self._get_manifest_path(split, config_hash)

        # Calculate statistics
        positive_count = sum(1 for _, _, _, _, label in pair_refs if label == 1)
        negative_count = len(pair_refs) - positive_count

        manifest = PairManifest(
            pair_refs=pair_refs,
            metadata={
                "split": split,
                "config_hash": config_hash,
                "generation_timestamp": datetime.now().isoformat(),
                "seed": self.seed,
                "positive_ratio": self.positive_ratio,
                "max_pairs_per_author": self.max_pairs_per_author,
                "min_samples_per_author": self.min_samples_per_author,
                "total_pairs": len(pair_refs),
                "positive_pairs": positive_count,
                "negative_pairs": negative_count,
                "actual_positive_ratio": positive_count / len(pair_refs) if pair_refs else 0,
                "num_authors": len(texts_by_author),
                "total_texts": sum(len(texts) for texts in texts_by_author.values()),
                "authors": sorted(texts_by_author.keys()),
                "author_text_counts": {author: len(texts) for author, texts in texts_by_author.items()},
            },
        )

        # Save to file
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info(f"Saved pair manifest to {manifest_path}")
        logger.info(f"  Total pairs: {len(pair_refs)}")
        logger.info(f"  Positive: {positive_count} ({positive_count / len(pair_refs) * 100:.1f}%)")
        logger.info(f"  Negative: {negative_count} ({negative_count / len(pair_refs) * 100:.1f}%)")

        return manifest_path

    def load_pairs_manifest(self, manifest_path: Path) -> PairManifest:
        """Load pairs manifest from file."""
        with open(manifest_path, "r") as f:
            data = json.load(f)

        manifest = PairManifest.from_dict(data)

        logger.info(
            f"Loaded manifest: {manifest.metadata['total_pairs']} pairs "
            f"({manifest.metadata['positive_pairs']} pos, {manifest.metadata['negative_pairs']} neg)"
        )

        return manifest

    def get_cached_pairs(
        self, texts_by_author: dict[str, list[str]], split: str
    ) -> Optional[list[Tuple[str, str, int]]]:
        """Get cached pairs if they exist and are valid."""
        config_hash = self._compute_config_hash(texts_by_author)
        manifest_path = self._get_manifest_path(split, config_hash)

        if manifest_path.exists():
            try:
                manifest = self.load_pairs_manifest(manifest_path)
                return manifest.reconstruct_pairs(texts_by_author)
            except Exception as e:
                logger.warning(f"Failed to load cached pairs: {e}")
                return None

        return None


def generate_reddit_pairs_for_training(
    train_texts_by_author: dict[str, list[str]],
    val_texts_by_author: dict[str, list[str]],
    cache_dir: str = "cache/reddit_pairs",
    positive_ratio: float = 0.5,
    max_pairs_per_author: int = 100,
    seed: int = 42,
) -> Tuple[list[Tuple[str, str, int]], list[Tuple[str, str, int]]]:
    """
    Generate training and validation pairs from Reddit data.

    Args:
        train_texts_by_author: Training texts by author
        val_texts_by_author: Validation texts by author
        cache_dir: Cache directory for pair manifests
        positive_ratio: Ratio of positive pairs
        max_pairs_per_author: Maximum positive pairs per author
        seed: Random seed

    Returns:
        Tuple of (train_pairs, val_pairs)
    """
    generator = RedditPairGenerator(
        cache_dir=cache_dir, positive_ratio=positive_ratio, max_pairs_per_author=max_pairs_per_author, seed=seed
    )

    # Generate pairs for both splits
    train_pairs = generator.generate_pairs(train_texts_by_author, "train")
    val_pairs = generator.generate_pairs(val_texts_by_author, "val")

    return train_pairs, val_pairs
