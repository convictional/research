"""PyTorch dataset for authorship verification training."""

import random
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset

from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
from features.email_patterns import EmailPatternExtractor


class AuthorshipPairDataset(Dataset):
    """Dataset for Siamese network training with text pairs."""

    def __init__(
        self,
        texts_by_author: Optional[dict[str, list[str]]] = None,
        semantic_extractor: Optional[SemanticFeatureExtractor] = None,
        style_extractor: Optional[StyleFeatureExtractor] = None,
        email_extractor: Optional[EmailPatternExtractor] = None,
        positive_ratio: float = 0.5,
        max_pairs_per_author: int = None,  # Remove artificial limit
        hard_negative: bool = False,
        hard_negative_top_k: int = 1,
        pairs: Optional[List[Tuple[str, str, int]]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            texts_by_author: dictionary mapping author_id -> list of texts (optional if pairs provided)
            semantic_extractor: Fitted semantic feature extractor
            style_extractor: Fitted style feature extractor
            email_extractor: Email pattern extractor
            positive_ratio: Ratio of positive (same author) pairs
            max_pairs_per_author: Maximum pairs to generate per author
            pairs: Pre-generated pairs (text1, text2, label) - if provided, skips pair generation
        """
        self.semantic_extractor = semantic_extractor
        self.style_extractor = style_extractor
        self.email_extractor = email_extractor
        self.positive_ratio = positive_ratio
        self.hard_negative = hard_negative
        self.hard_negative_top_k = hard_negative_top_k

        if pairs is not None:
            # Use pre-generated pairs (from manifest)
            self.pairs = pairs
            self.texts_by_author = texts_by_author  # May be None for pre-computed features
            self.authors = []  # Not needed when using pre-generated pairs
            self._embeddings_by_text = None
        else:
            # Generate pairs dynamically
            if texts_by_author is None:
                raise ValueError("Either texts_by_author or pairs must be provided")

            self.texts_by_author = texts_by_author
            self.authors = list(texts_by_author.keys())

            # Pre-compute semantic embeddings if hard-negative mining is enabled
            self._embeddings_by_text = None
            if self.hard_negative:
                self._precompute_embeddings()

            self.pairs = self._generate_pairs(max_pairs_per_author)

    @classmethod
    def from_manifest(
        cls,
        pairs: List[Tuple[str, str, int]],
        semantic_extractor: SemanticFeatureExtractor,
        style_extractor: StyleFeatureExtractor,
        email_extractor: EmailPatternExtractor,
    ) -> "AuthorshipPairDataset":
        """
        Create dataset from pre-generated pairs manifest.

        Args:
            pairs: List of (text1, text2, label) tuples
            semantic_extractor: Fitted semantic feature extractor
            style_extractor: Fitted style feature extractor
            email_extractor: Email pattern extractor

        Returns:
            AuthorshipPairDataset instance using the provided pairs
        """
        return cls(
            texts_by_author=None,
            semantic_extractor=semantic_extractor,
            style_extractor=style_extractor,
            email_extractor=email_extractor,
            pairs=pairs,
        )

    def _precompute_embeddings(self):
        """Compute and cache semantic embeddings for every text."""
        import logging

        logger = logging.getLogger(__name__)

        self._embeddings_by_text = {}
        total_texts = sum(len(texts) for texts in self.texts_by_author.values())
        processed = 0

        logger.info(f"Pre-computing embeddings for {total_texts} texts for hard-negative mining...")
        logger.info("This may take a few minutes on first run...")

        for author_idx, (author, texts) in enumerate(self.texts_by_author.items()):
            for txt in texts:
                # Avoid duplicate computation if the same text string appears twice
                if txt not in self._embeddings_by_text:
                    self._embeddings_by_text[txt] = self.semantic_extractor.extract_features(txt).detach()
                    processed += 1

                    # Log progress every 100 texts
                    if processed % 100 == 0:
                        logger.info(
                            f"  Computed embeddings for {processed}/{total_texts} texts ({100 * processed / total_texts:.1f}%)"
                        )

        logger.info(f"Finished computing {len(self._embeddings_by_text)} unique embeddings")

    def _generate_pairs(self, max_pairs_per_author: int) -> list[Tuple[str, str, int]]:
        """Generate training pairs with labels."""
        import logging

        logger = logging.getLogger(__name__)

        pairs = []

        logger.info(f"Generating training pairs for {len(self.authors)} authors...")

        # Generate positive pairs (same author)
        for author_idx, author in enumerate(self.authors):
            texts = self.texts_by_author[author]
            if len(texts) < 2:
                continue

            # Generate pairs within author
            author_pairs = 0
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if max_pairs_per_author is not None and author_pairs >= max_pairs_per_author:
                        break
                    pairs.append((texts[i], texts[j], 1))  # Same author = 1
                    author_pairs += 1
                if max_pairs_per_author is not None and author_pairs >= max_pairs_per_author:
                    break

        positive_count = len(pairs)
        logger.info(f"Generated {positive_count} positive pairs")

        # Generate negative pairs (different authors)
        negative_target = int(positive_count / self.positive_ratio - positive_count)
        negative_count = 0

        logger.info(
            f"Generating {negative_target} negative pairs with {'hard' if self.hard_negative else 'random'} sampling..."
        )

        while negative_count < negative_target:
            # Sample anchor author and at least one negative author
            author1, author2 = random.sample(self.authors, 2)

            # Choose anchor text from author1
            text1 = random.choice(self.texts_by_author[author1])

            if self.hard_negative and self._embeddings_by_text is not None:
                anchor_emb = self._embeddings_by_text[text1]

                # Gather embeddings & texts of candidate negatives (author2)
                candidate_texts = self.texts_by_author[author2]

                candidate_embeddings = torch.stack([self._embeddings_by_text[txt] for txt in candidate_texts])

                # Compute cosine similarity between anchor and all candidates
                sim_scores = torch.nn.functional.cosine_similarity(
                    anchor_emb.unsqueeze(0).expand_as(candidate_embeddings), candidate_embeddings, dim=1
                )

                # Select top-k most similar negatives and randomly pick one of them
                topk = min(self.hard_negative_top_k, len(candidate_texts))
                topk_indices = torch.topk(sim_scores, k=topk, largest=True).indices.tolist()
                chosen_idx = random.choice(topk_indices)
                text2 = candidate_texts[chosen_idx]
            else:
                # Fallback to random negative sampling
                text2 = random.choice(self.texts_by_author[author2])

            pairs.append((text1, text2, 0))  # Different authors = 0
            negative_count += 1

            # Log progress every 1000 negative pairs
            if negative_count % 1000 == 0:
                logger.info(
                    f"  Generated {negative_count}/{negative_target} negative pairs ({100 * negative_count / negative_target:.1f}%)"
                )

        logger.info(
            f"Finished generating {len(pairs)} total pairs ({positive_count} positive, {negative_count} negative)"
        )

        # Shuffle pairs
        random.shuffle(pairs)
        logger.info("Shuffled pairs for training")
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample."""
        text1, text2, label = self.pairs[idx]

        # Extract features for both texts
        features1 = self._extract_all_features(text1)
        features2 = self._extract_all_features(text2)

        return {
            "semantic_features1": features1["semantic"],
            "style_features1": features1["style"],
            "semantic_features2": features2["semantic"],
            "style_features2": features2["style"],
            "label": torch.tensor(label, dtype=torch.float32),
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


class TripletDataset(Dataset):
    """Dataset for triplet loss training."""

    def __init__(
        self,
        texts_by_author: dict[str, list[str]],
        semantic_extractor: SemanticFeatureExtractor,
        style_extractor: StyleFeatureExtractor,
        email_extractor: EmailPatternExtractor,
        triplets_per_author: int = 50,
    ):
        """
        Initialize triplet dataset.

        Args:
            texts_by_author: dictionary mapping author_id -> list of texts
            semantic_extractor: Fitted semantic feature extractor
            style_extractor: Fitted style feature extractor
            email_extractor: Email pattern extractor
            triplets_per_author: Number of triplets to generate per author
        """
        self.texts_by_author = texts_by_author
        self.semantic_extractor = semantic_extractor
        self.style_extractor = style_extractor
        self.email_extractor = email_extractor

        self.authors = [author for author, texts in texts_by_author.items() if len(texts) >= 2]
        self.triplets = self._generate_triplets(triplets_per_author)

    def _generate_triplets(self, triplets_per_author: int) -> list[Tuple[str, str, str]]:
        """Generate anchor-positive-negative triplets."""
        triplets = []

        for author in self.authors:
            author_texts = self.texts_by_author[author]
            other_authors = [a for a in self.authors if a != author]

            for _ in range(triplets_per_author):
                # Sample anchor and positive from same author
                anchor, positive = random.sample(author_texts, 2)

                # Sample negative from different author
                negative_author = random.choice(other_authors)
                negative = random.choice(self.texts_by_author[negative_author])

                triplets.append((anchor, positive, negative))

        random.shuffle(triplets)
        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a triplet sample."""
        anchor_text, positive_text, negative_text = self.triplets[idx]

        # Extract features for all texts
        anchor_features = self._extract_all_features(anchor_text)
        positive_features = self._extract_all_features(positive_text)
        negative_features = self._extract_all_features(negative_text)

        return {
            "anchor_semantic": anchor_features["semantic"],
            "anchor_style": anchor_features["style"],
            "positive_semantic": positive_features["semantic"],
            "positive_style": positive_features["style"],
            "negative_semantic": negative_features["semantic"],
            "negative_style": negative_features["style"],
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


def create_data_loaders(
    train_texts: dict[str, list[str]],
    val_texts: dict[str, list[str]],
    batch_size: int = 32,
    num_workers: int = 4,
    hard_negative: bool = False,
    hard_negative_top_k: int = 1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        train_texts: Training texts by author
        val_texts: Validation texts by author
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Initialize extractors
    semantic_extractor = SemanticFeatureExtractor()
    style_extractor = StyleFeatureExtractor()
    email_extractor = EmailPatternExtractor()

    # Fit style extractor on training data
    all_train_texts = []
    for texts in train_texts.values():
        all_train_texts.extend(texts)
    style_extractor.fit(all_train_texts)

    # Create datasets
    train_dataset = AuthorshipPairDataset(
        train_texts,
        semantic_extractor,
        style_extractor,
        email_extractor,
        hard_negative=hard_negative,
        hard_negative_top_k=hard_negative_top_k,
    )
    val_dataset = AuthorshipPairDataset(
        val_texts,
        semantic_extractor,
        style_extractor,
        email_extractor,
        hard_negative=False,  # Do not use hard negatives in validation
    )

    # Create data loaders (disable multiprocessing and pin_memory for MPS compatibility)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for MPS
        pin_memory=False,  # Disable pin_memory for MPS
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing for MPS
        pin_memory=False,  # Disable pin_memory for MPS
    )

    return train_loader, val_loader
