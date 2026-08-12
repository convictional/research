"""
PyTorch Dataset for loading training pairs.

Loads pairs from PostgreSQL and provides tokenized batches for training.
"""

import random
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ..data.models import TrainingPair


class TrainingPairDataset(Dataset):
    """
    Dataset for contrastive learning with anchor-positive pairs.

    Loads all pairs into memory at initialization (fast for 5K-50K pairs).
    Tokenizes on-the-fly using the provided tokenizer.
    """

    def __init__(
        self,
        pairs: list[TrainingPair],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset with training pairs.

        Args:
            pairs: List of TrainingPair objects from database
            tokenizer: HuggingFace tokenizer for the base model
            max_length: Maximum sequence length for tokenization
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return number of pairs in dataset."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single training pair.

        Returns:
            Dictionary with tokenized anchor and positive:
            {
                'anchor_input_ids': [seq_len],
                'anchor_attention_mask': [seq_len],
                'positive_input_ids': [seq_len],
                'positive_attention_mask': [seq_len],
            }
        """
        pair = self.pairs[idx]

        # Tokenize anchor
        anchor_encoded = self.tokenizer(
            pair.anchor_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize positive
        positive_encoded = self.tokenizer(
            pair.positive_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "anchor_input_ids": anchor_encoded["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_encoded["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoded["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoded["attention_mask"].squeeze(0),
        }

    @classmethod
    def train_val_split(
        cls,
        pairs: list[TrainingPair],
        tokenizer: AutoTokenizer,
        val_ratio: float = 0.1,
        max_length: int = 512,
        seed: int = 42,
    ) -> tuple["TrainingPairDataset", "TrainingPairDataset"]:
        """
        Split pairs into train and validation datasets.

        Args:
            pairs: List of all training pairs
            tokenizer: HuggingFace tokenizer
            val_ratio: Fraction of data for validation (default 0.1 = 10%)
            max_length: Maximum sequence length
            seed: Random seed for reproducible splits

        Returns:
            (train_dataset, val_dataset)
        """
        # Shuffle with seed for reproducibility
        pairs_copy = pairs.copy()
        random.seed(seed)
        random.shuffle(pairs_copy)

        # Split
        val_size = int(len(pairs_copy) * val_ratio)
        val_pairs = pairs_copy[:val_size]
        train_pairs = pairs_copy[val_size:]

        # Create datasets
        train_dataset = cls(train_pairs, tokenizer, max_length)
        val_dataset = cls(val_pairs, tokenizer, max_length)

        return train_dataset, val_dataset


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Stacks individual samples into batches.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Dictionary with batched tensors:
        {
            'anchor_input_ids': [batch_size, seq_len],
            'anchor_attention_mask': [batch_size, seq_len],
            'positive_input_ids': [batch_size, seq_len],
            'positive_attention_mask': [batch_size, seq_len],
        }
    """
    return {
        "anchor_input_ids": torch.stack([item["anchor_input_ids"] for item in batch]),
        "anchor_attention_mask": torch.stack([item["anchor_attention_mask"] for item in batch]),
        "positive_input_ids": torch.stack([item["positive_input_ids"] for item in batch]),
        "positive_attention_mask": torch.stack([item["positive_attention_mask"] for item in batch]),
    }
