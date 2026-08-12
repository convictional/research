"""
Create train/dev/test splits from standup entries.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

from .models import ExampleData, StandupEntry, Context
from .context_retriever import ContextRetriever
from .config import config


class DataSplitter:
    """
    Split standup entries into train/dev/test sets and prepare full examples.
    """

    def __init__(self, context_retriever: ContextRetriever):
        self.context_retriever = context_retriever

    async def prepare_dataset(
        self,
        standup_entries: List[StandupEntry],
        output_dir: Path = Path("data"),
    ) -> tuple[List[ExampleData], List[ExampleData], List[ExampleData]]:
        """
        Prepare complete dataset with context for each standup entry.

        Args:
            standup_entries: All standup entries (sorted by date ascending)
            output_dir: Directory to save splits

        Returns:
            Tuple of (train_examples, dev_examples, test_examples)
        """
        # Get split ratios from config
        train_ratio = config.get("data.train_ratio", 0.6)
        dev_ratio = config.get("data.dev_ratio", 0.2)

        # Calculate split indices
        n_total = len(standup_entries)
        n_train = int(n_total * train_ratio)
        n_dev = int(n_total * dev_ratio)

        # Split chronologically (important for temporal validity)
        train_entries = standup_entries[:n_train]
        dev_entries = standup_entries[n_train : n_train + n_dev]
        test_entries = standup_entries[n_train + n_dev :]

        print(f"Splitting {n_total} entries:")
        print(f"  Train: {len(train_entries)} ({train_entries[0].date.date()} to {train_entries[-1].date.date()})")
        print(f"  Dev:   {len(dev_entries)} ({dev_entries[0].date.date()} to {dev_entries[-1].date.date()})")
        print(f"  Test:  {len(test_entries)} ({test_entries[0].date.date()} to {test_entries[-1].date.date()})")

        # Prepare full examples with context
        print("\nFetching context for each entry...")
        train_examples = await self._prepare_examples(train_entries, standup_entries, "train")
        dev_examples = await self._prepare_examples(dev_entries, standup_entries, "dev")
        test_examples = await self._prepare_examples(test_entries, standup_entries, "test")

        # Save to disk
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_examples(train_examples, output_dir / "train.jsonl")
        self._save_examples(dev_examples, output_dir / "dev.jsonl")
        self._save_examples(test_examples, output_dir / "test.jsonl")

        print(f"\nSaved splits to {output_dir}/")

        return train_examples, dev_examples, test_examples

    async def _prepare_examples(
        self,
        entries: List[StandupEntry],
        all_entries: List[StandupEntry],
        split: str,
    ) -> List[ExampleData]:
        """Prepare full examples with context for a set of entries."""
        examples = []

        for entry in entries:
            # Find previous day's work (for context query)
            previous_work = self._get_previous_work(entry, all_entries)

            # Get historical context for this date
            context = await self.context_retriever.get_context_for_date(
                target_date=entry.date,
                previous_work=previous_work,
                top_k=config.get("context.top_k", 20),
            )

            # Create example
            example = ExampleData(
                id=f"{split}_{entry.date.strftime('%Y%m%d')}",
                standup_entry=entry,
                context=context,
                split=split,
            )

            examples.append(example)

        return examples

    def _get_previous_work(
        self, entry: StandupEntry, all_entries: List[StandupEntry]
    ) -> str:
        """
        Get what was worked on the day before (for context query).

        This uses the previous standup entry's priorities as the search query.
        """
        # Find the entry before this one
        for i, e in enumerate(all_entries):
            if e.date == entry.date and i > 0:
                prev_entry = all_entries[i - 1]
                # Combine all priorities into a query
                priorities_text = " ".join(
                    p.description for p in prev_entry.priorities
                )
                return priorities_text

        # No previous entry found, return None for fallback query
        return None

    def _save_examples(self, examples: List[ExampleData], filepath: Path):
        """Save examples to JSONL file."""
        with open(filepath, "w") as f:
            for example in examples:
                f.write(example.model_dump_json() + "\n")

    @staticmethod
    def load_examples(filepath: Path) -> List[ExampleData]:
        """Load examples from JSONL file."""
        examples = []
        with open(filepath, "r") as f:
            for line in f:
                examples.append(ExampleData.model_validate_json(line))
        return examples
