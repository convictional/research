#!/usr/bin/env python3
"""Regenerate dataset using team-level ground truth.

This script creates train/dev/test splits where ground truth is the
pooled priorities from ALL team members, not just one person.

Usage:
    uv run python scripts/03_regenerate_team_data.py
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from src.config import config
from src.standup_parser import parse_team_standups
from src.context_retriever import ContextRetriever
from src.data_splitter import DataSplitter


async def main():
    print("=" * 60)
    print("REGENERATE TEAM DATASET")
    print("=" * 60)

    # Load API key for embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env.secrets")

    # Parse team standups
    print("\nParsing team standup documents...")
    db_cutoff = datetime.strptime(
        config.get("data.db_content_cutoff", "2025-05-19"), "%Y-%m-%d"
    )

    team_entries = parse_team_standups(before_date=db_cutoff)

    print(f"Found {len(team_entries)} team standup dates")

    # Show sample
    if team_entries:
        sample = team_entries[-1]
        print(f"  Sample date {sample.date.date()}: {len(sample.priorities)} priorities")

    # Initialize context retriever
    print("\nInitializing context retriever...")
    context_retriever = ContextRetriever(openai_api_key)
    await context_retriever.connect()
    print("Connected to database")

    # Prepare dataset
    print("\nPreparing dataset with context...")
    data_splitter = DataSplitter(context_retriever)

    # Use a separate output directory for team data
    data_dir = Path("data_team")
    data_dir.mkdir(exist_ok=True)

    train_examples, dev_examples, test_examples = await data_splitter.prepare_dataset(
        team_entries, output_dir=data_dir
    )

    print(f"\nDataset created:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Dev: {len(dev_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")

    # Show sample ground truth counts
    if train_examples:
        sample = train_examples[0]
        gt_count = len(sample.standup_entry.priorities)
        print(f"\nSample ground truth ({sample.standup_entry.date.date()}): {gt_count} priorities")
        for p in sample.standup_entry.priorities[:3]:
            print(f"  {p.rationale} {p.description[:50]}...")

    await context_retriever.close()

    print("\nDone! Files saved to data_team/")


if __name__ == "__main__":
    asyncio.run(main())
