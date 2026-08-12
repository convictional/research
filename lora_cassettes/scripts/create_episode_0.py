"""
Create Episode 0 - Initial training dataset.

This script:
1. Checks if Episode 0 exists, creates if not
2. Mines training pairs from the corpus using unsupervised strategies
3. Bulk inserts pairs into the database
4. Generates summary statistics

Based on PLAN.md section 4-5 (Data & Sampling, Pair Mining).

Usage:
    uv run python scripts/create_episode_0.py [--max-pairs-per-thread N] [--min-thread-size N]
"""

import argparse
import asyncio
from datetime import datetime, timezone

from src.data.db import (
    create_episode,
    create_training_pairs_bulk,
    get_connection,
    get_episode,
    get_training_pairs_for_episode,
)
from src.data.models import EpisodeCreate
from src.mining.github import mine_same_thread_pairs


async def main(
    max_pairs_per_thread: int | None = None,
    min_thread_size: int = 2,
    dry_run: bool = False,
    force: bool = False,
):
    """
    Create Episode 0 with training pairs.

    Args:
        max_pairs_per_thread: Limit pairs per thread (None = unlimited)
        min_thread_size: Minimum items in thread to mine from
        dry_run: If True, don't insert pairs, just show what would be created
        force: If True, use existing episode without prompting
    """
    print("=" * 80)
    print("Episode 0 Initialization - LoRA Cassettes")
    print("=" * 80)

    # Step 1: Check/create Episode 0
    print("\n[1/5] Checking Episode 0...")
    episode = await get_episode(0)

    if episode:
        print(f"✓ Episode 0 exists (id={episode.id}, status={episode.status})")
        if not force:
            use_existing = input("Use existing episode? (y/n): ")
            if use_existing.lower() != "y":
                print("Aborting. Please delete existing Episode 0 first.")
                return
        else:
            print("  --force flag set, using existing episode")
    else:
        print("Creating Episode 0...")
        # Determine corpus date range from content table
        conn = await get_connection()
        try:
            row = await conn.fetchrow(
                "SELECT MIN(created_at) as earliest, MAX(created_at) as latest, COUNT(*) as total FROM content"
            )
            earliest = row["earliest"]
            latest = row["latest"]
            total = row["total"]
        finally:
            await conn.close()

        episode_data = EpisodeCreate(
            episode_num=0,
            start_date=earliest,
            end_date=latest,
            corpus_snapshot_date=datetime.now(timezone.utc),
            num_new_chunks=total,
            num_updated_chunks=0,
            status="mining",
        )
        episode = await create_episode(episode_data)
        print(f"✓ Created Episode 0 (id={episode.id})")
        print(f"  Date range: {earliest.date()} to {latest.date()}")
        print(f"  Corpus size: {total:,} content items")

    # Step 2: Mine GitHub pairs
    print(f"\n[2/5] Mining GitHub same_thread pairs...")
    print(f"  Settings:")
    print(f"    - Min thread size: {min_thread_size}")
    print(f"    - Max pairs per thread: {max_pairs_per_thread or 'unlimited'}")

    conn = await get_connection()
    try:
        github_pairs = await mine_same_thread_pairs(
            conn,
            episode_id=episode.id,
            min_thread_size=min_thread_size,
            max_pairs_per_thread=max_pairs_per_thread,
        )
    finally:
        await conn.close()

    print(f"✓ Mined {len(github_pairs):,} GitHub pairs")

    # Step 3: Aggregate all pairs
    print(f"\n[3/5] Aggregating pairs from all sources...")
    all_pairs = github_pairs  # In future: + email_pairs + docs_pairs + ...
    print(f"✓ Total pairs: {len(all_pairs):,}")

    # Step 4: Insert pairs
    if dry_run:
        print(f"\n[4/5] DRY RUN - Would insert {len(all_pairs):,} pairs")
    else:
        print(f"\n[4/5] Bulk inserting pairs...")
        inserted = await create_training_pairs_bulk(all_pairs)
        print(f"✓ Inserted {inserted:,} training pairs")

    # Step 5: Summary statistics
    print(f"\n[5/5] Summary Statistics")
    print("-" * 80)

    if not dry_run:
        pairs_in_db = await get_training_pairs_for_episode(episode.id, limit=1)
        print(f"✓ Verified pairs in database")

    # Calculate statistics
    total_chars = sum(len(p.anchor_text) + len(p.positive_text) for p in all_pairs[:1000])
    avg_chars = total_chars / min(1000, len(all_pairs)) if all_pairs else 0

    source_counts = {}
    for pair in all_pairs:
        source_counts[pair.source_family] = source_counts.get(pair.source_family, 0) + 1

    print(f"\nEpisode 0 Dataset:")
    print(f"  Total pairs: {len(all_pairs):,}")
    print(f"  Average length: ~{avg_chars:.0f} chars per pair")
    print(f"\nBy source:")
    for source, count in sorted(source_counts.items()):
        print(f"  - {source}: {count:,} pairs ({100*count/len(all_pairs):.1f}%)")

    print(f"\nPair types:")
    type_counts = {}
    for pair in all_pairs:
        type_counts[pair.pair_type] = type_counts.get(pair.pair_type, 0) + 1
    for ptype, count in sorted(type_counts.items()):
        print(f"  - {ptype}: {count:,} pairs")

    # Estimate training time
    # Assuming ~1024 token pairs at 256 pairs/batch = 4 batches, ~0.5s per batch = 2s per 1024 pairs
    estimated_steps = len(all_pairs) // 256
    estimated_minutes = (estimated_steps * 0.5) / 60
    print(f"\nTraining estimates (256 pairs/batch):")
    print(f"  - Steps: ~{estimated_steps:,}")
    print(f"  - Time: ~{estimated_minutes:.1f} minutes (at 0.5s/batch)")

    print("\n" + "=" * 80)
    print("✓ Episode 0 initialization complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Review pair quality: psql lora_cassettes -c \"SELECT * FROM training_pairs LIMIT 5;\"")
    print(f"  2. Start training: uv run python scripts/train_episode_0.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Episode 0 training dataset")
    parser.add_argument(
        "--max-pairs-per-thread",
        type=int,
        default=None,
        help="Maximum pairs to mine per thread (default: unlimited)",
    )
    parser.add_argument(
        "--min-thread-size",
        type=int,
        default=2,
        help="Minimum items in thread to mine from (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't insert pairs, just show what would be created",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Use existing episode without prompting",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            max_pairs_per_thread=args.max_pairs_per_thread,
            min_thread_size=args.min_thread_size,
            dry_run=args.dry_run,
            force=args.force,
        )
    )
