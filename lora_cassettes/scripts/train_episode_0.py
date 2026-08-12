"""
Train Episode 0 LoRA adapter using contrastive learning.

This script:
1. Loads Episode 0 training pairs from database
2. Creates train/validation split
3. Initializes LoRA encoder on MPS device
4. Trains with InfoNCE loss and in-batch negatives
5. Saves adapter and creates registry entry

Based on PLAN.md section 6 (Episodic Fine-Tuning).

Usage:
    # Validation run (100 pairs, 5 epochs)
    PYTHONPATH=. uv run python scripts/train_episode_0.py --subset 100 --epochs 5

    # Full training (5K pairs, 20 epochs)
    PYTHONPATH=. uv run python scripts/train_episode_0.py

    # Custom hyperparameters
    PYTHONPATH=. uv run python scripts/train_episode_0.py --batch-size 256 --lr 1e-4 --device cpu
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.db import (
    create_adapter,
    get_connection,
    get_episode,
    get_training_pairs_for_episode,
)
from src.data.models import AdapterCreate
from src.model.encoder import LoRAEncoderRetriever
from src.training.config import TrainingConfig
from src.training.contrastive import ContrastiveTrainer
from src.training.dataset import TrainingPairDataset, collate_fn


async def load_training_data(episode_num: int, subset: int | None = None):
    """
    Load training pairs for an episode.

    Args:
        episode_num: Episode number to load
        subset: If provided, only load first N pairs (for testing)

    Returns:
        (episode, pairs)
    """
    print(f"\n[1/6] Loading Episode {episode_num} data...")

    # Get episode
    episode = await get_episode(episode_num)
    if not episode:
        raise ValueError(f"Episode {episode_num} not found. Run create_episode_0.py first.")

    print(f"✓ Episode {episode_num} found (id={episode.id}, status={episode.status})")

    # Load training pairs
    pairs = await get_training_pairs_for_episode(episode.id)

    if not pairs:
        raise ValueError(f"No training pairs found for Episode {episode_num}")

    # Subset if requested
    if subset:
        pairs = pairs[:subset]
        print(f"✓ Using subset: {len(pairs)} pairs (first {subset})")
    else:
        print(f"✓ Loaded {len(pairs):,} training pairs")

    return episode, pairs


def create_dataloaders(
    pairs,
    tokenizer,
    config: TrainingConfig,
):
    """
    Create train and validation dataloaders.

    Args:
        pairs: List of training pairs
        tokenizer: HuggingFace tokenizer
        config: Training configuration

    Returns:
        (train_loader, val_loader)
    """
    print(f"\n[2/6] Creating dataloaders...")

    # Split into train/val
    train_dataset, val_dataset = TrainingPairDataset.train_val_split(
        pairs,
        tokenizer,
        val_ratio=config.validation_split,
        max_length=config.max_seq_length,
    )

    print(f"✓ Train: {len(train_dataset)} pairs")
    print(f"✓ Val: {len(val_dataset)} pairs")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # Not needed on MPS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    return train_loader, val_loader


def initialize_model(config: TrainingConfig):
    """
    Initialize LoRA encoder for training.

    Args:
        config: Training configuration

    Returns:
        (encoder, peft_model, tokenizer)
    """
    print(f"\n[3/6] Initializing model...")

    base_model_path = "src/model/base_models/e5-base-v2"

    # Load encoder
    encoder = LoRAEncoderRetriever(
        base_model_path=base_model_path,
        device="cpu",  # Start on CPU, will move to device in trainer
        normalize_embeddings=True,
    )

    print(f"✓ Loaded base model: {base_model_path}")

    # Create LoRA config
    lora_config = encoder.create_lora_config(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )

    # Prepare for training
    peft_model = encoder.prepare_for_training(lora_config)

    # Verify only LoRA params are trainable
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    print(f"✓ LoRA config: r={config.lora_r}, α={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"✓ Trainable params: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")

    return encoder, peft_model, encoder.tokenizer


async def save_adapter_registry(
    episode,
    adapter_path: Path,
    config: TrainingConfig,
    metrics: dict,
):
    """
    Create or update adapter registry entry in database.

    Args:
        episode: Episode object
        adapter_path: Path to saved adapter
        config: Training configuration
        metrics: Training metrics dictionary
    """
    print(f"\n[6/6] Creating adapter registry entry...")

    # Create semantic version ID
    adapter_id = f"convx/{config.adapter_name}-v1.0.0"

    # Check if adapter already exists
    from src.data.db import get_adapter_by_id
    existing = await get_adapter_by_id(adapter_id)

    if existing:
        print(f"  ! Adapter {adapter_id} already exists (id={existing.id})")
        print(f"  Updating existing adapter...")

        # Delete old entry
        conn = await get_connection()
        try:
            await conn.execute("DELETE FROM adapters WHERE adapter_id = $1", adapter_id)
            print(f"  ✓ Deleted old entry")
        finally:
            await conn.close()

    adapter_data = AdapterCreate(
        adapter_id=adapter_id,
        base_model="e5-base-v2",
        episode_id=episode.id,
        sources=["github"],  # Currently only GitHub pairs
        objective="contrastive",
        train_start_date=episode.start_date,
        train_end_date=episode.end_date,
        replay_pct=0.0,  # No replay in Episode 0
        hnsw_index_id=None,  # Not built yet
        lora_config={
            "r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
        },
        training_config={
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
            "temperature": config.temperature,
        },
        metrics={
            "train_loss": metrics["final_train_loss"],
            "val_loss": metrics["final_val_loss"],
            "best_val_loss": metrics["best_val_loss"],
            "training_time": metrics["total_time"],
        },
        stability_delta=None,  # Will compute in evaluation
        status="active",
        storage_path=str(adapter_path),
        created_by="train_episode_0.py",
    )

    adapter = await create_adapter(adapter_data)
    print(f"✓ Created adapter registry entry: {adapter.adapter_id}")
    print(f"  ID: {adapter.id}")
    print(f"  Status: {adapter.status}")

    return adapter


async def main(args):
    """
    Main training function.

    Args:
        args: Parsed command-line arguments
    """
    print("=" * 80)
    print("LoRA Cassettes - Episode 0 Training")
    print("=" * 80)

    # Create config
    config = TrainingConfig(
        device=args.device,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        adapter_name="episode_0",
        checkpoint_every_n_epochs=args.checkpoint_every,
    )

    # Load data
    episode, pairs = await load_training_data(args.episode, subset=args.subset)

    # Create dataloaders
    encoder, peft_model, tokenizer = initialize_model(config)
    train_loader, val_loader = create_dataloaders(pairs, tokenizer, config)

    # Create output directory
    output_dir = Path(config.output_dir) / config.adapter_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/6] Output directory: {output_dir}")

    # Initialize trainer
    print(f"\n[5/6] Training...")
    trainer = ContrastiveTrainer(
        model=peft_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
    )

    # Train
    metrics = trainer.train()

    # Save adapter registry
    final_adapter_path = Path(metrics["final_checkpoint"])
    adapter = await save_adapter_registry(episode, final_adapter_path, config, metrics)

    # Print summary
    print("\n" + "=" * 80)
    print("✓ Training Complete!")
    print("=" * 80)
    print(f"\nMetrics:")
    print(f"  Final Train Loss: {metrics['final_train_loss']:.4f}")
    if metrics['final_val_loss']:
        print(f"  Final Val Loss: {metrics['final_val_loss']:.4f}")
        print(f"  Best Val Loss: {metrics['best_val_loss']:.4f}")
    print(f"  Total Time: {metrics['total_time']:.1f}s")
    print(f"  Epochs: {metrics['epochs_completed']}")

    print(f"\nAdapter:")
    print(f"  ID: {adapter.adapter_id}")
    print(f"  Path: {final_adapter_path}")
    print(f"  Size: {sum(f.stat().st_size for f in final_adapter_path.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")

    print(f"\nNext steps:")
    print(f"  1. Test adapter loading:")
    print(f"     PYTHONPATH=. uv run python -c \"from src.model.encoder import LoRAEncoderRetriever; \\")
    print(f"       enc = LoRAEncoderRetriever('src/model/base_models/e5-base-v2'); \\")
    print(f"       enc.load_adapter('{final_adapter_path}'); \\")
    print(f"       print(enc.encode(['test']).shape)\"")
    print(f"  2. Implement retrieval pipeline")
    print(f"  3. Create evaluation queries")
    print(f"  4. Run baseline evaluation")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Episode 0 LoRA adapter")

    # Data
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode number to train on (default: 0)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use only first N pairs (for testing). Default: use all pairs",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size (default: 512 for M3 Max)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to train on (default: mps for M3 Max)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5). 0 = only save final",
    )

    args = parser.parse_args()

    # Run training
    asyncio.run(main(args))
