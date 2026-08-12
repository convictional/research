"""
Contrastive learning with InfoNCE loss.

Implements the training loop for LoRA adapter fine-tuning using contrastive learning
with in-batch negatives.
"""

import math
import os
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig


class InfoNCE(nn.Module):
    """
    InfoNCE loss for contrastive learning.

    Given a batch of (anchor, positive) pairs, treats each pair as a positive
    and all other pairs in the batch as negatives.

    Based on: https://arxiv.org/abs/1807.03748
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling (lower = harder negatives)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor_embeddings: [batch_size, embedding_dim]
            positive_embeddings: [batch_size, embedding_dim]

        Returns:
            Scalar loss value
        """
        batch_size = anchor_embeddings.shape[0]

        # Normalize embeddings (L2 norm)
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Compute similarity matrix: [batch_size, batch_size]
        # similarity[i, j] = cosine similarity between anchor[i] and positive[j]
        similarity = torch.mm(anchor_embeddings, positive_embeddings.t()) / self.temperature

        # Labels: diagonal elements are positives (anchor[i] matches positive[i])
        labels = torch.arange(batch_size, device=similarity.device)

        # Cross-entropy loss: anchor[i] should match positive[i], not positive[j!=i]
        loss = F.cross_entropy(similarity, labels)

        return loss


class ContrastiveTrainer:
    """
    Trainer for LoRA adapter using contrastive learning.

    Handles training loop, validation, checkpointing, and metrics tracking.
    Optimized for M3 Max with MPS acceleration.
    """

    def __init__(
        self,
        model: PeftModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainingConfig,
        output_dir: Path,
    ):
        """
        Initialize trainer.

        Args:
            model: PEFT model with LoRA adapters
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            config: Training configuration
            output_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Loss function
        self.criterion = InfoNCE(temperature=config.temperature)

        # Optimizer (only LoRA parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler (cosine with warmup)
        num_training_steps = len(train_loader) * config.max_epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        self.scheduler = self._get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Mixed precision scaler
        self.use_amp = config.use_mixed_precision and config.device in ["mps", "cuda"]
        self.scaler = torch.amp.GradScaler(self.device) if self.use_amp else None

        # Metrics
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Training steps: {num_training_steps}")
        print(f"  Warmup steps: {num_warmup_steps}")

    @staticmethod
    def _get_cosine_schedule_with_warmup(
        optimizer: AdamW,
        num_warmup_steps: int,
        num_training_steps: int,
    ) -> LambdaLR:
        """
        Create cosine learning rate schedule with warmup.

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps

        Returns:
            LambdaLR scheduler
        """

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(math.pi * progress))))

        return LambdaLR(optimizer, lr_lambda)

    def _encode_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of text using the model.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, hidden_size]
        """
        # Forward pass through model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling (same as LoRAEncoderRetriever)
        token_embeddings = outputs[0]  # [batch_size, seq_len, hidden_size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        return embeddings

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.max_epochs}",
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            anchor_ids = batch["anchor_input_ids"].to(self.device)
            anchor_mask = batch["anchor_attention_mask"].to(self.device)
            positive_ids = batch["positive_input_ids"].to(self.device)
            positive_mask = batch["positive_attention_mask"].to(self.device)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                # Encode anchors and positives
                anchor_emb = self._encode_batch(anchor_ids, anchor_mask)
                positive_emb = self._encode_batch(positive_ids, positive_mask)

                # Compute loss
                loss = self.criterion(anchor_emb, positive_emb)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Clear MPS cache to prevent memory leak (Apple Silicon specific)
            if self.device.type == "mps":
                torch.mps.empty_cache()

            # Optimizer step (every gradient_accumulation_steps)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    # Optimizer step
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Track loss
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar with memory stats
            avg_loss = total_loss / num_batches
            postfix = {
                "loss": f"{avg_loss:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            }

            # Add memory monitoring
            if self.device.type == "mps":
                # MPS (GPU) memory
                mps_mem = torch.mps.current_allocated_memory() / 1024**3  # GB
                postfix["mps"] = f"{mps_mem:.1f}G"

            # System RAM (main process + DataLoader workers)
            process = psutil.Process(os.getpid())
            main_mem = process.memory_info().rss

            # Include DataLoader worker processes (spawned as children)
            children_mem = sum(
                child.memory_info().rss
                for child in process.children(recursive=True)
                if child.is_running()
            )

            total_mem = (main_mem + children_mem) / 1024**3  # GB
            postfix["ram"] = f"{total_mem:.1f}G"

            progress_bar.set_postfix(postfix)

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate on validation set.

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Move to device
            anchor_ids = batch["anchor_input_ids"].to(self.device)
            anchor_mask = batch["anchor_attention_mask"].to(self.device)
            positive_ids = batch["positive_input_ids"].to(self.device)
            positive_mask = batch["positive_attention_mask"].to(self.device)

            # Forward pass
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                anchor_emb = self._encode_batch(anchor_ids, anchor_mask)
                positive_emb = self._encode_batch(positive_ids, positive_mask)
                loss = self.criterion(anchor_emb, positive_emb)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def save_checkpoint(self, epoch: int) -> Path:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter only (not the full model)
        self.model.save_pretrained(checkpoint_dir)

        return checkpoint_dir

    def train(self) -> dict:
        """
        Run full training loop.

        Returns:
            Dictionary with training metrics and paths
        """
        print("\n" + "=" * 80)
        print(f"Starting training: {self.config.adapter_name}")
        print("=" * 80)
        print(self.config)
        print("=" * 80 + "\n")

        start_time = time.time()

        for epoch in range(1, self.config.max_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate() if self.val_loader else None

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{self.config.max_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"{'='*80}\n")

            # Save checkpoint
            if self.config.checkpoint_every_n_epochs > 0 and epoch % self.config.checkpoint_every_n_epochs == 0:
                checkpoint_path = self.save_checkpoint(epoch)
                print(f"✓ Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_path = self.output_dir / "final"
        self.model.save_pretrained(final_path)
        print(f"\n✓ Saved final adapter: {final_path}")

        total_time = time.time() - start_time

        return {
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "best_val_loss": min(self.val_losses) if self.val_losses else None,
            "total_time": total_time,
            "epochs_completed": self.config.max_epochs,
            "final_checkpoint": str(final_path),
        }
