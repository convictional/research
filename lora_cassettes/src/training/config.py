"""
Training configuration using Pydantic.

Defines all hyperparameters with sensible defaults optimized for M3 Max hardware.
Based on PLAN.md section 6 (Episodic Fine-Tuning).
"""

from pydantic import BaseModel, Field, field_validator


class TrainingConfig(BaseModel):
    """
    Configuration for contrastive LoRA training.

    Optimized for Apple M3 Max with 128GB RAM and MPS acceleration.
    """

    # Hardware
    device: str = Field(
        default="mps",
        description="Device to train on ('mps', 'cuda', 'cpu'). MPS recommended for M3 Max.",
    )
    use_mixed_precision: bool = Field(
        default=True,
        description="Use mixed precision (fp16) for 2x speedup on MPS/CUDA",
    )

    # Data
    batch_size: int = Field(
        default=512,
        description="Batch size. Large due to 128GB RAM. Each batch provides batch_size-1 negatives.",
        gt=0,
    )
    num_workers: int = Field(
        default=8,
        description="DataLoader worker processes for parallel data loading",
        ge=0,
    )
    max_seq_length: int = Field(
        default=512,
        description="Maximum sequence length for tokenization",
        gt=0,
    )
    validation_split: float = Field(
        default=0.1,
        description="Fraction of data to use for validation (0.1 = 10%)",
        gt=0,
        lt=1,
    )

    # Training
    max_epochs: int = Field(
        default=20,
        description="Maximum number of training epochs",
        gt=0,
    )
    learning_rate: float = Field(
        default=2e-4,
        description="Learning rate for AdamW optimizer (PLAN.md section 6)",
        gt=0,
    )
    weight_decay: float = Field(
        default=0.01,
        description="Weight decay for AdamW optimizer (PLAN.md section 6)",
        ge=0,
    )
    gradient_accumulation_steps: int = Field(
        default=2,
        description="Accumulate gradients over N steps. Logical batch = batch_size * N",
        gt=0,
    )
    warmup_ratio: float = Field(
        default=0.05,
        description="Fraction of training for learning rate warmup (5% per PLAN.md)",
        ge=0,
        le=1,
    )
    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping (prevents exploding gradients)",
        gt=0,
    )

    # LoRA (from PLAN.md section 6)
    lora_r: int = Field(
        default=16,
        description="LoRA rank (PLAN.md: r=16)",
        gt=0,
    )
    lora_alpha: int = Field(
        default=32,
        description="LoRA alpha parameter (PLAN.md: α=32)",
        gt=0,
    )
    lora_dropout: float = Field(
        default=0.05,
        description="LoRA dropout rate (PLAN.md: 0.05)",
        ge=0,
        le=1,
    )

    # Contrastive Learning
    temperature: float = Field(
        default=0.07,
        description="Temperature for contrastive loss. Lower = harder negatives (PLAN.md: 0.05-0.1)",
        gt=0,
    )

    # Checkpointing
    checkpoint_every_n_epochs: int = Field(
        default=5,
        description="Save checkpoint every N epochs. 0 = only save final",
        ge=0,
    )
    keep_last_n_checkpoints: int = Field(
        default=3,
        description="Keep only the last N checkpoints to save disk space",
        gt=0,
    )

    # Output
    output_dir: str = Field(
        default="src/model/adapters",
        description="Directory to save adapter checkpoints",
    )
    adapter_name: str = Field(
        default="episode_0",
        description="Name for this adapter (e.g., 'episode_0', 'episode_1')",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device is one of the supported options."""
        if v not in ["mps", "cuda", "cpu"]:
            raise ValueError(f"Device must be one of ['mps', 'cuda', 'cpu'], got '{v}'")
        return v

    @property
    def logical_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def num_negatives(self) -> int:
        """Number of negative examples per anchor (in-batch negatives)."""
        return self.batch_size - 1

    def __str__(self) -> str:
        """Pretty print configuration."""
        return (
            f"TrainingConfig(\n"
            f"  Device: {self.device}\n"
            f"  Batch size: {self.batch_size} (logical: {self.logical_batch_size})\n"
            f"  Epochs: {self.max_epochs}\n"
            f"  Learning rate: {self.learning_rate}\n"
            f"  LoRA: r={self.lora_r}, α={self.lora_alpha}, dropout={self.lora_dropout}\n"
            f"  Temperature: {self.temperature}\n"
            f"  Negatives per sample: {self.num_negatives}\n"
            f")"
        )
