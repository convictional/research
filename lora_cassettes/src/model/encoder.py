"""
Encoder wrapper with LoRA adapter support for retrieval.

Implements the adapter loading/swapping functionality described in PLAN.md section 8.
"""

from contextlib import contextmanager
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer


class LoRAEncoderRetriever:
    """
    Encoder with swappable LoRA adapters for retrieval.

    Based on PLAN.md section 8.2 - uses frozen base model with hot-swappable adapters.
    Adapters are only used for embedding computation, never for generation.
    """

    def __init__(
        self,
        base_model_path: str | Path,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the retriever with a base encoder model.

        Args:
            base_model_path: Path to the base model (e.g., 'src/model/base_models/e5-base-v2')
            device: Device to run the model on ('cpu', 'cuda', 'mps')
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.base_model_path = Path(base_model_path)

        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_path))
        self.base_model = AutoModel.from_pretrained(str(self.base_model_path))
        self.base_model.to(self.device)
        self.base_model.eval()

        # Track currently loaded adapter
        self.current_adapter: str | None = None
        self.peft_model: PeftModel | None = None

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling over token embeddings (e5 style).

        Takes attention mask into account for correct averaging.
        """
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Encode texts into embeddings using the current model (base or with adapter).

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Tensor of shape (len(texts), hidden_size) with embeddings
        """
        all_embeddings = []

        # Use base model or adapter model
        model = self.peft_model if self.peft_model is not None else self.base_model

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Forward pass
                outputs = model(**encoded)

                # Pool to get sentence embeddings
                embeddings = self.mean_pooling(outputs, encoded["attention_mask"])

                # Normalize if requested
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def load_adapter(self, adapter_path: str | Path) -> None:
        """
        Load a LoRA adapter from disk.

        Args:
            adapter_path: Path to the adapter directory (contains adapter_config.json and adapter_model.safetensors)
        """
        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        # Unload current adapter if any
        if self.peft_model is not None:
            self.unload_adapter()

        # Load adapter
        self.peft_model = PeftModel.from_pretrained(
            self.base_model,
            str(adapter_path),
            is_trainable=False,
        )
        self.peft_model.to(self.device)
        self.peft_model.eval()

        self.current_adapter = str(adapter_path)

    def unload_adapter(self) -> None:
        """Unload the current adapter and revert to base model."""
        if self.peft_model is not None:
            # Merge and unload to avoid memory leaks
            self.peft_model = None
            self.current_adapter = None

    @contextmanager
    def use_adapter(self, adapter_path: str | Path):
        """
        Context manager for temporarily using an adapter.

        Usage:
            with encoder.use_adapter('path/to/adapter'):
                embeddings = encoder.encode(texts)

        Args:
            adapter_path: Path to the adapter to use
        """
        previous_adapter = self.current_adapter

        try:
            self.load_adapter(adapter_path)
            yield self
        finally:
            self.unload_adapter()
            if previous_adapter is not None:
                self.load_adapter(previous_adapter)

    def create_lora_config(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> LoraConfig:
        """
        Create a LoRA configuration for training.

        Based on PLAN.md section 6: r=16, α=32, dropout=0.05,
        targeting attention (q,k,v,o) + MLP projections.

        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout rate
            target_modules: Modules to apply LoRA to (defaults to BERT attention + MLPs)

        Returns:
            LoraConfig object
        """
        if target_modules is None:
            # BERT-based models (e5-base-v2 uses BERT architecture)
            target_modules = [
                "query",
                "key",
                "value",
                "dense",  # Output projection in attention
                "intermediate.dense",  # MLP first layer
                "output.dense",  # MLP second layer
            ]

        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,  # Not using any specific task type (we're doing custom contrastive training)
        )

    def prepare_for_training(
        self,
        lora_config: LoraConfig | None = None,
    ) -> PeftModel:
        """
        Prepare the model for LoRA training by wrapping with PEFT.

        Args:
            lora_config: LoRA configuration (uses default if None)

        Returns:
            PeftModel ready for training
        """
        if lora_config is None:
            lora_config = self.create_lora_config()

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Wrap with PEFT
        peft_model = get_peft_model(self.base_model, lora_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return peft_model
