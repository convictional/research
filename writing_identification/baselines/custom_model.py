"""Custom pre-trained Siamese model baseline for authorship verification."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines.base import BaselineModel
from features.extractors import SemanticFeatureExtractor
from models.siamese import SiameseNetwork

logger = logging.getLogger(__name__)


class CustomModelBaseline(BaselineModel):
    """Baseline using our custom pre-trained Siamese network."""

    def __init__(self, checkpoint_path: str):
        """Initialize the custom model baseline.

        Args:
            checkpoint_path: Path to model checkpoint. Required.
        """
        super().__init__("Custom Siamese Network")

        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for CustomModelBaseline")

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize feature extractors first (may be overridden by checkpoint)
        self.semantic_extractor = SemanticFeatureExtractor()
        self.style_extractor = None  # Will be loaded from checkpoint
        self.email_extractor = None  # May be loaded from checkpoint

        # Store whether this model uses email features (determined from checkpoint)
        self.use_email_features = False

        # Load checkpoint (may override extractors)
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load model and feature extractors from checkpoint."""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Get configuration from checkpoint
        if "config" in checkpoint:
            config = checkpoint["config"]
            model_config = config.get("model", {})
            encoder_type = model_config.get("encoder_type", "fusion")
            semantic_dim = model_config.get("embedding_dim", 768)  # Note: embedding_dim is semantic dim
            # Style dim from the config - DON'T add email features as they weren't used in this checkpoint
            style_dim = model_config.get("style_feature_dim", 1169)
            hidden_dim = model_config.get("hidden_dim", 512)
            output_dim = model_config.get("final_embedding_dim", 256)
            dropout_rate = model_config.get("dropout_rate", 0.2)
            normalize_embeddings = True  # Default to True as it's not in the config
        else:
            # Use default values if config not in checkpoint
            logger.warning("Config not found in checkpoint, using defaults")
            encoder_type = "fusion"
            semantic_dim = 768
            style_dim = 1207  # 1169 style + 38 email features
            hidden_dim = 512
            output_dim = 256
            dropout_rate = 0.2
            normalize_embeddings = True

        # Initialize Siamese network
        self.model = SiameseNetwork(
            encoder_type=encoder_type,
            semantic_dim=semantic_dim,
            style_dim=style_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            normalize_embeddings=normalize_embeddings,
        )

        # Load model state
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Handle two-stage training checkpoint with 'siamese_network.' prefix
            if any(key.startswith("siamese_network.") for key in state_dict.keys()):
                # Remove the 'siamese_network.' prefix
                state_dict = {
                    k.replace("siamese_network.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("siamese_network.")
                }
            self.model.load_state_dict(state_dict)
        else:
            # Old checkpoint format
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Load extractors from checkpoint
        if "style_extractor" in checkpoint and checkpoint["style_extractor"] is not None:
            logger.info("Loading fitted extractors from checkpoint")
            self.style_extractor = checkpoint["style_extractor"]
            self.semantic_extractor = checkpoint.get("semantic_extractor", self.semantic_extractor)
            email_ext = checkpoint.get("email_extractor", None)
            if email_ext is not None:
                self.email_extractor = email_ext
        elif "style_extractor_state" in checkpoint:  # Old checkpoint format
            logger.info("Loading fitted style extractor from old checkpoint format")
            self.style_extractor = checkpoint["style_extractor_state"]
        else:
            # CRITICAL ERROR - can't do inference without extractors!
            raise ValueError(
                "No fitted extractors found in checkpoint! "
                "This model checkpoint cannot be used for inference. "
                "Please retrain with the updated training script that saves extractors."
            )

        # Store config for later use
        self.config = checkpoint.get("config", None)

        # Load the optimal threshold from training
        self.optimal_threshold = checkpoint.get("best_threshold", None)

        # If not found at top level, try to extract from training history
        if self.optimal_threshold is None and "training_history" in checkpoint:
            saved_epoch = checkpoint.get("epoch", None)
            if saved_epoch is not None:
                # Find the threshold for this epoch in training history
                for entry in checkpoint["training_history"]:
                    if entry.get("epoch") == saved_epoch:
                        self.optimal_threshold = entry.get("val_threshold", None)
                        if self.optimal_threshold:
                            logger.info(f"Extracted threshold from training history: {self.optimal_threshold:.4f}")
                        break

        logger.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
        if "best_metric" in checkpoint:
            logger.info(f"Best validation metric: {checkpoint['best_metric']:.4f}")
        if self.optimal_threshold is not None:
            logger.info(f"Using saved optimal threshold: {self.optimal_threshold:.4f}")
        else:
            logger.warning("No optimal threshold found - will use calibration data for threshold selection")

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Args:
            text: Input text to encode

        Returns:
            Embedding vector as numpy array
        """
        # Ensure we have the required extractors
        if self.style_extractor is None:
            raise ValueError("Style extractor not loaded from checkpoint - cannot perform inference")

        # Extract features
        with torch.no_grad():
            # Semantic features (768D)
            semantic_features = self.semantic_extractor.extract_features(text)
            if not isinstance(semantic_features, torch.Tensor):
                semantic_features = torch.tensor(semantic_features)

            # Style features
            style_features = self.style_extractor.extract_features(text)
            if not isinstance(style_features, torch.Tensor):
                style_features = torch.tensor(style_features)

            # Check if we need to add email features
            # The model expects 1169 features, but style extractor may only produce 1131
            # The difference of 38 is exactly the email feature dimension
            if style_features.shape[0] == 1131 and hasattr(self.model.encoder, "style_proj"):
                # Model expects email features but they're not in the extractor
                # Add zero-padding for the missing email features
                email_padding = torch.zeros(38)
                combined_style = torch.cat([style_features, email_padding])
                logger.debug(
                    f"Added zero-padding for missing email features: {style_features.shape[0]} -> {combined_style.shape[0]}"
                )
            else:
                combined_style = style_features

            # Add batch dimension
            semantic_features = semantic_features.unsqueeze(0).to(self.device)
            combined_style = combined_style.unsqueeze(0).to(self.device)

            # Get embedding from model
            embedding = self.model.forward_one(semantic_features, combined_style)

            # Convert to numpy
            return embedding.cpu().numpy().squeeze()

    def has_saved_threshold(self) -> bool:
        """Check if this model has a saved optimal threshold from training."""
        return self.optimal_threshold is not None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine") -> float:
        """Compute similarity between two embeddings.

        Override to ensure consistency with training approach.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric (always uses cosine for consistency)

        Returns:
            Similarity score
        """
        # Always use cosine similarity to match training
        embedding1_t = torch.tensor(embedding1).unsqueeze(0)
        embedding2_t = torch.tensor(embedding2).unsqueeze(0)
        similarity = F.cosine_similarity(embedding1_t, embedding2_t, dim=1)
        return similarity.item()
