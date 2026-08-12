"""ModernBERT baseline model for authorship verification."""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaselineModel

logger = logging.getLogger(__name__)


class ModernBERTBaseline(BaselineModel):
    """ModernBERT baseline model for authorship verification."""

    def __init__(self, model_name: str = "gabrielloiseau/ModernBERT-base-authorship-verification"):
        """Initialize ModernBERT model.

        Args:
            model_name: HuggingFace model identifier
        """
        super().__init__("ModernBERT")
        self.model_name_hf = model_name

        # Initialize model
        self._load_model()

    def _load_model(self):
        """Load the ModernBERT model using sentence-transformers."""
        logger.info(f"Loading ModernBERT model: {self.model_name_hf}")

        try:
            self.model = SentenceTransformer(self.model_name_hf)

            # Get device info
            device = self.model.device
            logger.info(f"ModernBERT model loaded successfully on {device}")
            logger.info(f"Model max sequence length: {self.model.max_seq_length}")
            logger.info(f"Model embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        except Exception as e:
            logger.error(f"Failed to load ModernBERT model: {e}")
            raise

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Args:
            text: Input text to encode

        Returns:
            768-dimensional embedding vector
        """
        try:
            # Use sentence-transformers encode method
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding

        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            # Return zero vector as fallback
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return np.zeros(embedding_dim)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into embedding vectors.

        Args:
            texts: list of input texts to encode

        Returns:
            Batch of 768-dim embedding vectors (n_texts, 768)
        """
        try:
            # Use sentence-transformers batch encoding (more efficient)
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            # Return zero vectors as fallback
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((len(texts), embedding_dim))

    def encode_text_with_prompt(self, text: str, prompt: str = None) -> np.ndarray:
        """Encode text with an optional prompt for better authorship representation.

        Some sentence transformers work better with prompts that specify the task.

        Args:
            text: Input text to encode
            prompt: Optional prompt to prepend (e.g., "Represent this text for authorship analysis:")

        Returns:
            768-dimensional embedding vector
        """
        if prompt:
            prompted_text = f"{prompt} {text}"
        else:
            # Use a generic authorship-focused prompt
            prompted_text = f"Represent this text for authorship verification: {text}"

        return self.encode_text(prompted_text)
