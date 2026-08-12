"""LUAR (Learning Universal Authorship Representations) baseline model."""

import logging
from typing import Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaselineModel

logger = logging.getLogger(__name__)


class LUARBaseline(BaselineModel):
    """LUAR baseline model for authorship verification."""

    def __init__(self, model_name: str = "rrivera1849/LUAR-CRUD", episode_length: int = 16):
        """Initialize LUAR model.

        Args:
            model_name: HuggingFace model identifier
            episode_length: Number of text samples per episode (LUAR requirement)
        """
        super().__init__("LUAR")
        self.model_name_hf = model_name
        self.episode_length = episode_length
        self.max_token_length = 32  # LUAR's fixed token length

        # Initialize model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the LUAR model and tokenizer from HuggingFace."""
        logger.info(f"Loading LUAR model: {self.model_name_hf}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name_hf, trust_remote_code=True)
            self.model.eval()  # Set to evaluation mode

            # Move to appropriate device
            # LUAR has compatibility issues with MPS, so use CPU fallback
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                logger.warning("Using CPU for LUAR due to MPS compatibility issues")

            self.model.to(self.device)
            logger.info(f"LUAR model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load LUAR model: {e}")
            raise

    def _prepare_episodic_input(self, text: str, episode_length: Optional[int] = None) -> list[str]:
        """Prepare text for LUAR's episodic input format.

        LUAR expects multiple text samples per author (episode). For single text evaluation,
        we'll create an "episode" by splitting the text or repeating it.

        Args:
            text: Input text
            episode_length: Length of episode (defaults to self.episode_length)

        Returns:
            list of text fragments forming an episode
        """
        if episode_length is None:
            episode_length = self.episode_length

        # Split text into sentences for more natural episodes
        sentences = text.split(". ")
        if len(sentences) < 2:
            # If no sentence breaks, split by other punctuation or chunks
            sentences = text.split("? ") if "? " in text else text.split("! ") if "! " in text else [text]

        # Create episode by taking sentences
        if len(sentences) >= episode_length:
            # Take evenly spaced sentences
            indices = np.linspace(0, len(sentences) - 1, episode_length).astype(int)
            episode = [sentences[i] for i in indices]
        else:
            # Repeat sentences to reach episode length
            episode = []
            for i in range(episode_length):
                episode.append(sentences[i % len(sentences)])

        # Ensure all texts end with period for consistency
        episode = [sent if sent.endswith(".") else sent + "." for sent in episode]

        return episode

    def _encode_episode(self, episode: list[str]) -> np.ndarray:
        """Encode an episode using LUAR.

        Args:
            episode: list of text samples forming an episode

        Returns:
            512-dimensional embedding vector
        """
        # Tokenize all texts in the episode
        tokenized = self.tokenizer(
            episode, max_length=self.max_token_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Reshape for LUAR's expected input format
        batch_size = 1
        episode_length = len(episode)

        input_ids = tokenized["input_ids"].reshape(batch_size, episode_length, -1)
        attention_mask = tokenized["attention_mask"].reshape(batch_size, episode_length, -1)

        # Move to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Get embedding
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # LUAR outputs a single embedding per batch
            # Convert to CPU first, then to numpy for MPS compatibility
            if hasattr(outputs, "cpu"):
                embedding = outputs.cpu().numpy().flatten()  # Should be 512-dim
            else:
                # If outputs is already a tensor, handle it properly
                embedding = outputs.detach().cpu().numpy().flatten()

        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Args:
            text: Input text to encode

        Returns:
            512-dimensional embedding vector
        """
        # Prepare episodic input
        episode = self._prepare_episodic_input(text)

        # Encode episode
        embedding = self._encode_episode(episode)

        return embedding

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into embedding vectors.

        For LUAR, we process each text individually since each needs its own episode.

        Args:
            texts: list of input texts to encode

        Returns:
            Batch of 512-dim embedding vectors (n_texts, 512)
        """
        embeddings = []

        for text in texts:
            embedding = self.encode_text(text)
            embeddings.append(embedding)

        return np.array(embeddings)
