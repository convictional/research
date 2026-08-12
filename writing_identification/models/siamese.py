"""Siamese network for authorship verification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import FeatureFusionEncoder, SimpleEncoder, AttentionEncoder
from .losses import ContrastiveLoss, CosineSimilarityLoss


class SiameseNetwork(nn.Module):
    """Siamese network for authorship verification."""

    def __init__(
        self,
        encoder_type: str = "fusion",
        semantic_dim: int = 384,
        style_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout_rate: float = 0.2,
        normalize_embeddings: bool = True,
    ):
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == "fusion":
            self.encoder = FeatureFusionEncoder(
                semantic_dim=semantic_dim,
                style_dim=style_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout_rate=dropout_rate,
                normalize_output=normalize_embeddings,
            )
        elif encoder_type == "simple":
            total_dim = semantic_dim + style_dim
            self.encoder = SimpleEncoder(
                input_dim=total_dim,
                hidden_dims=[hidden_dim, hidden_dim // 2],
                output_dim=output_dim,
                dropout_rate=dropout_rate,
                normalize_output=normalize_embeddings,
            )
        elif encoder_type == "attention":
            self.encoder = AttentionEncoder(
                semantic_dim=semantic_dim,
                style_dim=style_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout_rate=dropout_rate,
                normalize_output=normalize_embeddings,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Move to MPS if available
        if torch.backends.mps.is_available():
            self.to(torch.device("mps"))

    def forward_one(self, semantic_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for one input."""
        if self.encoder_type == "simple":
            # Concatenate features for simple encoder
            combined_features = torch.cat([semantic_features, style_features], dim=1)
            return self.encoder(combined_features)
        else:
            # Separate semantic and style inputs for fusion/attention encoders
            return self.encoder(semantic_features, style_features)

    def forward(
        self,
        semantic_features1: torch.Tensor,
        style_features1: torch.Tensor,
        semantic_features2: torch.Tensor,
        style_features2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Siamese network.

        Args:
            semantic_features1: First text semantic features
            style_features1: First text style features
            semantic_features2: Second text semantic features
            style_features2: Second text style features

        Returns:
            Tuple of (embedding1, embedding2)
        """
        embedding1 = self.forward_one(semantic_features1, style_features1)
        embedding2 = self.forward_one(semantic_features2, style_features2)

        return embedding1, embedding2

    def similarity_from_embeddings(self, e1: torch.Tensor, e2: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
        """
        Compute similarity from precomputed embeddings.

        Args:
            e1: First embeddings
            e2: Second embeddings
            metric: Similarity metric ("cosine" or "euclidean")

        Returns:
            Similarity scores
        """
        if metric == "cosine":
            return F.cosine_similarity(e1, e2, dim=1)
        elif metric == "euclidean":
            distance = F.pairwise_distance(e1, e2)
            return 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def compute_similarity(
        self,
        semantic_features1: torch.Tensor,
        style_features1: torch.Tensor,
        semantic_features2: torch.Tensor,
        style_features2: torch.Tensor,
        metric: str = "cosine",
    ) -> torch.Tensor:
        """
        Compute similarity between two texts.

        Args:
            semantic_features1: First text semantic features
            style_features1: First text style features
            semantic_features2: Second text semantic features
            style_features2: Second text style features
            metric: Similarity metric ("cosine" or "euclidean")

        Returns:
            Similarity scores
        """
        embedding1, embedding2 = self.forward(semantic_features1, style_features1, semantic_features2, style_features2)

        return self.similarity_from_embeddings(embedding1, embedding2, metric)


class AuthorshipVerifier(nn.Module):
    """Complete authorship verification system."""

    def __init__(self, siamese_network: SiameseNetwork, loss_type: str = "contrastive", margin: float = 1.0):
        super().__init__()

        self.siamese_network = siamese_network
        self.loss_type = loss_type

        # Initialize loss function
        if loss_type == "contrastive":
            self.loss_fn = ContrastiveLoss(margin=margin)
        elif loss_type == "cosine":
            self.loss_fn = CosineSimilarityLoss(margin=margin)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported: 'contrastive', 'cosine'")

        # Move to MPS if available
        if torch.backends.mps.is_available():
            self.to(torch.device("mps"))

    def forward(
        self,
        semantic_features1: torch.Tensor,
        style_features1: torch.Tensor,
        semantic_features2: torch.Tensor,
        style_features2: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass with loss computation.

        Args:
            semantic_features1: First text semantic features
            style_features1: First text style features
            semantic_features2: Second text semantic features
            style_features2: Second text style features
            labels: Ground truth labels (for training)

        Returns:
            Dictionary with embeddings, similarities, and loss (if labels provided)
        """
        # Get embeddings
        embedding1, embedding2 = self.siamese_network.forward(
            semantic_features1, style_features1, semantic_features2, style_features2
        )

        # Compute similarity from embeddings (no duplicate forward pass)
        similarity = self.siamese_network.similarity_from_embeddings(embedding1, embedding2, metric="cosine")

        result = {"embedding1": embedding1, "embedding2": embedding2, "similarity": similarity}

        # Compute loss if labels are provided
        if labels is not None:
            # Use the configured loss function (contrastive or cosine similarity)
            # Note: triplet loss is not suitable for pair-based training
            loss = self.loss_fn(embedding1, embedding2, labels)

            # Add uniformity regularization only for true negatives
            batch_size = embedding1.shape[0]
            if batch_size > 1:
                # Get all embeddings for uniformity computation
                all_emb = torch.cat([embedding1, embedding2], dim=0)  # [2*B, D]
                all_emb = F.normalize(all_emb, p=2, dim=1)

                # Compute pairwise similarities
                sim_matrix = all_emb @ all_emb.t()  # [2*B, 2*B]

                # Create mask for true negatives only
                # Expand labels to cover all pairwise combinations
                expanded_labels = labels.unsqueeze(1).expand(batch_size, batch_size)
                # Different authors (0) become True negatives
                neg_mask = expanded_labels == 0

                # Only penalize similarities between clear negatives
                if neg_mask.sum() > 0:
                    neg_sims = sim_matrix[:batch_size, :batch_size][neg_mask]
                    uniformity_loss = 0.001 * (neg_sims**2).mean()  # Very small weight
                    loss = loss + uniformity_loss

            result["loss"] = loss

        return result

    def predict(
        self,
        semantic_features1: torch.Tensor,
        style_features1: torch.Tensor,
        semantic_features2: torch.Tensor,
        style_features2: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Make authorship predictions.

        Args:
            semantic_features1: First text semantic features
            style_features1: First text style features
            semantic_features2: Second text semantic features
            style_features2: Second text style features
            threshold: Decision threshold

        Returns:
            Tuple of (predictions, similarities)
        """
        self.eval()
        with torch.no_grad():
            embedding1, embedding2 = self.siamese_network.forward(
                semantic_features1, style_features1, semantic_features2, style_features2
            )
            similarity = self.siamese_network.similarity_from_embeddings(embedding1, embedding2, metric="cosine")

            predictions = (similarity > threshold).float()

        return predictions, similarity
