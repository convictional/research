"""Loss functions for Siamese networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embedding1: First embedding (batch_size, embedding_dim)
            embedding2: Second embedding (batch_size, embedding_dim)
            label: Binary labels (1 for same author, 0 for different authors)

        Returns:
            Contrastive loss
        """
        # Compute Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)

        # Contrastive loss formula
        # For same author (label=1): minimize distance
        # For different authors (label=0): maximize distance up to margin
        loss_same = label * torch.pow(distance, 2)
        loss_different = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        loss = 0.5 * (loss_same + loss_different)
        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    """Improved cosine hinge loss with separate positive and negative margins."""

    def __init__(self, pos_margin: float = 0.0, neg_margin: float = 0.2, margin: float = None):
        super().__init__()
        # Support legacy margin parameter for backward compatibility
        if margin is not None:
            self.pos_margin = 0.0
            self.neg_margin = margin
        else:
            self.pos_margin = pos_margin
            self.neg_margin = neg_margin

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine hinge loss.

        Args:
            embedding1: First embedding (batch_size, embedding_dim)
            embedding2: Second embedding (batch_size, embedding_dim)
            label: Binary labels (1 for same author, 0 for different authors)

        Returns:
            Cosine hinge loss
        """
        # Ensure labels are float
        label = label.float()

        # L2 normalize embeddings for stable cosine computation
        embedding1 = F.normalize(embedding1, p=2, dim=-1)
        embedding2 = F.normalize(embedding2, p=2, dim=-1)

        # Compute cosine similarity
        cosine_sim = (embedding1 * embedding2).sum(dim=-1)

        # Hinge losses with separate margins
        # Positive: push above pos_margin (0 by default)
        pos_loss = label * F.relu(self.pos_margin - cosine_sim)
        # Negative: push below neg_margin
        neg_loss = (1 - label) * F.relu(cosine_sim - self.neg_margin)

        return (pos_loss + neg_loss).mean()
