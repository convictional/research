"""Classification-based authorship verification models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import FeatureFusionEncoder, SimpleEncoder, AttentionEncoder
from .heads import AuthorClassificationHead


class AuthorClassifier(nn.Module):
    """Author classification model for pretraining embeddings."""

    def __init__(
        self,
        encoder_type: str = "fusion",
        semantic_dim: int = 768,
        style_dim: int = 1169,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        num_authors: int = 1000,
        dropout_rate: float = 0.2,
        head_type: str = "arcface",
        margin_s: float = 30.0,
        margin_m: float = 0.30,
    ):
        """
        Initialize author classifier.

        Args:
            encoder_type: Type of encoder ("fusion", "simple", "attention")
            semantic_dim: Semantic feature dimension
            style_dim: Style feature dimension
            hidden_dim: Hidden dimension for encoder
            embedding_dim: Final embedding dimension
            num_authors: Number of author classes
            dropout_rate: Dropout rate
            head_type: Classification head type ("arcface", "cosface", "linear")
            margin_s: Scale parameter for margin heads
            margin_m: Margin parameter
        """
        super().__init__()

        self.encoder_type = encoder_type
        self.num_authors = num_authors
        self.embedding_dim = embedding_dim

        # Initialize encoder
        if encoder_type == "fusion":
            self.encoder = FeatureFusionEncoder(
                semantic_dim=semantic_dim,
                style_dim=style_dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                dropout_rate=dropout_rate,
                normalize_output=True,  # Always normalize for classification
            )
        elif encoder_type == "simple":
            total_dim = semantic_dim + style_dim
            self.encoder = SimpleEncoder(
                input_dim=total_dim,
                hidden_dims=[hidden_dim, hidden_dim // 2],
                output_dim=embedding_dim,
                dropout_rate=dropout_rate,
                normalize_output=True,
            )
        elif encoder_type == "attention":
            self.encoder = AttentionEncoder(
                semantic_dim=semantic_dim,
                style_dim=style_dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                dropout_rate=dropout_rate,
                normalize_output=True,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Classification head
        self.classifier_head = AuthorClassificationHead(
            input_dim=embedding_dim,
            num_authors=num_authors,
            head_type=head_type,
            s=margin_s,
            m=margin_m,
            dropout_rate=dropout_rate,
        )

    def forward(
        self, semantic_features: torch.Tensor, style_features: torch.Tensor, author_ids: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through encoder and classifier.

        Args:
            semantic_features: Semantic features [batch_size, semantic_dim]
            style_features: Style features [batch_size, style_dim]
            author_ids: Author class labels [batch_size] (required for margin heads)

        Returns:
            dictionary containing embeddings and logits
        """
        # Encode features
        if self.encoder_type == "simple":
            combined_features = torch.cat([semantic_features, style_features], dim=1)
            embeddings = self.encoder(combined_features)
        else:
            embeddings = self.encoder(semantic_features, style_features)

        result = {"embeddings": embeddings}

        # Classification
        if author_ids is not None:
            logits = self.classifier_head(embeddings, author_ids)
            result["logits"] = logits
        else:
            # For inference, return embeddings only
            result["embeddings"] = self.classifier_head.get_embeddings(embeddings)

        return result

    def get_embeddings(self, semantic_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """
        Get normalized embeddings for verification.

        Args:
            semantic_features: Semantic features
            style_features: Style features

        Returns:
            L2-normalized embeddings
        """
        with torch.no_grad():
            if self.encoder_type == "simple":
                combined_features = torch.cat([semantic_features, style_features], dim=1)
                embeddings = self.encoder(combined_features)
            else:
                embeddings = self.encoder(semantic_features, style_features)

            return self.classifier_head.get_embeddings(embeddings)


class TwoStageAuthorshipVerifier(nn.Module):
    """Two-stage model: classification pretraining + verification fine-tuning."""

    def __init__(
        self, classifier: AuthorClassifier, verification_head: Optional[nn.Module] = None, freeze_encoder: bool = False
    ):
        """
        Initialize two-stage verifier.

        Args:
            classifier: Pretrained author classifier
            verification_head: Optional verification head for fine-tuning
            freeze_encoder: Whether to freeze encoder during fine-tuning
        """
        super().__init__()

        self.classifier = classifier
        self.verification_head = verification_head
        self.freeze_encoder = freeze_encoder

        # Optionally freeze encoder parameters
        if freeze_encoder:
            for param in self.classifier.encoder.parameters():
                param.requires_grad = False

    def forward_classification(
        self, semantic_features: torch.Tensor, style_features: torch.Tensor, author_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass for classification training."""
        return self.classifier(semantic_features, style_features, author_ids)

    def forward_verification(
        self,
        semantic_features1: torch.Tensor,
        style_features1: torch.Tensor,
        semantic_features2: torch.Tensor,
        style_features2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for verification."""
        # Get embeddings from classifier
        emb1 = self.classifier.get_embeddings(semantic_features1, style_features1)
        emb2 = self.classifier.get_embeddings(semantic_features2, style_features2)

        result = {"embedding1": emb1, "embedding2": emb2, "similarity": F.cosine_similarity(emb1, emb2, dim=1)}

        # Optional verification-specific processing
        if self.verification_head is not None:
            # Could add specialized heads for verification fine-tuning
            pass

        return result

    def compute_verification_loss(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
        loss_type: str = "cosine_hinge",
    ) -> torch.Tensor:
        """
        Compute verification loss.

        Args:
            embeddings1: First embeddings
            embeddings2: Second embeddings
            labels: Binary labels (1=same, 0=different)
            loss_type: Type of loss function

        Returns:
            Verification loss
        """
        if loss_type == "cosine_hinge":
            from .losses import CosineSimilarityLoss

            loss_fn = CosineSimilarityLoss(pos_margin=0.0, neg_margin=0.2)
            return loss_fn(embeddings1, embeddings2, labels)
        elif loss_type == "contrastive":
            from .losses import ContrastiveLoss

            loss_fn = ContrastiveLoss(margin=1.0)
            return loss_fn(embeddings1, embeddings2, labels)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


def create_classification_model(config) -> AuthorClassifier:
    """Create classification model from config."""
    return AuthorClassifier(
        encoder_type=config.model.encoder_type,
        semantic_dim=config.model.embedding_dim,
        style_dim=config.model.style_feature_dim,
        hidden_dim=config.model.hidden_dim,
        embedding_dim=config.model.final_embedding_dim,
        num_authors=getattr(config.model, "num_authors", 1000),
        dropout_rate=config.model.dropout_rate,
        head_type=getattr(config.model, "head_type", "arcface"),
        margin_s=getattr(config.model, "margin_s", 30.0),
        margin_m=getattr(config.model, "margin_m", 0.30),
    )


def create_two_stage_model(classifier: AuthorClassifier, freeze_encoder: bool = False) -> TwoStageAuthorshipVerifier:
    """Create two-stage model from pretrained classifier."""
    return TwoStageAuthorshipVerifier(classifier=classifier, freeze_encoder=freeze_encoder)
