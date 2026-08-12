"""Classification heads for authorship verification."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """Angular margin product (ArcFace) for face recognition and authorship verification."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.30, easy_margin: bool = False):
        """
        Initialize ArcMarginProduct head.

        Args:
            in_features: Size of input embedding dimension
            out_features: Number of classes (authors)
            s: Scale parameter (typically 30-64)
            m: Angular margin parameter (typically 0.2-0.5)
            easy_margin: Whether to use easy margin formulation
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Learnable class weights (to be normalized)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cosine and sine of margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # threshold = cos(π - m)
        self.mm = math.sin(math.pi - m) * m  # mm = sin(π - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with angular margin.

        Args:
            embeddings: L2-normalized input embeddings [batch_size, in_features]
            labels: Ground truth class labels [batch_size]

        Returns:
            Logits with angular margin applied [batch_size, out_features]
        """
        # L2 normalize input embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity (embeddings · weights)
        cosine = F.linear(embeddings, weight)  # [batch_size, out_features]
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)  # Numerical stability

        # Compute sine from cosine for angle addition formula
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Apply angular margin: cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply easy margin or hard margin
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert labels to one-hot
        one_hot = F.one_hot(labels, num_classes=self.out_features).float()

        # Apply margin to target class, keep original cosine for others
        output = one_hot * phi + (1.0 - one_hot) * cosine

        # Apply scale
        output = output * self.s

        return output


class CosFaceProduct(nn.Module):
    """Cosine face margin product (CosFace) alternative to ArcFace."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        """
        Initialize CosFaceProduct head.

        Args:
            in_features: Size of input embedding dimension
            out_features: Number of classes (authors)
            s: Scale parameter
            m: Cosine margin parameter
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cosine margin.

        Args:
            embeddings: L2-normalized input embeddings [batch_size, in_features]
            labels: Ground truth class labels [batch_size]

        Returns:
            Logits with cosine margin applied [batch_size, out_features]
        """
        # L2 normalize input embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)

        # Convert labels to one-hot
        one_hot = F.one_hot(labels, num_classes=self.out_features).float()

        # Apply margin to target class: cos(θ) - m
        output = cosine - one_hot * self.m

        # Apply scale
        output = output * self.s

        return output


class AuthorClassificationHead(nn.Module):
    """Complete classification head with optional projection and margin."""

    def __init__(
        self,
        input_dim: int,
        num_authors: int,
        embedding_dim: int = None,
        head_type: str = "arcface",
        s: float = 30.0,
        m: float = 0.30,
        dropout_rate: float = 0.1
    ):
        """
        Initialize classification head.

        Args:
            input_dim: Input feature dimension
            num_authors: Number of author classes
            embedding_dim: Optional embedding dimension (None = use input_dim)
            head_type: Type of margin head ("arcface", "cosface", "linear")
            s: Scale parameter for margin heads
            m: Margin parameter
            dropout_rate: Dropout rate before classification
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_authors = num_authors
        self.embedding_dim = embedding_dim or input_dim
        self.head_type = head_type

        # Optional projection layer
        if embedding_dim and embedding_dim != input_dim:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.projection = None

        # Classification head
        if head_type == "arcface":
            self.classifier = ArcMarginProduct(self.embedding_dim, num_authors, s=s, m=m)
        elif head_type == "cosface":
            self.classifier = CosFaceProduct(self.embedding_dim, num_authors, s=s, m=m)
        elif head_type == "linear":
            self.classifier = nn.Linear(self.embedding_dim, num_authors)
        else:
            raise ValueError(f"Unknown head type: {head_type}")

        # Dropout before classification
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through classification head.

        Args:
            features: Input features [batch_size, input_dim]
            labels: Class labels for margin computation (required for margin heads)

        Returns:
            Logits or embeddings [batch_size, num_authors or embedding_dim]
        """
        # Optional projection
        if self.projection is not None:
            features = self.projection(features)

        # L2 normalize features for margin heads
        if self.head_type in ["arcface", "cosface"]:
            features = F.normalize(features, p=2, dim=1)

        # Dropout
        features = self.dropout(features)

        # Classification
        if self.head_type in ["arcface", "cosface"] and labels is not None:
            return self.classifier(features, labels)
        else:
            return self.classifier(features) if self.head_type == "linear" else features

    def get_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (before classification layer).

        Args:
            features: Input features

        Returns:
            L2-normalized embeddings
        """
        if self.projection is not None:
            features = self.projection(features)

        return F.normalize(features, p=2, dim=1)
