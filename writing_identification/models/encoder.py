"""Neural network encoders for authorship verification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionEncoder(nn.Module):
    """Encoder that fuses semantic and stylometric features."""

    def __init__(
        self,
        semantic_dim: int = 384,
        style_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout_rate: float = 0.2,
        normalize_output: bool = True
    ):
        super().__init__()

        self.semantic_dim = semantic_dim
        self.style_dim = style_dim
        self.normalize_output = normalize_output

        # Semantic feature processing with layer normalization
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        # Style feature processing with layer normalization
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        # Fusion and final projection with deeper architecture
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),  # Additional layer for capacity
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with improved strategy."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for ReLU networks
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                # Initialize layer norm weights and bias
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, semantic_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            semantic_features: Semantic embeddings (batch_size, semantic_dim)
            style_features: Stylometric features (batch_size, style_dim)

        Returns:
            Fused embeddings (batch_size, output_dim)
        """
        # Process each feature type
        semantic_proj = self.semantic_proj(semantic_features)
        style_proj = self.style_proj(style_features)

        # Concatenate and fuse
        fused = torch.cat([semantic_proj, style_proj], dim=1)
        output = self.fusion(fused)

        # Optionally apply L2 normalization
        if self.normalize_output:
            output = F.normalize(output, p=2, dim=1)

        return output


class SimpleEncoder(nn.Module):
    """Simple encoder for baseline comparison."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256],
        output_dim: int = 128,
        dropout_rate: float = 0.2,
        normalize_output: bool = True
    ):
        super().__init__()
        self.normalize_output = normalize_output

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Final projection layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with improved strategy."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for ReLU networks
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            features: Input features (batch_size, input_dim)

        Returns:
            Encoded embeddings (batch_size, output_dim)
        """
        output = self.encoder(features)
        # Optionally apply L2 normalization
        if self.normalize_output:
            output = F.normalize(output, p=2, dim=1)
        return output


class AttentionEncoder(nn.Module):
    """Encoder with attention mechanism for feature weighting."""

    def __init__(
        self,
        semantic_dim: int = 384,
        style_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout_rate: float = 0.2,
        normalize_output: bool = True
    ):
        super().__init__()

        self.semantic_dim = semantic_dim
        self.style_dim = style_dim
        self.normalize_output = normalize_output

        # Feature projections
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim // 2)
        self.style_proj = nn.Linear(style_dim, hidden_dim // 2)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 2),  # 2 attention weights for semantic and style
            nn.Softmax(dim=1)
        )

        # Final encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with improved strategy."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for ReLU networks
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, semantic_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention-weighted features.

        Args:
            semantic_features: Semantic embeddings (batch_size, semantic_dim)
            style_features: Stylometric features (batch_size, style_dim)

        Returns:
            Attended embeddings (batch_size, output_dim)
        """
        # Project features
        semantic_proj = self.semantic_proj(semantic_features)
        style_proj = self.style_proj(style_features)

        # Concatenate for attention computation
        combined = torch.cat([semantic_proj, style_proj], dim=1)

        # Compute attention weights
        attention_weights = self.attention(combined)  # (batch_size, 2)

        # Apply attention weights
        weighted_semantic = semantic_proj * attention_weights[:, 0:1]
        weighted_style = style_proj * attention_weights[:, 1:2]

        # Combine and encode
        attended_features = torch.cat([weighted_semantic, weighted_style], dim=1)
        output = self.encoder(attended_features)

        # Optionally apply L2 normalization
        if self.normalize_output:
            output = F.normalize(output, p=2, dim=1)

        return output
