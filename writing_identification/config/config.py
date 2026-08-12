"""Configuration for writing identification experiment."""

from typing import Optional
from pydantic import BaseModel
import torch


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    # Upgraded semantic backbone to a stronger MPNet variant for richer embeddings
    sentence_bert_model: str = "all-mpnet-base-v2"

    # `all-mpnet-base-v2` outputs 768-dimensional sentence embeddings
    embedding_dim: int = 768

    # Model architecture
    encoder_type: str = "fusion"  # Options: "fusion", "simple", "attention"
    hidden_dim: int = 512
    style_feature_dim: int = 1169  # Actual extracted features: 3*333 vectorizer + 130 fixed + 38 email
    final_embedding_dim: int = 256
    dropout_rate: float = 0.2

    # Classification head parameters
    num_authors: int | None = None  # Number of authors (set automatically from data)
    head_type: str = "arcface"  # Options: "arcface", "cosface", "linear"
    margin_s: float = 30.0  # Scale parameter for margin heads
    margin_m: float = 0.30  # Margin parameter for angular margin


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = 32  # Decreased for more gradient updates
    learning_rate: float = 1e-4  # Increased to escape collapsed state
    num_epochs: int = 50
    margin: float = 0.5  # Increased margin for better separation
    validation_split: float = 0.2
    early_stopping_patience: int = 10  # More patience for convergence
    save_checkpoint_every: int = 5
    use_gpu_cache: bool = True  # Cache entire dataset in VRAM when possible

    # Two-stage training parameters
    training_mode: str = "classification"  # Options: "classification", "verification", "two_stage"
    classification_epochs: int = 20  # Number of classification pretraining epochs
    verification_epochs: int = 10  # Number of verification fine-tuning epochs
    freeze_encoder_during_finetune: bool = False  # Whether to freeze encoder during verification

    # Learning rate scheduler configuration
    use_scheduler: bool = True
    classification_scheduler: str = "cosine"  # Options: "cosine", "onecycle", "none"
    verification_scheduler: str = "cosine"  # Options: "cosine", "warmup_cosine", "none"
    classification_max_lr: float = 5e-4  # Maximum learning rate for OneCycleLR
    verification_base_lr: float = 1e-6  # Base learning rate for verification stage
    scheduler_warmup_epochs: int = 5  # Warmup epochs for cosine with warmup
    scheduler_t_max: int = 20  # T_max for CosineAnnealingLR (classification epochs)


class DataConfig(BaseModel):
    """Data processing configuration."""

    min_text_length: int = 100
    max_text_length: int = 5000
    char_ngram_range: tuple[int, int] = (2, 4)
    max_features: int = 1000
    pos_tag_features: bool = True
    function_words_features: bool = True


class PrecomputedFeaturesConfig(BaseModel):
    """Pre-computed features configuration."""

    use_precomputed_features: bool = False
    feature_version: str = "1.0"
    # Local paths
    train_features_path: str | None = None  # "cache/precomputed/features_train_v1.0.h5"
    val_features_path: str | None = None  # "cache/precomputed/features_val_v1.0.h5"
    # GCS paths (for Vertex AI)
    gcs_train_features: str | None = None  # "gs://bucket/precomputed-features/v1.0/train_features.h5"
    gcs_val_features: str | None = None  # "gs://bucket/precomputed-features/v1.0/val_features.h5"
    gcs_val_extractors: str | None = None  # "gs://bucket/precomputed-features/v1.0/val_extractors.pkl"
    # Auto-download from GCS if local files don't exist
    auto_download_from_gcs: bool = True
    # Validation settings
    validate_features_on_load: bool = True
    cache_dir: str = "cache/precomputed"


class RedditDataConfig(BaseModel):
    """Reddit dataset configuration."""
    dataset_name: str = "subreddit-Cornell"  # ConvoKit dataset name
    subreddits: list[str] | None = [
        # Business & Entrepreneurship
        "business", "Entrepreneur", "startups", "smallbusiness", "consulting",
        # Tech & Work
        "sysadmin", "ITCareerQuestions", "datascience", "MachineLearning", "programming",
        # Finance & Operations
        "finance", "FinancialCareers", "accounting", "AskHR",
        # Professional Communication
        "careerguidance"
    ]  # Target subreddits (None = all)
    min_comment_length: int = 100  # Minimum characters per comment
    max_comment_length: int = 5000  # Maximum characters per comment
    min_comments_per_author: int = 10  # Minimum comments per author
    max_authors: int | None = 5000  # Maximum authors to include
    exclude_deleted: bool = True  # Exclude [deleted] and [removed] content
    exclude_bots: bool = True  # Exclude known bot accounts
    cache_dir: str = "cache/reddit"
    force_regenerate: bool = False  # Force regeneration even if cache exists


class ExternalDataConfig(BaseModel):
    """External dataset configuration."""

    use_external_data: bool = False
    dataset_name: str = "reddit"  # Options: "reddit" (swan07 removed in favor of Reddit-only pipeline)
    training_strategy: str = "external_only"  # Options: external_only, external_pretrain, mixed
    max_external_samples: int | None = None  # None = use all available
    external_train_split: str = "train"
    external_val_split: str = "validation"
    cache_external_features: bool = True

    # Reddit dataset configuration
    reddit_config: RedditDataConfig = RedditDataConfig()

    # Reddit pair generation for verification
    reddit_pairs_config: Optional["RedditPairsConfig"] = None

    # GCS cache settings (faster than downloading from HuggingFace each time)
    use_gcs_cache: bool = False  # Enable to use pre-cached datasets from GCS
    gcs_train_metadata: str | None = None  # "gs://bucket/path/to/train_metadata.json"
    gcs_val_metadata: str | None = None  # "gs://bucket/path/to/val_metadata.json"
    gcs_test_metadata: str | None = None  # "gs://bucket/path/to/test_metadata.json"
    # Fine-tuning parameters for external_pretrain strategy
    pretrain_epochs: int = 30
    finetune_epochs: int = 10
    finetune_learning_rate: float = 1e-5  # Lower LR for fine-tuning


class RedditPairsConfig(BaseModel):
    """Configuration for Reddit pair generation in two-stage training."""

    cache_dir: str = "cache/reddit_pairs"
    positive_ratio: float = 0.5
    max_pairs_per_author: int = 100
    min_samples_per_author: int = 2
    seed: int = 42
    force_regenerate: bool = False


class ExperimentConfig(BaseModel):
    """Main experiment configuration."""

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    external_data: ExternalDataConfig = ExternalDataConfig()
    precomputed_features: PrecomputedFeaturesConfig = PrecomputedFeaturesConfig()

    # Device configuration
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    random_seed: int = 42

    # Paths
    model_save_path: str = "models/checkpoints"
    results_path: str = "results"
    data_path: str = "data"


# Global config instance
config = ExperimentConfig()

# Set default Reddit pairs config
config.external_data.reddit_pairs_config = RedditPairsConfig()
