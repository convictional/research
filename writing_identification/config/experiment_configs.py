"""Experimental configurations for Reddit-based authorship verification training.

All configurations now use Reddit data exclusively for consistent two-stage training.
Swan07 dependencies have been removed in favor of unified Reddit pipeline.
"""

from typing import Any

# Configuration 1: Balanced Features with Improved Training
BALANCED_FEATURES_CONFIG = {
    "model": {
        "style_feature_dim": 1169,  # Actual extracted dimensions
        "hidden_dim": 768,  # Increased from 512
        "dropout_rate": 0.3,  # Increased for regularization
        "final_embedding_dim": 512,  # Increased capacity
    },
    "training": {
        "learning_rate": 5e-4,  # Increased from 1e-4
        "margin": 0.5,  # Already updated
        "batch_size": 32,  # Already optimized
        "num_epochs": 60,  # More epochs for convergence
        "early_stopping_patience": 15,  # More patience
    },
    "data": {
        "max_features": 1000,  # Already reduced
        "min_text_length": 200,  # Increased minimum length
    },
}

# Configuration 2: Attention Architecture
ATTENTION_CONFIG = {
    "model": {
        "encoder_type": "attention",  # Switch to attention encoder
        "hidden_dim": 1024,  # Increased capacity
        "final_embedding_dim": 512,  # Larger embedding space
        "dropout_rate": 0.2,  # Lower dropout for deeper model
    },
    "training": {
        "learning_rate": 1e-4,  # Conservative for attention
        "margin": 0.7,  # Larger margin
        "batch_size": 24,  # Smaller batch for attention
        "num_epochs": 80,  # More epochs for complex model
    },
}

# Configuration 3: Triplet Loss Approach
TRIPLET_CONFIG = {
    "loss_type": "triplet",  # Switch to triplet loss
    "margin": 1.0,  # Larger margin for triplet
    "hard_negative_top_k": 10,  # More hard negatives
    "training": {
        "learning_rate": 3e-4,  # Medium learning rate
        "batch_size": 16,  # Smaller for triplet complexity
        "num_epochs": 100,  # More epochs needed
    },
    "model": {
        "hidden_dim": 1024,  # Increased capacity
        "final_embedding_dim": 256,  # Standard embedding size
    },
}

# Configuration 4: Deep Network with Residual Connections
DEEP_NETWORK_CONFIG = {
    "model": {
        "encoder_type": "deep_fusion",  # New encoder type (to be implemented)
        "hidden_dim": 512,
        "num_layers": 6,  # Deeper network
        "use_residual": True,  # Residual connections
        "final_embedding_dim": 384,
    },
    "training": {
        "learning_rate": 1e-4,
        "margin": 0.6,
        "batch_size": 20,  # Smaller for memory
        "warmup_epochs": 5,  # Learning rate warmup
    },
}

# Configuration 5: High Learning Rate with Aggressive Regularization
AGGRESSIVE_CONFIG = {
    "training": {
        "learning_rate": 1e-3,  # Very high learning rate
        "margin": 0.8,  # Large margin
        "batch_size": 16,  # Small batch for stability
        "weight_decay": 1e-4,  # Higher weight decay
        "gradient_clip_norm": 0.5,  # Tighter gradient clipping
    },
    "model": {
        "dropout_rate": 0.4,  # High dropout
        "hidden_dim": 256,  # Smaller network
        "final_embedding_dim": 128,  # Compact embeddings
    },
}

# Configuration 6: Semantic-Focused Architecture
SEMANTIC_FOCUSED_CONFIG = {
    "model": {
        "style_feature_dim": 1169,  # Use actual dimensions (other configs can override max_features)
        "embedding_dim": 768,  # Keep semantic features full
        "hidden_dim": 1024,  # Large semantic processing
        "semantic_weight": 2.0,  # Weight semantic features more
    },
    "training": {
        "learning_rate": 2e-4,
        "margin": 0.4,  # Smaller margin
        "batch_size": 48,  # Larger batch for stability
    },
}

# Configuration 7: Anti-Overfitting (Regularized)
REGULARIZED_CONFIG = {
    "model": {
        "style_feature_dim": 1169,  # Actual extracted dimensions
        "hidden_dim": 256,  # Reduced capacity (vs 768 in balanced)
        "dropout_rate": 0.5,  # Increased dropout (vs 0.3)
        "final_embedding_dim": 128,  # Smaller embeddings (vs 512)
    },
    "training": {
        "learning_rate": 1e-4,  # Reduced learning rate (vs 5e-4)
        "margin": 0.3,  # Smaller margin for easier learning
        "batch_size": 16,  # Smaller batch for more updates
        "num_epochs": 100,  # More epochs with slower learning
        "early_stopping_patience": 20,  # More patience for gradual improvement
        "weight_decay": 1e-4,  # Higher weight decay (vs 1e-5)
        "validation_split": 0.3,  # Larger validation set (vs 0.2)
    },
    "data": {
        "max_features": 500,  # Further reduce style features
        "min_text_length": 300,  # Longer text samples for better quality
    },
}

# Configuration 8: Reddit Data Only with Contrastive Loss (Current)
EXTERNAL_CONTRASTIVE_CONFIG = {
    "external_data": {
        "use_external_data": True,
        "dataset_name": "reddit",
        "training_strategy": "external_only",  # Train on external, validate on internal
        "max_external_samples": None,  # Use all 315k samples
        "external_train_split": "train",
        "external_val_split": "validation",
        "cache_external_features": True,
        "use_gcs_cache": True,  # Use cached data from GCS
        # Reddit data loaded dynamically - no GCS metadata needed
        "gcs_train_metadata": None,
        "gcs_val_metadata": None,
    },
    "precomputed_features": {
        "use_precomputed_features": True,  # Enable pre-computed features when available
        "feature_version": "1.0",
        "gcs_train_features": "gs://${GCS_BUCKET}/precomputed-features/v20250923/features_reddit_train.h5",
        "gcs_val_features": "gs://${GCS_BUCKET}/precomputed-features/v20250924/features_reddit_validation.h5",
        "auto_download_from_gcs": True,
        "validate_features_on_load": True,
    },
    "model": {
        "style_feature_dim": 1169,
        "hidden_dim": 1536,  # Larger capacity for diverse external data
        "dropout_rate": 0.3,  # Lower dropout for large dataset
        "final_embedding_dim": 512,
    },
    "training": {
        "learning_rate": 1e-5,  # Slightly higher for large batches
        "margin": 0.5,  # Contrastive margin
        "batch_size": 1024,  # 16x larger batch size
        "num_epochs": 15,  # Fewer epochs with larger batches
        "early_stopping_patience": 5,
        "weight_decay": 1e-5,
    },
    "data": {"max_features": 1000, "min_text_length": 100},
}

# Configuration 9: Reddit Data Only with Triplet Loss
EXTERNAL_TRIPLET_CONFIG = {
    "loss_type": "triplet",  # Switch to triplet loss
    "margin": 1.0,  # Larger margin for triplet
    "hard_negative_top_k": 10,  # More hard negatives for external diversity
    "external_data": {
        "use_external_data": True,
        "dataset_name": "reddit",
        "training_strategy": "external_only",
        "max_external_samples": None,
        "external_train_split": "train",
        "external_val_split": "validation",
        "cache_external_features": True,
        "use_gcs_cache": True,  # Use cached data from GCS
        # Reddit data loaded dynamically - no GCS metadata needed
        "gcs_train_metadata": None,
        "gcs_val_metadata": None,
    },
    "precomputed_features": {
        "use_precomputed_features": True,  # Enable pre-computed features when available
        "feature_version": "1.0",
        "gcs_train_features": "gs://${GCS_BUCKET}/precomputed-features/v20250923/features_reddit_train.h5",
        "gcs_val_features": "gs://${GCS_BUCKET}/precomputed-features/v20250924/features_reddit_validation.h5",
        "auto_download_from_gcs": True,
        "validate_features_on_load": True,
    },
    "model": {
        "style_feature_dim": 1169,
        "hidden_dim": 2048,  # Increased capacity for triplet complexity
        "dropout_rate": 0.3,  # Moderate dropout for triplet training
        "final_embedding_dim": 512,
    },
    "training": {
        "learning_rate": 1e-4,  # Higher LR for large batch triplet
        "batch_size": 1024,
        "num_epochs": 15,  # Fewer epochs with larger batches
        "early_stopping_patience": 6,
        "weight_decay": 1e-5,
    },
    "data": {"max_features": 1000, "min_text_length": 100},
}

# Configuration 10: Reddit Data Only with Attention Encoder
EXTERNAL_ATTENTION_CONFIG = {
    "external_data": {
        "use_external_data": True,
        "dataset_name": "reddit",
        "training_strategy": "external_only",
        "max_external_samples": None,
        "external_train_split": "train",
        "external_val_split": "validation",
        "cache_external_features": True,
        "use_gcs_cache": True,  # Use cached data from GCS
        # Reddit data loaded dynamically - no GCS metadata needed
        "gcs_train_metadata": None,
        "gcs_val_metadata": None,
    },
    "precomputed_features": {
        "use_precomputed_features": True,  # Enable pre-computed features when available
        "feature_version": "1.0",
        "gcs_train_features": "gs://${GCS_BUCKET}/precomputed-features/v20250923/features_reddit_train.h5",
        "gcs_val_features": "gs://${GCS_BUCKET}/precomputed-features/v20250924/features_reddit_validation.h5",
        "auto_download_from_gcs": True,
        "validate_features_on_load": True,
    },
    "model": {
        "encoder_type": "attention",  # Switch to attention encoder
        "style_feature_dim": 1169,
        "hidden_dim": 2048,  # Large capacity for attention
        "dropout_rate": 0.3,  # Lower dropout for deeper attention model
        "final_embedding_dim": 1024,  # Larger embedding space for attention
    },
    "training": {
        "learning_rate": 1e-4,  # Slightly higher for larger batches
        "margin": 0.7,  # Larger margin for attention embeddings
        "batch_size": 1024,  # 16x larger (attention uses more memory)
        "num_epochs": 15,  # Fewer epochs with larger batches
        "early_stopping_patience": 8,
        "weight_decay": 1e-5,
    },
    "data": {"max_features": 1000, "min_text_length": 100},
}

# Configuration 9: External Pre-training + Internal Fine-tuning
EXTERNAL_PRETRAIN_CONFIG = {
    "external_data": {
        "use_external_data": True,
        "dataset_name": "reddit",
        "training_strategy": "external_pretrain",
        "max_external_samples": None,  # Use all external data
        "pretrain_epochs": 15,  # Pre-train on external data
        "finetune_epochs": 10,  # Fine-tune on internal data
        "finetune_learning_rate": 5e-5,  # Lower LR for fine-tuning
    },
    "model": {"style_feature_dim": 1169, "hidden_dim": 512, "dropout_rate": 0.25, "final_embedding_dim": 256},
    "training": {
        "learning_rate": 1e-4,  # Higher LR for pre-training
        "margin": 0.5,
        "batch_size": 64,
        "num_epochs": 15,  # This will be pre-training epochs
        "early_stopping_patience": 5,
        "weight_decay": 1e-5,
    },
    "data": {"max_features": 1000, "min_text_length": 100},
}

# Configuration 10: Mixed Training (External + Internal)
MIXED_TRAINING_CONFIG = {
    "external_data": {
        "use_external_data": True,
        "dataset_name": "reddit",
        "training_strategy": "mixed",  # Combine both datasets
        "max_external_samples": None,  # Use all external data
    },
    "model": {
        "style_feature_dim": 1169,
        "hidden_dim": 512,
        "dropout_rate": 0.3,  # Higher dropout for mixed training
        "final_embedding_dim": 256,
    },
    "training": {
        "learning_rate": 2e-4,  # Slightly higher for diverse mixed data
        "margin": 0.4,
        "batch_size": 48,  # Moderate batch size
        "num_epochs": 25,
        "early_stopping_patience": 8,
        "weight_decay": 1e-5,
    },
    "data": {"max_features": 1000, "min_text_length": 100},
}

# Configuration 11: Reddit Classification Training
REDDIT_CLASSIFICATION_CONFIG = {
    "external_data": {
        "use_external_data": True,
        "dataset_name": "reddit",
        "training_strategy": "classification",
        "reddit_config": {
            "dataset_name": "subreddit-Cornell",
            "subreddits": [
                # Business & Entrepreneurship
                "business",
                "Entrepreneur",
                "startups",
                "smallbusiness",
                "consulting",
                # Tech & Work
                "sysadmin",
                "ITCareerQuestions",
                "datascience",
                "MachineLearning",
                "programming",
                # Finance & Operations
                "finance",
                "FinancialCareers",
                "accounting",
                "AskHR",
                # Professional Communication
                "careerguidance",
            ],
            "min_comments_per_author": 15,  # Higher threshold for better quality
            "max_authors": 2000,  # Start with manageable number
            "min_comment_length": 150,  # Longer comments for better features
            "max_comment_length": 3000,  # Reasonable upper bound
            "exclude_bots": True,
            "exclude_deleted": True,
        },
    },
    "training": {
        "training_mode": "classification",
        "classification_epochs": 25,
        "learning_rate": 1e-4,
        "batch_size": 128,  # Larger batches for classification
        "early_stopping_patience": 8,
        "use_gpu_cache": True,  # Cache features in GPU memory
    },
    "model": {
        "head_type": "arcface",
        "margin_s": 30.0,
        "margin_m": 0.30,
        "hidden_dim": 1024,  # Larger capacity for many authors
        "final_embedding_dim": 512,  # Good embedding size for ArcFace
        "dropout_rate": 0.3,  # Higher dropout for regularization
    },
    "data": {"max_features": 1000, "min_text_length": 150},
}

# Configuration 12: Reddit Two-Stage Training
REDDIT_TWO_STAGE_CONFIG = {
    "external_data": {
        "use_external_data": True,
        "dataset_name": "reddit",
        "training_strategy": "two_stage",
        "reddit_config": {
            "dataset_name": "subreddit",  # ConvoKit subreddit corpus
            "subreddits": [
                # Start with just business subreddit for testing (already downloaded)
                "business"
            ],
            "min_comments_per_author": 5,  # Lower threshold for more authors in test
            "max_authors": 1000,  # Increase 100x for real training
            "min_comment_length": 50,  # Lower minimum for testing
            "max_comment_length": 3000,
            "exclude_bots": True,
            "exclude_deleted": True,
        },
    },
    "training": {
        "training_mode": "two_stage",
        "classification_epochs": 10,  # Stage 1 - increase 3x for real training
        "verification_epochs": 15,  # Stage 2 - increase 3x for real training
        "learning_rate": 5e-5,  # Cut learning rate in half given additional epochs
        "batch_size": 128,  # Increase batch size given more samples
        "freeze_encoder_during_finetune": False,  # Allow end-to-end fine-tuning
        "use_gpu_cache": True,
        "early_stopping_patience": 5,
    },
    "model": {
        "encoder_type": "fusion",  # Explicitly set
        "head_type": "arcface",
        "margin_s": 25.0,  # Slightly lower for stability
        "margin_m": 0.3,  # Standard ArcFace margin
        "hidden_dim": 768,  # Moderate increase
        "final_embedding_dim": 384,  # Moderate increase
        "dropout_rate": 0.3,  # Slightly higher for larger model
    },
    "data": {"max_features": 1000, "min_text_length": 120},
    "precomputed_features": {
        "use_precomputed_features": False,  # Enable pre-computed features for Stage 2
        # "feature_version": "1.0",
        # Using one of the available pre-computed feature sets
        # Local paths will be auto-generated for Reddit data when needed
    },
}

# Configuration 13: Reddit Two-Stage Training with Attention Encoder for Verification
REDDIT_TWO_STAGE_ATTENTION_CONFIG = {
    "external_data": {
        "use_external_data": True,
        "dataset_name": "reddit",
        "training_strategy": "two_stage",
        "reddit_config": {
            "dataset_name": "subreddit-Cornell",
            "subreddits": [
                # Business & Entrepreneurship
                "business",
                "Entrepreneur",
                "startups",
                # "smallbusiness",
                # "consulting",
                # # Tech & Work
                # "sysadmin",
                # "ITCareerQuestions",
                # "datascience",
                # "MachineLearning",
                # "programming",
                # # Finance & Operations
                # "finance",
                # "FinancialCareers",
                # "accounting",
                "AskHR",
                # # Professional Communication
                # "careerguidance",
            ],
            "min_comments_per_author": 10,  # Good balance between quality and quantity
            "max_authors": 5000,  # Large dataset for better generalization
            "min_comment_length": 100,  # Meaningful content
            "max_comment_length": 3000,
            "exclude_bots": True,
            "exclude_deleted": True,
            "force_regenerate": False,  # Use cache when available
        },
        "reddit_pairs_config": {
            "positive_ratio": 0.5,
            "max_pairs_per_author": 20,  # More pairs per author
            "min_samples_per_author": 5,
            "seed": 42,
            "force_regenerate": False,  # Use cache when available
            "cache_dir": "cache/reddit_pairs",
        },
    },
    "training": {
        "training_mode": "two_stage",
        # Stage 1: Classification with fusion encoder
        "classification_epochs": 2,
        "classification_scheduler": "cosine",  # CosineAnnealingLR
        "classification_max_lr": 5e-4,  # For OneCycleLR if used
        # Stage 2: Verification with attention encoder
        "verification_epochs": 2,
        "verification_base_lr": 5e-6,  # Very low for fine-tuning with attention
        "verification_scheduler": "cosine",
        # General training settings
        "learning_rate": 1e-4,  # Fallback/initial LR
        "batch_size": 96,  # Moderate batch size for attention memory requirements
        "early_stopping_patience": 8,
        "use_scheduler": True,
        "use_gpu_cache": True,
        "save_checkpoint_every": 5,
    },
    "model": {
        # Stage 1 uses fusion encoder (default)
        # Stage 2 will switch to attention encoder
        "encoder_type": "attention",  # This will be used for verification stage
        "head_type": "arcface",  # For classification stage
        "margin_s": 30.0,  # ArcFace scale
        "margin_m": 0.3,  # ArcFace margin
        "hidden_dim": 1024,  # Larger hidden dim for attention
        "final_embedding_dim": 512,  # Larger embeddings for attention
        "dropout_rate": 0.25,  # Moderate dropout for attention
        "embedding_dim": 768,  # Semantic embedding dim
        "style_feature_dim": 1169,  # Style features dim
    },
    "data": {
        "max_features": 1000,
        "min_text_length": 100,
    },
    "precomputed_features": {
        "use_precomputed_features": True,  # Try to use pre-computed features for Stage 2
        "feature_version": "1.0",
        "auto_download_from_gcs": True,
        "validate_features_on_load": True,
        "cache_dir": "cache/precomputed_features",
        # GCS paths will be auto-generated for Reddit data
        "gcs_train_features": "gs://${GCS_BUCKET}/precomputed-features/v20250929/features_reddit_train.h5",
        "gcs_val_features": "gs://${GCS_BUCKET}/precomputed-features/v20250929/features_reddit_validation.h5",
        "gcs_val_extractors": "gs://${GCS_BUCKET}/precomputed-features/v20250929/features_reddit_validation.extractors.pkl",
    },
}

# Configuration for quick local testing with Reddit data
REDDIT_SMALL_TEST_CONFIG = {
    "external_data": {
        "use_external_data": True,
        "dataset_name": "reddit",
        "training_strategy": "classification",
        "reddit_config": {
            "dataset_name": "subreddit",  # ConvoKit subreddit corpus
            "subreddits": [
                # Start with just business subreddit for testing (already downloaded)
                "business"
            ],
            "min_comments_per_author": 5,  # Lower threshold for more authors in test
            "max_authors": 100,  # Small number for quick testing
            "min_comment_length": 50,  # Lower minimum for testing
            "max_comment_length": 3000,
            "exclude_bots": True,
            "exclude_deleted": True,
        },
    },
    "training": {
        "training_mode": "classification",
        "classification_epochs": 5,  # Just a few epochs to test
        "learning_rate": 1e-4,
        "batch_size": 32,  # Smaller batch for MPS
        "early_stopping_patience": 3,
        "use_gpu_cache": False,  # Disable for testing
    },
    "model": {
        "head_type": "arcface",
        "margin_s": 30.0,
        "margin_m": 0.30,
        "hidden_dim": 512,  # Smaller model for testing
        "final_embedding_dim": 256,
        "dropout_rate": 0.2,
    },
}


def get_experiment_config(config_name: str) -> dict[str, Any]:
    """Get configuration by name."""
    configs = {
        "balanced": BALANCED_FEATURES_CONFIG,
        "attention": ATTENTION_CONFIG,
        "triplet": TRIPLET_CONFIG,
        "deep": DEEP_NETWORK_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
        "semantic": SEMANTIC_FOCUSED_CONFIG,
        "regularized": REGULARIZED_CONFIG,
        "external_contrastive": EXTERNAL_CONTRASTIVE_CONFIG,
        "external_triplet": EXTERNAL_TRIPLET_CONFIG,
        "external_attention": EXTERNAL_ATTENTION_CONFIG,
        "external_pretrain": EXTERNAL_PRETRAIN_CONFIG,
        "mixed": MIXED_TRAINING_CONFIG,
        "reddit_classification": REDDIT_CLASSIFICATION_CONFIG,
        "reddit_two_stage": REDDIT_TWO_STAGE_CONFIG,
        "reddit_two_stage_attention": REDDIT_TWO_STAGE_ATTENTION_CONFIG,
        "reddit_test": REDDIT_SMALL_TEST_CONFIG,
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    return configs[config_name]


def list_experiment_configs() -> dict[str, str]:
    """List all available experiment configurations with descriptions."""
    return {
        "balanced": "Balanced features with improved training hyperparameters",
        "attention": "Attention-based encoder with larger capacity",
        "triplet": "Triplet loss training with hard negative mining",
        "deep": "Deep network with residual connections",
        "aggressive": "High learning rate with aggressive regularization",
        "semantic": "Focus on semantic features over stylometric",
        "regularized": "Anti-overfitting config with heavy regularization",
        "external_contrastive": "Train on 315k external samples with contrastive loss",
        "external_triplet": "Train on 315k external samples with triplet loss",
        "external_attention": "Train on 315k external samples with attention encoder",
        "external_pretrain": "Pre-train on external data, then fine-tune on internal",
        "mixed": "Combined training with external + internal data",
        "reddit_classification": "Classification training on Reddit dataset with ArcFace head",
        "reddit_two_stage": "Two-stage: Reddit classification pretraining + verification fine-tuning",
        "reddit_test": "Small Reddit test config for quick local testing (100 authors, 5 epochs)",
    }
