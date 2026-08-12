"""Vertex AI training wrapper for authorship verification model."""

import sys
import logging
import asyncio
import pickle
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from google.cloud import logging as cloud_logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.vertex_config import get_merged_config, VertexConfig
from config.config import ExperimentConfig, RedditPairsConfig
from scripts.train import AuthorshipTrainer
from utils.gcs_storage import HybridStorage
from utils.metrics_logger import MetricsLogger
from data.extract_training_data import AuthorshipDataExtractor
from data.dataset import AuthorshipPairDataset
from data.classification_dataset import AuthorClassificationDataset
from data.precomputed_dataset import create_precomputed_data_loaders, load_extractors_from_h5
from data.reddit_dataset import load_reddit_dataset_for_training
from data.reddit_pairs import RedditPairGenerator
from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
from features.email_patterns import EmailPatternExtractor
from models.classification import AuthorClassifier
from models.siamese import SiameseNetwork, AuthorshipVerifier


# Setup logging
def setup_logging(vertex_config: VertexConfig):
    """Configure logging for Vertex AI and local development."""
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(__name__)

    # Add Google Cloud Logging if on Vertex AI
    if vertex_config.vertex_ai_enabled and vertex_config.gcp_project:
        try:
            client = cloud_logging.Client(project=vertex_config.gcp_project)
            client.setup_logging()
            logger.info("Google Cloud Logging initialized")
        except Exception as e:
            logger.warning(f"Failed to setup Cloud Logging: {e}")

    return logger


class VertexAuthTrainer(AuthorshipTrainer):
    """Extended trainer with Vertex AI and GCS support."""

    def __init__(self, experiment_config: ExperimentConfig, vertex_config: VertexConfig, storage: HybridStorage):
        """
        Initialize Vertex-aware trainer.

        Args:
            experiment_config: Experiment configuration
            vertex_config: Vertex AI configuration
            storage: Hybrid storage handler
        """
        # Initialize base trainer without config override
        super().__init__(config_override=None)

        # Use the properly merged Pydantic config from vertex_config.apply_overrides
        self.config = experiment_config
        self.vertex_config = vertex_config
        self.storage = storage
        self.run_id = vertex_config.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")

        # Determine optimal DataLoader settings based on environment
        self.use_cuda = torch.cuda.is_available()
        self.dataloader_num_workers = 4 if self.use_cuda else 0  # Use parallel loading on CUDA
        self.dataloader_pin_memory = self.use_cuda  # Pin memory for faster GPU transfer

        if self.use_cuda:
            logger.info(
                f"CUDA detected - using optimized DataLoader settings (workers={self.dataloader_num_workers}, pin_memory={self.dataloader_pin_memory})"
            )
        else:
            logger.info(
                f"No CUDA - using CPU DataLoader settings (workers={self.dataloader_num_workers}, pin_memory={self.dataloader_pin_memory})"
            )

        # Initialize unified metrics logger
        self.metrics_logger = MetricsLogger(
            experiment_name=vertex_config.experiment_name,
            run_id=self.run_id,
            config=experiment_config.model_dump(),
            enable_tensorboard=vertex_config.use_tensorboard,
            enable_vertex_ai=vertex_config.vertex_ai_enabled,
            enable_json=True,
            gcs_bucket=vertex_config.gcs_bucket,
            project_id=vertex_config.gcp_project,
        )

        # Keep legacy vertex_ai_initialized for backward compatibility
        self.vertex_ai_initialized = self.metrics_logger.vertex_ai_enabled

    def _init_vertex_ai(self):
        """Legacy method - now handled by MetricsLogger."""
        # This method is kept for compatibility but functionality moved to MetricsLogger
        pass

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = "_") -> dict:
        """Flatten nested dictionary for Vertex AI parameter logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            elif v is not None:
                items.append((new_key, v))
        return dict(items)

    def _sanitize_metrics(self, metrics: dict) -> dict:
        """Convert NumPy types to Python native types for Vertex AI logging."""
        sanitized = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):  # NumPy scalar
                sanitized[k] = v.item()
            elif isinstance(v, (int, float, str, bool)):
                sanitized[k] = v
            else:
                # Convert other types to string as fallback
                sanitized[k] = str(v)
        return sanitized

    async def prepare_data(
        self, texts_by_author: dict[str, list] = None, use_gcs: bool = False, gcs_bucket: str = None
    ):
        """Override to support external dataset integration."""

        # Handle Reddit dataset for classification
        if self.config.external_data.use_external_data and self.config.external_data.dataset_name == "reddit":
            self.prepare_reddit_data()
            return

        # Check if external data is enabled
        if self.config.external_data.use_external_data:
            strategy = self.config.external_data.training_strategy
            logger.info(f"Using external data training strategy: {strategy}")
            await self._prepare_external_primary_data(texts_by_author, use_gcs, gcs_bucket)
        else:
            logger.info("Using standard internal data only")
            await super().prepare_data(texts_by_author, use_gcs, gcs_bucket)

    async def _prepare_external_primary_data(
        self, texts_by_author: dict[str, list] = None, use_gcs: bool = False, gcs_bucket: str = None
    ):
        """Prepare training data with external dataset as primary source.

        This method now primarily handles pre-computed features.
        For Reddit-based external data, use prepare_reddit_data() instead.
        """

        strategy = self.config.external_data.training_strategy

        # Check for pre-computed features first
        if self.config.precomputed_features.use_precomputed_features:
            precomputed_paths = await self._prepare_precomputed_features()
            if precomputed_paths:
                # Load pre-computed features and create data loaders
                train_path, val_path = precomputed_paths
                self.train_loader, self.val_loader = await self._load_precomputed_data(train_path, val_path)

                # Load extractors from pre-computed features if available
                semantic_ext, style_ext, email_ext = load_extractors_from_h5(Path(train_path))
                if style_ext is not None:
                    self.semantic_extractor = semantic_ext
                    self.style_extractor = style_ext
                    self.email_extractor = email_ext
                    logger.info("✓ Loaded fitted extractors from pre-computed features")

                logger.info("Using pre-computed features for training")
                logger.info(f"Training samples: {len(self.train_loader.dataset):,}")
                logger.info(f"Validation samples: {len(self.val_loader.dataset):,}")
                logger.info(f"Training strategy: {strategy} (with pre-computed features)")
                return  # Skip the rest of data preparation

        # If no pre-computed features and not Reddit, fall back to internal data only
        logger.info("No pre-computed features available, using internal data only")
        await super().prepare_data(texts_by_author, use_gcs, gcs_bucket)

    async def _load_internal_data(
        self, texts_by_author: dict[str, list] = None, use_gcs: bool = False, gcs_bucket: str = None
    ) -> dict[str, list]:
        """Load internal data using existing logic."""
        if texts_by_author is None:
            if use_gcs and gcs_bucket:
                logger.info("Loading training data from GCS dataset...")
                from data.gcs_loader import GCSDataLoader

                gcs_loader = GCSDataLoader(gcs_bucket)
                texts_by_author = gcs_loader.load_training_data()
                logger.info(f"Loaded data for {len(texts_by_author)} authors from GCS")
            else:
                logger.info("Loading real training data from database...")
                extractor = AuthorshipDataExtractor()
                samples_by_author = await extractor.extract_training_samples()
                filtered_samples = extractor.filter_samples_for_training(samples_by_author)

                # Convert ContentSample objects to text strings
                texts_by_author = {}
                for author, samples in filtered_samples.items():
                    texts_by_author[author] = [sample.content for sample in samples]

                logger.info(f"Loaded data for {len(texts_by_author)} authors")

        # Filter authors with minimum samples
        min_samples = 5
        filtered_texts = {author: texts for author, texts in texts_by_author.items() if len(texts) >= min_samples}
        logger.info(f"Internal data: {len(filtered_texts)} authors with {min_samples}+ samples")

        return filtered_texts

    def prepare_reddit_data(self):
        """Prepare Reddit dataset for classification training."""
        logger.info("Loading Reddit dataset for classification training...")

        # Load Reddit data using configuration
        train_texts, val_texts, author_metadata = load_reddit_dataset_for_training(
            self.config.external_data.reddit_config
        )

        # Set number of authors for classification model
        self.config.model.num_authors = len(train_texts)
        logger.info(f"Reddit dataset: {len(train_texts)} authors for training")
        logger.info(f"Total training comments: {sum(len(texts) for texts in train_texts.values())}")
        logger.info(f"Total validation comments: {sum(len(texts) for texts in val_texts.values())}")

        # Create data loaders for classification
        if self.config.training.training_mode in ["classification", "two_stage"]:
            # For two-stage training, store texts for later verification stage
            if self.config.training.training_mode == "two_stage":
                self.train_texts_by_author = train_texts
                self.val_texts_by_author = val_texts
                self.author_metadata = author_metadata

            # Create classification data loaders for both modes
            self.train_loader, self.val_loader = self.create_classification_data_loaders(train_texts, val_texts)
        else:
            # For other modes, we might need different handling
            self.train_texts_by_author = train_texts
            self.val_texts_by_author = val_texts
            self.author_metadata = author_metadata

    def create_classification_data_loaders(self, train_texts: dict[str, list[str]], val_texts: dict[str, list[str]]):
        """Create data loaders for classification training."""

        # Download and load pre-fitted extractors instead of fitting in memory
        logger.info("Loading pre-fitted extractors from GCS...")
        extractor_cache_path = Path("cache/reddit_extractors.pkl")
        extractor_cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not extractor_cache_path.exists():
            # Get extractor path from config
            precomp_config = self.config.precomputed_features
            gcs_extractor_path = precomp_config.gcs_val_extractors

            if not gcs_extractor_path:
                raise RuntimeError("gcs_val_extractors not specified in config - cannot download pre-fitted extractors")

            logger.info(f"Downloading extractors from {gcs_extractor_path}")

            try:
                if self.storage.use_gcs:
                    self.storage.gcs.download_precomputed_features(gcs_extractor_path, str(extractor_cache_path))
                    logger.info("✓ Downloaded pre-fitted extractors")
                else:
                    raise RuntimeError("GCS storage not available but extractors not found locally")
            except Exception as e:
                logger.error(f"Failed to download extractors: {e}")
                raise RuntimeError(f"Cannot proceed without pre-fitted extractors: {e}")

        # Load the pre-fitted extractors
        logger.info("Loading extractors from cache...")
        with open(extractor_cache_path, "rb") as f:
            extractors = pickle.load(f)  # Load without encoding parameter

        # Initialize semantic and email extractors fresh (no fitting required)
        self.semantic_extractor = SemanticFeatureExtractor()
        self.email_extractor = EmailPatternExtractor()

        # Only load the fitted style extractor from pickle
        self.style_extractor = extractors.get("style_extractor")
        logger.info("✓ Initialized fresh semantic and email extractors, loaded fitted style extractor")

        if self.style_extractor is None:
            raise RuntimeError("Style extractor not found in downloaded file")

        logger.info("✓ Loaded pre-fitted extractors successfully")

        # Create classification datasets
        train_dataset = AuthorClassificationDataset(
            train_texts, self.semantic_extractor, self.style_extractor, self.email_extractor
        )
        val_dataset = AuthorClassificationDataset(
            val_texts, self.semantic_extractor, self.style_extractor, self.email_extractor
        )

        # Create data loaders with optimized settings for CUDA
        # Note: For classification, we need num_workers=0 because the semantic feature
        # extractor uses CUDA and multiprocessing with CUDA requires special handling
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,  # Must be 0 for CUDA feature extraction
            pin_memory=self.dataloader_pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,  # Must be 0 for CUDA feature extraction
            pin_memory=self.dataloader_pin_memory,
        )

        logger.info("Classification datasets created:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")

        return train_loader, val_loader

    async def _prepare_precomputed_features(self) -> tuple[str, str] | None:
        """Check for and download pre-computed features if configured."""
        precomp_config = self.config.precomputed_features

        if not precomp_config.use_precomputed_features:
            return None

        logger.info("Checking for pre-computed features...")

        # Determine local cache paths
        cache_dir = Path(precomp_config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        local_train_path = None
        local_val_path = None

        # Check if local files are specified and exist
        if precomp_config.train_features_path:
            train_path = Path(precomp_config.train_features_path)
            if train_path.exists():
                local_train_path = str(train_path)
                logger.info(f"Found local training features: {local_train_path}")

        if precomp_config.val_features_path:
            val_path = Path(precomp_config.val_features_path)
            if val_path.exists():
                local_val_path = str(val_path)
                logger.info(f"Found local validation features: {local_val_path}")

        # Download from GCS if files not found locally and auto-download enabled
        if precomp_config.auto_download_from_gcs and self.storage.use_gcs:
            if not local_train_path and precomp_config.gcs_train_features:
                local_train_path = str(cache_dir / "train_features.h5")
                logger.info(f"Downloading training features from {precomp_config.gcs_train_features}")
                try:
                    self.storage.gcs.download_precomputed_features(precomp_config.gcs_train_features, local_train_path)
                    logger.info(f"✓ Downloaded to {local_train_path}")
                except Exception as e:
                    logger.error(f"Failed to download training features: {e}")
                    local_train_path = None

            if not local_val_path and precomp_config.gcs_val_features:
                local_val_path = str(cache_dir / "val_features.h5")
                logger.info(f"Downloading validation features from {precomp_config.gcs_val_features}")
                try:
                    self.storage.gcs.download_precomputed_features(precomp_config.gcs_val_features, local_val_path)
                    logger.info(f"✓ Downloaded to {local_val_path}")
                except Exception as e:
                    logger.error(f"Failed to download validation features: {e}")
                    local_val_path = None

        # Check if we have both files
        if local_train_path and local_val_path:
            logger.info("✓ Pre-computed features available for training")
            return local_train_path, local_val_path
        elif precomp_config.use_precomputed_features:
            logger.warning(
                "Pre-computed features configured but not available - falling back to on-the-fly extraction"
            )
            return None
        else:
            return None

    async def _load_precomputed_data(self, train_path: str, val_path: str) -> tuple[DataLoader, DataLoader]:
        """Load pre-computed features and create data loaders."""
        logger.info("Loading data from pre-computed features...")

        # Check if GPU caching should be used
        use_gpu_cache = self.use_cuda and getattr(self.config.training, "use_gpu_cache", True)

        if use_gpu_cache:
            logger.info("GPU caching enabled - dataset will be loaded into VRAM")
            # Force optimal settings for GPU-cached data
            num_workers = 0  # No workers needed for GPU-cached data
            pin_memory = False  # No need to pin memory
        else:
            # Use configured dataloader settings
            num_workers = self.dataloader_num_workers
            pin_memory = self.dataloader_pin_memory

        device = next(iter(self.model.parameters())).device if hasattr(self, "model") else "cuda"

        train_loader, val_loader = create_precomputed_data_loaders(
            train_h5_path=train_path,
            val_h5_path=val_path,
            batch_size=self.config.training.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_gpu_cache=use_gpu_cache,
            device=device,
        )

        logger.info(
            f"✓ Loaded pre-computed features: {len(train_loader)} train batches, {len(val_loader)} val batches"
        )

        # Log performance optimization info
        if use_gpu_cache:
            logger.info("🚀 Using GPU-cached datasets for maximum training performance")
        else:
            logger.info("📁 Using CPU-based datasets with disk I/O")

        return train_loader, val_loader

    def save_checkpoint(self, epoch: int, is_best: bool = False, best_threshold: float = None):
        """Override to save checkpoints to GCS with extractors."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.model_dump(),
            "vertex_config": self.vertex_config.model_dump(),
            "training_history": self.training_history,
            "run_id": self.run_id,
            "best_threshold": best_threshold,  # Save optimal threshold from validation
            # Add extractors - critical for inference!
            "semantic_extractor": self.semantic_extractor
            if hasattr(self, "semantic_extractor") and self.semantic_extractor is not None
            else None,
            "style_extractor": self.style_extractor
            if hasattr(self, "style_extractor") and self.style_extractor is not None
            else None,
            "email_extractor": self.email_extractor
            if hasattr(self, "email_extractor") and self.email_extractor is not None
            else None,
        }

        # Save using hybrid storage (local + GCS)
        checkpoint_path = self.storage.save_checkpoint(checkpoint, epoch, is_best)

        if is_best:
            logger.info(f"Best model saved at epoch {epoch}: {checkpoint_path}")
        else:
            logger.info(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

        # Log checkpoint save to metrics
        self.metrics_logger.log_metrics(
            {"checkpoint_saved": 1, "checkpoint_epoch": epoch, "is_best_checkpoint": int(is_best)}, step=epoch
        )

    def save_training_history(self):
        """Override to save training history to GCS."""
        # Convert tensors to serializable format
        serializable_history = []
        for epoch_data in self.training_history:
            epoch_dict = {}
            for key, value in epoch_data.items():
                if hasattr(value, "item"):
                    epoch_dict[key] = value.item()
                else:
                    epoch_dict[key] = value
            serializable_history.append(epoch_dict)

        # Save using hybrid storage
        history_path = self.storage.save_training_history(serializable_history, self.run_id)
        logger.info(f"Training history saved: {history_path}")

        # Log final summary metrics
        if self.training_history:
            final_metrics = self.training_history[-1]
            final_metrics_clean = {
                f"final_{k}": v for k, v in final_metrics.items() if isinstance(v, (int, float)) or hasattr(v, "item")
            }
            self.metrics_logger.log_metrics(final_metrics_clean)

    async def train(self, num_epochs: int = None):
        """Override train method to add Vertex AI logging."""
        # Handle two-stage training separately
        if self.config.training.training_mode == "two_stage":
            await self.train_two_stage()
            return

        if num_epochs is None:
            # Use appropriate epochs based on training mode
            if self.config.training.training_mode == "classification":
                num_epochs = (
                    self.config.training.classification_epochs
                    if hasattr(self.config.training, "classification_epochs")
                    else self.config.training.num_epochs
                )
            else:
                num_epochs = self.config.training.num_epochs

        logger.info(f"Starting Vertex AI training for {num_epochs} epochs...")
        logger.info(f"Run ID: {self.run_id}")

        # Log initial training configuration
        training_config = {"run_id": self.run_id, "num_epochs": num_epochs, "start_time": datetime.now().isoformat()}
        self.metrics_logger.log_metrics(training_config)

        # Run base training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_threshold = 0.5  # Default threshold

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train and validate based on mode
            if self.config.training.training_mode == "classification":
                train_metrics = self.train_classification_epoch(epoch)
                val_metrics = self.validate_classification_epoch(epoch)
            else:
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate_epoch(epoch)

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.training_history.append(epoch_metrics)

            # Log metrics
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                f"Val AUC: {val_metrics['val_auc']:.4f}"
            )

            # Log metrics to all backends (TensorBoard, Vertex AI, JSON)
            self.metrics_logger.log_metrics(epoch_metrics, step=epoch)

                # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_loss"])
            elif type(self.scheduler).__name__ not in ["OneCycleLR"]:  # OneCycleLR steps per batch
                self.scheduler.step()

            # Check for model improvement
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                best_threshold = val_metrics.get("val_threshold", 0.5)
                self.save_checkpoint(epoch, is_best=True, best_threshold=best_threshold)
            else:
                patience_counter += 1

            # Save regular checkpoint
            if epoch % self.config.training.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.info("Training completed!")
        self.save_training_history()

        # Log training completion
        completion_metrics = {
            "training_completed": 1,
            "total_epochs": epoch + 1,
            "end_time": datetime.now().isoformat(),
            "best_val_loss": best_val_loss,
        }
        self.metrics_logger.log_metrics(completion_metrics)

        # Close metrics logger (handles TensorBoard, Vertex AI, JSON cleanup)
        self.metrics_logger.close()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Override to add enhanced logging."""
        # Get base metrics from parent
        metrics = super().train_epoch(epoch)

        # Add learning rate logging
        current_lr = self.optimizer.param_groups[0]["lr"]
        metrics["learning_rate"] = current_lr

        return metrics

    def validate_epoch(self, epoch: int) -> dict[str, float]:
        """Override to add enhanced logging."""
        # Get base validation metrics
        metrics = super().validate_epoch(epoch)

        # Log embedding diversity metrics to TensorBoard as histograms
        if hasattr(self, "last_val_embeddings") and self.metrics_logger.tensorboard_writer:
            try:
                # Log embedding norms
                embedding_norms = torch.norm(self.last_val_embeddings, dim=1)
                self.metrics_logger.log_histogram("embeddings/norms", embedding_norms, step=epoch)

                # Log embedding sample for visualization
                if epoch % 10 == 0:  # Every 10 epochs
                    sample_size = min(100, len(self.last_val_embeddings))
                    indices = torch.randperm(len(self.last_val_embeddings))[:sample_size]
                    sample_embeddings = self.last_val_embeddings[indices]
                    self.metrics_logger.log_embeddings(sample_embeddings, step=epoch)

            except Exception as e:
                logger.warning(f"Failed to log embedding metrics: {e}")

        return metrics

    def build_classification_model(self, semantic_dim: int, style_dim: int):
        """Build classification model with ArcFace head."""
        num_authors = getattr(self.config.model, "num_authors", None)
        if num_authors is None:
            raise ValueError("num_authors must be set for classification model")

        logger.info(f"Building classification model for {num_authors} authors")

        model = AuthorClassifier(
            encoder_type=self.config.model.encoder_type,
            semantic_dim=semantic_dim,
            style_dim=style_dim,
            hidden_dim=self.config.model.hidden_dim,
            embedding_dim=self.config.model.final_embedding_dim,
            num_authors=num_authors,
            dropout_rate=self.config.model.dropout_rate,
            head_type=self.config.model.head_type,
            margin_s=self.config.model.margin_s,
            margin_m=self.config.model.margin_m,
        )

        logger.info(
            f"Classification model: {self.config.model.head_type} head, "
            f"margin_s={self.config.model.margin_s}, margin_m={self.config.model.margin_m}"
        )

        # Store dimensions as model attributes for later use
        model.semantic_dim = semantic_dim
        model.style_dim = style_dim

        return model

    def build_verification_model(self, semantic_dim: int, style_dim: int):
        """Build verification model (existing Siamese network)."""
        siamese_net = SiameseNetwork(
            encoder_type=self.config.model.encoder_type,
            semantic_dim=semantic_dim,
            style_dim=style_dim,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=self.config.model.final_embedding_dim,
            dropout_rate=self.config.model.dropout_rate,
            normalize_embeddings=True,  # Critical for cosine similarity loss
        )

        # Build complete verification system
        model = AuthorshipVerifier(
            siamese_network=siamese_net,
            loss_type="cosine",  # Using cosine loss for normalized embeddings
            margin=self.config.training.margin,
        )

        logger.info("Verification model: Siamese network with cosine similarity loss")

        # Store dimensions as model attributes for later use
        model.semantic_dim = semantic_dim
        model.style_dim = style_dim

        return model

    def build_model(self):
        """Build and initialize the model - override parent implementation."""
        # Calculate feature dimensions with fallbacks
        semantic_dim = getattr(self.config.model, "embedding_dim", 768)
        style_dim = getattr(self.config.model, "style_feature_dim", 1169)

        # Validate dimensions
        logger.info(f"Model dimensions - Semantic: {semantic_dim}, Style: {style_dim}")
        if semantic_dim <= 0 or style_dim <= 0:
            raise ValueError(f"Invalid feature dimensions: semantic={semantic_dim}, style={style_dim}")

        # Build model based on training mode
        if self.config.training.training_mode == "classification":
            self.model = self.build_classification_model(semantic_dim, style_dim)
        elif self.config.training.training_mode == "two_stage":
            # Start with classification model for stage 1
            self.model = self.build_classification_model(semantic_dim, style_dim)
        else:
            # Default: verification model
            self.model = self.build_verification_model(semantic_dim, style_dim)

        # Move to device
        self.model = self.model.to(self.device)
        logger.info(f"Model built with {self.count_parameters()} parameters")

    def count_parameters(self) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def setup_optimizer(self, stage: str = "verification"):
        """Setup optimizer and scheduler with stage-specific configuration - override parent implementation."""
        # Set stage-specific learning rate
        if stage == "classification":
            lr = self.config.training.classification_learning_rate if hasattr(self.config.training, "classification_learning_rate") else 1e-4
            logger.info(f"Classification stage optimizer: lr={lr}")
        else:
            lr = self.config.training.learning_rate
            logger.info(f"Verification stage optimizer: lr={lr}")

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Setup scheduler based on stage and configuration
        if hasattr(self.config.training, "use_scheduler") and self.config.training.use_scheduler:
            if stage == "classification":
                scheduler_type = getattr(self.config.training, "classification_scheduler", "cosine")
                if scheduler_type == "onecycle":
                    # OneCycleLR for classification
                    max_lr = getattr(self.config.training, "classification_max_lr", 5e-4)
                    epochs = getattr(self.config.training, "classification_epochs", 10)
                    steps_per_epoch = len(self.train_loader)
                    self.scheduler = OneCycleLR(
                        self.optimizer,
                        max_lr=max_lr,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        pct_start=0.3,
                        anneal_strategy="cos",
                        final_div_factor=100,
                    )
                    logger.info(f"Using OneCycleLR scheduler: max_lr={max_lr}, epochs={epochs}")
                else:
                    # CosineAnnealingLR for classification
                    epochs = getattr(self.config.training, "classification_epochs", 10)
                    self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
                    logger.info(f"Using CosineAnnealingLR scheduler for {epochs} epochs")
            else:
                # Verification stage scheduler
                scheduler_type = getattr(self.config.training, "verification_scheduler", "cosine")
                if scheduler_type == "cosine":
                    epochs = getattr(self.config.training, "verification_epochs", 15)
                    self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-7)
                    logger.info(f"Using CosineAnnealingLR scheduler for verification: {epochs} epochs")
                else:
                    # Default to ReduceLROnPlateau
                    self.scheduler = ReduceLROnPlateau(
                        self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
                    )
                    logger.info("Using ReduceLROnPlateau scheduler for verification")
        else:
            # Default scheduler for backward compatibility
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
            )
            logger.info("Using default ReduceLROnPlateau scheduler")

    def train_classification_epoch(self, epoch: int) -> dict[str, float]:
        """Train classification model for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0

        # Classification uses CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

        for batch_idx, batch in enumerate(self.train_loader):
            # Classification batch format: dict with semantic_features, style_features, author_id
            semantic_features = batch["semantic_features"].to(self.device)
            style_features = batch["style_features"].to(self.device)
            author_ids = batch["author_id"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Get model outputs (includes logits for ArcFace/classification)
            outputs = self.model(semantic_features, style_features, author_ids)

            # Compute loss (model returns dict with 'logits' key)
            logits = outputs["logits"]
            loss = criterion(logits, author_ids)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Step OneCycleLR scheduler per batch if using it
            if self.scheduler is not None and type(self.scheduler).__name__ == "OneCycleLR":
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(author_ids.cpu().numpy())
            num_batches += 1

            if batch_idx % 50 == 0:
                logger.info(f"Classification Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(all_labels, all_predictions)

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss, "train_accuracy": accuracy, "train_loss_unscaled": avg_loss}

    def validate_classification_epoch(self, epoch: int) -> dict[str, float]:
        """Validate classification model for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_logits = []
        num_batches = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.val_loader:
                semantic_features = batch["semantic_features"].to(self.device)
                style_features = batch["style_features"].to(self.device)
                author_ids = batch["author_id"].to(self.device)

                outputs = self.model(semantic_features, style_features, author_ids)

                # Compute loss (model returns dict with 'logits' key)
                logits = outputs["logits"]
                loss = criterion(logits, author_ids)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(author_ids.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                num_batches += 1

        # Calculate metrics
        from sklearn.metrics import accuracy_score, top_k_accuracy_score
        import numpy as np

        accuracy = accuracy_score(all_labels, all_predictions)
        avg_loss = total_loss / num_batches

        # Calculate top-5 accuracy
        all_logits_np = np.array(all_logits)
        all_labels_np = np.array(all_labels)

        top5_accuracy = top_k_accuracy_score(all_labels_np, all_logits_np, k=5)

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_top5_accuracy": top5_accuracy,
            "val_loss_unscaled": avg_loss,
            "val_threshold": 0.5,  # Not used in classification, but kept for consistency
        }

    async def train_two_stage(self):
        """Two-stage training: classification then verification."""
        logger.info("Starting two-stage training in Vertex AI...")

        # Stage 1: Classification training
        logger.info("=" * 60)
        logger.info("STAGE 1: Classification Training")
        logger.info("=" * 60)

        # Setup optimizer for classification stage
        self.setup_optimizer(stage="classification")

        classification_epochs = getattr(self.config.training, "classification_epochs", 10)
        logger.info(f"Training classification model for {classification_epochs} epochs...")

        best_val_acc = 0.0
        best_classification_epoch = -1
        best_classification_threshold = 0.5  # Default threshold

        for epoch in range(classification_epochs):
            logger.info(f"Classification Epoch {epoch + 1}/{classification_epochs}")

            train_metrics = self.train_classification_epoch(epoch)
            val_metrics = self.validate_classification_epoch(epoch)

            # Log metrics
            logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f} (unscaled: {train_metrics['train_loss_unscaled']:.4f}), "
                f"Train Acc: {train_metrics['train_accuracy']:.4f}"
            )
            logger.info(
                f"Val Loss: {val_metrics['val_loss']:.4f} (unscaled: {val_metrics['val_loss_unscaled']:.4f}), "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )

            if "val_top5_accuracy" in val_metrics:
                logger.info(f"Val Top-5 Acc: {val_metrics['val_top5_accuracy']:.4f}")

            # Save metrics
            epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "stage": "classification"}
            self.training_history.append(epoch_metrics)

            # Log to metrics logger
            self.metrics_logger.log_metrics(epoch_metrics, step=epoch)

            # Track best model
            if val_metrics["val_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["val_accuracy"]
                best_classification_epoch = epoch + 1  # Store 1-based for logging
                best_classification_threshold = val_metrics.get("val_threshold", 0.5)
                self.save_checkpoint(epoch, is_best=True, best_threshold=best_classification_threshold)
                logger.info(f"New best classification accuracy: {best_val_acc:.4f} at epoch {best_classification_epoch}")

        # Stage 2: Convert to verification model and fine-tune
        logger.info("=" * 60)
        logger.info("STAGE 2: Verification Fine-tuning")
        logger.info("=" * 60)

        # Load the best classification checkpoint before conversion
        best_checkpoint_path = Path(self.config.model_save_path) / "best_model.pt"
        if best_checkpoint_path.exists():
            logger.info(f"Loading best classification model from epoch {best_classification_epoch}")
            # Use weights_only=False to support loading custom classes (PyTorch 2.6+ compatibility)
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning("No best checkpoint found, using last epoch weights")

        # Extract encoder from classification model
        logger.info("Converting classification model to verification model...")

        # Get the encoder from the classification model
        if hasattr(self.model, "encoder"):
            pretrained_encoder = self.model.encoder
        else:
            logger.error("Classification model doesn't have an encoder attribute")
            return

        # Get dimensions from config with fallbacks (AuthorClassifier doesn't store these as attributes)
        semantic_dim = getattr(self.config.model, "embedding_dim", 768)
        style_dim = getattr(self.config.model, "style_feature_dim", 1169)

        logger.info(f"Using dimensions - Semantic: {semantic_dim}, Style: {style_dim}")

        # Build verification model with pretrained encoder
        verification_model = self.build_verification_model(semantic_dim, style_dim)

        # Transfer encoder weights with proper attribute checking
        if (hasattr(verification_model, "siamese_network") and
            hasattr(verification_model.siamese_network, "encoder")):
            verification_model.siamese_network.encoder.load_state_dict(pretrained_encoder.state_dict())
            logger.info("Transferred encoder weights from classification to verification model")
        else:
            logger.error("Could not transfer encoder weights - verification model structure unexpected")

        # Replace model
        self.model = verification_model.to(self.device)

        # Create verification data loaders
        logger.info("Creating verification data loaders...")

        # Check if pre-computed features are available
        use_precomputed = False
        if hasattr(self.config, "precomputed_features") and self.config.precomputed_features.use_precomputed_features:
            precomputed_paths = await self._prepare_precomputed_features()
            if precomputed_paths:
                train_path, val_path = precomputed_paths
                self.train_loader, self.val_loader = await self._load_precomputed_data(train_path, val_path)
                use_precomputed = True

                # Load extractors from pre-computed features
                semantic_ext, style_ext, email_ext = load_extractors_from_h5(Path(train_path))
                if style_ext is not None:
                    self.semantic_extractor = semantic_ext
                    self.style_extractor = style_ext
                    self.email_extractor = email_ext
                    logger.info("✓ Loaded fitted extractors from pre-computed features")

        # Fall back to on-the-fly feature extraction if not using pre-computed
        if not use_precomputed:
            # Check if we have the required data
            if not hasattr(self, 'train_texts_by_author') or not self.train_texts_by_author:
                logger.error("Two-stage training requires train_texts_by_author data")
                raise ValueError("Cannot run two-stage training without author text data")

            logger.info(
                f"Processing {len(self.train_texts_by_author)} authors with "
                f"{sum(len(texts) for texts in self.train_texts_by_author.values())} training texts"
            )

            # Check if we already have extractors from classification stage
            if not hasattr(self, 'style_extractor') or self.style_extractor is None:
                # Load pre-fitted extractors (same logic as in create_classification_data_loaders)
                logger.info("Loading pre-fitted extractors for verification stage...")
                extractor_cache_path = Path("cache/reddit_extractors.pkl")

                if not extractor_cache_path.exists():
                    # Get extractor path from config
                    precomp_config = self.config.precomputed_features
                    gcs_extractor_path = precomp_config.gcs_val_extractors

                    if not gcs_extractor_path:
                        raise RuntimeError("gcs_val_extractors not specified in config - cannot download pre-fitted extractors")

                    logger.info(f"Downloading extractors from {gcs_extractor_path}")

                    try:
                        if self.storage.use_gcs:
                            self.storage.gcs.download_precomputed_features(gcs_extractor_path, str(extractor_cache_path))
                            logger.info("✓ Downloaded pre-fitted extractors")
                        else:
                            raise RuntimeError("GCS storage not available but extractors not found locally")
                    except Exception as e:
                        logger.error(f"Failed to download extractors: {e}")
                        raise RuntimeError(f"Cannot proceed without pre-fitted extractors: {e}")

                # Load the pre-fitted extractors
                with open(extractor_cache_path, "rb") as f:
                    extractors = pickle.load(f)  # Load without encoding parameter

                # Initialize semantic and email extractors fresh (no fitting required)
                self.semantic_extractor = SemanticFeatureExtractor()
                self.email_extractor = EmailPatternExtractor()

                # Only load the fitted style extractor from pickle
                self.style_extractor = extractors.get("style_extractor")
                logger.info("✓ Initialized fresh semantic and email extractors, loaded fitted style extractor")

                if self.style_extractor is None:
                    raise RuntimeError("Style extractor not found in downloaded file")

                logger.info("✓ Loaded pre-fitted extractors successfully")
            else:
                logger.info("Using extractors from classification stage")

            logger.info("Generating Reddit pairs for verification training...")

            # Generate Reddit pairs using the new pair generator
            pair_config = self.config.external_data.reddit_pairs_config
            if pair_config is None:
                # Use default config if not set
                pair_config = RedditPairsConfig()

            pair_generator = RedditPairGenerator(
                cache_dir=pair_config.cache_dir,
                positive_ratio=pair_config.positive_ratio,
                max_pairs_per_author=pair_config.max_pairs_per_author,
                min_samples_per_author=pair_config.min_samples_per_author,
                seed=pair_config.seed
            )

            # Generate deterministic pairs
            train_pairs = pair_generator.generate_pairs(self.train_texts_by_author, "train", pair_config.force_regenerate)
            val_pairs = pair_generator.generate_pairs(self.val_texts_by_author, "val", pair_config.force_regenerate)

            logger.info(f"Generated {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")

            # Create datasets using pre-generated pairs
            train_dataset = AuthorshipPairDataset.from_manifest(
                train_pairs,
                self.semantic_extractor,
                self.style_extractor,
                self.email_extractor,
            )
            val_dataset = AuthorshipPairDataset.from_manifest(
                val_pairs,
                self.semantic_extractor,
                self.style_extractor,
                self.email_extractor,
            )

            # Create data loaders
            # Note: For verification, we need num_workers=0 because the semantic feature
            # extractor uses CUDA and multiprocessing with CUDA requires special handling
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=0,  # Must be 0 for CUDA feature extraction
                pin_memory=self.dataloader_pin_memory,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=0,  # Must be 0 for CUDA feature extraction
                pin_memory=self.dataloader_pin_memory,
            )

        logger.info(f"Verification training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Verification validation samples: {len(self.val_loader.dataset)}")

        # Setup optimizer for verification with lower learning rate
        verification_lr = getattr(self.config.training, "verification_base_lr", 1e-6)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=verification_lr)
        logger.info(f"Verification optimizer: lr={verification_lr}")

        # Setup scheduler for verification
        if hasattr(self.config.training, "use_scheduler") and self.config.training.use_scheduler:
            verification_epochs = getattr(self.config.training, "verification_epochs", 15)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=verification_epochs, eta_min=1e-7)
            logger.info(f"Using CosineAnnealingLR scheduler for {verification_epochs} verification epochs")
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7)

        # Train verification model
        verification_epochs = getattr(self.config.training, "verification_epochs", 15)
        logger.info(f"Training verification model for {verification_epochs} epochs...")

        best_val_loss = float("inf")
        patience_counter = 0
        best_verification_threshold = 0.5

        for epoch in range(verification_epochs):
            logger.info(f"Verification Epoch {epoch + 1}/{verification_epochs}")

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            # Log metrics
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                f"Val AUC: {val_metrics['val_auc']:.4f}"
            )

            # Save metrics
            epoch_metrics = {
                **train_metrics,
                **val_metrics,
                "epoch": classification_epochs + epoch,
                "stage": "verification",
                "verification_epoch": epoch
            }
            self.training_history.append(epoch_metrics)

            # Log to metrics logger
            self.metrics_logger.log_metrics(epoch_metrics, step=classification_epochs + epoch)

            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_loss"])
            else:
                self.scheduler.step()

            # Early stopping check
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                best_verification_threshold = val_metrics.get("val_threshold", 0.5)
                self.save_checkpoint(
                    classification_epochs + epoch,
                    is_best=True,
                    best_threshold=best_verification_threshold
                )
            else:
                patience_counter += 1

            # Save periodic checkpoints during verification
            if (classification_epochs + epoch) % self.config.training.save_checkpoint_every == 0:
                self.save_checkpoint(classification_epochs + epoch, is_best=False)

            # Check for mode collapse
            if (val_metrics.get("val_sim_std", 1.0) < 0.001 and epoch > 5) or (
                val_metrics.get("val_embedding_dist_mean", 1.0) < 0.05 and epoch > 5
            ):
                logger.error(f"🚨 MODE COLLAPSE DETECTED! Stopping training at epoch {epoch + 1}")
                logger.error(f"   Similarity std: {val_metrics.get('val_sim_std', 'N/A'):.6f}")
                logger.error(f"   Embedding distance: {val_metrics.get('val_embedding_dist_mean', 'N/A'):.4f}")
                break

            if patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} verification epochs")
                break

        logger.info("=" * 60)
        logger.info("Two-stage training completed!")
        logger.info(f"Best classification accuracy: {best_val_acc:.4f}")
        logger.info(f"Best verification loss: {best_val_loss:.4f}")
        logger.info("=" * 60)

        # Save final training history
        self.save_training_history()

        # Log completion metrics
        completion_metrics = {
            "training_completed": 1,
            "total_epochs": classification_epochs + epoch + 1,
            "best_classification_accuracy": best_val_acc,
            "best_verification_loss": best_val_loss,
        }
        self.metrics_logger.log_metrics(completion_metrics)


async def main():
    """Main training function for Vertex AI."""
    # Get merged configuration
    experiment_config, vertex_config = get_merged_config()

    # Setup logging
    global logger
    logger = setup_logging(vertex_config)

    logger.info("=" * 50)
    logger.info("Vertex AI Authorship Verification Training")
    logger.info("=" * 50)
    logger.info(f"Experiment: {vertex_config.experiment_name}")
    logger.info(f"GCS Bucket: {vertex_config.gcs_bucket}")
    logger.info(f"Vertex AI Enabled: {vertex_config.vertex_ai_enabled}")

    # Initialize storage
    storage = HybridStorage(
        local_base_path=experiment_config.model_save_path,
        gcs_bucket=vertex_config.gcs_bucket if vertex_config.gcs_bucket else None,
        gcs_project=vertex_config.gcp_project if vertex_config.gcp_project else None,
        experiment_name=vertex_config.experiment_name,
        use_gcs=vertex_config.vertex_ai_enabled and bool(vertex_config.gcs_bucket),
    )

    # Initialize trainer
    trainer = VertexAuthTrainer(experiment_config, vertex_config, storage)

    # Load data
    logger.info("Loading training data...")
    await trainer.prepare_data(
        use_gcs=vertex_config.vertex_ai_enabled and bool(vertex_config.gcs_bucket), gcs_bucket=vertex_config.gcs_bucket
    )

    # Build model
    logger.info("Building model...")
    trainer.build_model()

    # Setup optimizer
    trainer.setup_optimizer()

    # Start training
    await trainer.train()

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    # Note: Hyperparameter search is handled by Vertex AI's HyperparameterTuningJob
    # which orchestrates multiple training runs with different parameters.
    # Each individual run still uses this script in regular training mode.
    #
    # To run hyperparameter search, use:
    # python scripts/submit_vertex_job.py --job-type hyperparam
    #
    # This script always runs a single training job with the configuration
    # provided via environment variables (including CONFIG_OVERRIDES).

    asyncio.run(main())
