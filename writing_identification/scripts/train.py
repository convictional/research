"""Training script for authorship verification model."""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, top_k_accuracy_score

# Initialize feature extractors and store as instance variables

from config.config import config, ExperimentConfig
from data.classification_dataset import AuthorClassificationDataset
from data.dataset import create_data_loaders
from data.extract_training_data import AuthorshipDataExtractor
from data.gcs_loader import GCSDataLoader
from data.precomputed_dataset import create_precomputed_data_loaders
from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
from features.email_patterns import EmailPatternExtractor
from models.siamese import SiameseNetwork, AuthorshipVerifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthorshipTrainer:
    """Trainer for authorship verification models."""

    def __init__(self, config_override: dict = None):
        """Initialize trainer with configuration."""
        if config_override:
            # Deep merge override dict into Pydantic config
            merged_dict = config.model_dump()
            self._deep_merge_dict(merged_dict, config_override)
            self.config = ExperimentConfig.model_validate(merged_dict)
        else:
            self.config = config

        # Set random seed for reproducibility
        torch.manual_seed(self.config.random_seed)

        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model components
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # Initialize feature extractors (will be set during training)
        self.semantic_extractor = None
        self.style_extractor = None
        self.email_extractor = None

        # Training state
        self.train_loader = None
        self.val_loader = None
        self.training_history = []

        # Create output directories
        self.setup_directories()

    def _deep_merge_dict(self, base: dict, override: dict):
        """Recursively merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value
                logger.info(f"Config override: {key} = {value}")

    def setup_directories(self):
        """Create necessary directories for saving outputs."""
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.results_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.data_path).mkdir(parents=True, exist_ok=True)

    async def prepare_data(
        self, texts_by_author: dict[str, list[str]] = None, use_gcs: bool = False, gcs_bucket: str = None
    ):
        """Prepare training and validation data."""
        # Handle Reddit dataset for classification
        if self.config.external_data.use_external_data and self.config.external_data.dataset_name == "reddit":
            self.prepare_reddit_data()
            return

        # Existing logic for other datasets
        if texts_by_author is None:
            if use_gcs and gcs_bucket:
                logger.info("Loading training data from GCS dataset...")
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

        # Split data into train/validation
        train_texts = {}
        val_texts = {}

        for author, texts in texts_by_author.items():
            validation_split = self.config.training.validation_split
            split_idx = int(len(texts) * (1 - validation_split))
            train_texts[author] = texts[:split_idx]
            val_texts[author] = texts[split_idx:]

        # Store texts for two-stage training if needed
        if self.config.training.training_mode == "two_stage":
            # Simply use the existing dictionaries - no reconstruction needed!
            self.train_texts_by_author = train_texts  # Already a dict[str, List[str]]
            self.val_texts_by_author = val_texts  # Already a dict[str, List[str]]
            logger.info(f"Prepared {len(train_texts)} authors for two-stage training")

        # Create data loaders with hard-negative mining enabled for training
        self.train_loader, self.val_loader = create_data_loaders(
            train_texts,
            val_texts,
            batch_size=self.config.training.batch_size,
            hard_negative=True,
            hard_negative_top_k=3,  # consider a small pool of hardest negatives
        )

        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

    def prepare_reddit_data(self):
        """Prepare Reddit dataset for classification training."""
        from data.reddit_dataset import load_reddit_dataset_for_training

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

        self.semantic_extractor = SemanticFeatureExtractor()
        self.style_extractor = StyleFeatureExtractor()
        self.email_extractor = EmailPatternExtractor()

        # Fit style extractor on all texts
        all_texts = []
        for author_texts in list(train_texts.values()) + list(val_texts.values()):
            all_texts.extend(author_texts)

        logger.info(f"Fitting style extractor on {len(all_texts)} total texts...")
        self.style_extractor.fit(all_texts)
        logger.info("Style extractor fitted successfully")

        # Create classification datasets
        train_dataset = AuthorClassificationDataset(
            train_texts, self.semantic_extractor, self.style_extractor, self.email_extractor
        )
        val_dataset = AuthorClassificationDataset(
            val_texts, self.semantic_extractor, self.style_extractor, self.email_extractor
        )

        # Create data loaders
        num_workers = 0 if torch.backends.mps.is_available() else 4

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        logger.info("Classification datasets created:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")

        return train_loader, val_loader

    def build_model(self):
        """Build and initialize the model."""
        # Calculate feature dimensions
        semantic_dim = self.config.model.embedding_dim
        style_dim = self.config.model.style_feature_dim

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

    def build_classification_model(self, semantic_dim: int, style_dim: int):
        """Build classification model with ArcFace head."""
        from models.classification import AuthorClassifier

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
        return model

    def count_parameters(self) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def setup_optimizer(self, stage: str = "classification"):
        """Setup optimizer and learning rate scheduler for specific training stage."""
        if stage == "classification":
            self.setup_classification_optimizer()
        elif stage == "verification":
            self.setup_verification_optimizer()
        else:
            # Fallback for backward compatibility
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=1e-5,
                eps=1e-8,
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            )

    def setup_classification_optimizer(self):
        """Setup optimizer and scheduler for classification stage."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=1e-5,
            eps=1e-8,
        )

        if not self.config.training.use_scheduler or self.config.training.classification_scheduler == "none":
            self.scheduler = None
        elif self.config.training.classification_scheduler == "onecycle":
            steps_per_epoch = len(self.train_loader)
            total_steps = steps_per_epoch * self.config.training.classification_epochs
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.classification_max_lr,
                total_steps=total_steps,
                pct_start=0.3,  # 30% warmup
                anneal_strategy="cosine",
            )
        elif self.config.training.classification_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.scheduler_t_max, eta_min=1e-7
            )
        else:
            # Default: ReduceLROnPlateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            )

    def setup_verification_optimizer(self):
        """Setup optimizer and scheduler for verification stage."""
        # Use lower learning rate for verification fine-tuning
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.verification_base_lr,
            weight_decay=1e-5,
            eps=1e-8,
        )

        if not self.config.training.use_scheduler or self.config.training.verification_scheduler == "none":
            self.scheduler = None
        elif self.config.training.verification_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.verification_epochs, eta_min=1e-8
            )
        elif self.config.training.verification_scheduler == "warmup_cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.config.training.scheduler_warmup_epochs, T_mult=1, eta_min=1e-8
            )
        else:
            # Default: ReduceLROnPlateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=3,  # Shorter patience for verification
                min_lr=1e-8,
            )

    def _step_scheduler(self, val_metrics: dict, epoch: int):
        """Step the learning rate scheduler based on scheduler type."""
        if self.scheduler is None:
            return

        scheduler_type = type(self.scheduler).__name__

        if scheduler_type == "ReduceLROnPlateau":
            self.scheduler.step(val_metrics["val_loss"])
        elif scheduler_type in ["OneCycleLR"]:
            # OneCycleLR is stepped per batch, not per epoch - skip here
            pass
        else:
            # CosineAnnealingLR, CosineAnnealingWarmRestarts, etc.
            self.scheduler.step()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Handle both tuple format (external data) and dict format (internal data)
            if isinstance(batch, (list, tuple)):
                # External data format: (features1, features2, labels)
                features1, features2, labels = batch
                features1 = features1.to(self.device)
                features2 = features2.to(self.device)
                labels = labels.to(self.device)

                # Split combined features back into semantic and style using config
                semantic_dim = self.config.model.embedding_dim
                style_dim = self.config.model.style_feature_dim

                semantic_features1 = features1[:, :semantic_dim]
                style_features1 = features1[:, semantic_dim : semantic_dim + style_dim]
                semantic_features2 = features2[:, :semantic_dim]
                style_features2 = features2[:, semantic_dim : semantic_dim + style_dim]

            else:
                # Internal data format: dict
                batch = {k: v.to(self.device) for k, v in batch.items()}
                semantic_features1 = batch["semantic_features1"]
                style_features1 = batch["style_features1"]
                semantic_features2 = batch["semantic_features2"]
                style_features2 = batch["style_features2"]
                labels = batch["label"]

            # Forward pass
            self.optimizer.zero_grad()

            outputs = self.model(
                semantic_features1,
                style_features1,
                semantic_features2,
                style_features2,
                labels,
            )

            loss = outputs["loss"]

            # Backward pass with gradient clipping
            loss.backward()

            # Clip gradients to prevent explosion with higher learning rate
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Log if gradients are large
            if grad_norm > 10.0:
                logger.warning(f"Large gradient norm: {grad_norm:.2f}")

            self.optimizer.step()

            # Step OneCycleLR scheduler per batch
            if self.scheduler is not None and type(self.scheduler).__name__ == "OneCycleLR":
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}

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

            # Compute loss
            logits = outputs["logits"]
            loss = criterion(logits, author_ids)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Step OneCycleLR scheduler per batch
            if self.scheduler is not None and type(self.scheduler).__name__ == "OneCycleLR":
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Track accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(author_ids.cpu().numpy())

            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Compute metrics
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_predictions)

        # Get scale factor for unscaled loss (ArcFace/CosFace use scale factor s)
        scale_factor = 1.0
        if hasattr(self.model, "classification_head") and hasattr(self.model.classification_head, "s"):
            scale_factor = self.model.classification_head.s
        unscaled_loss = avg_loss / scale_factor

        return {"train_loss": avg_loss, "train_loss_unscaled": unscaled_loss, "train_accuracy": accuracy}

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

                # Forward pass
                outputs = self.model(semantic_features, style_features, author_ids)
                logits = outputs["logits"]

                # Compute loss
                loss = criterion(logits, author_ids)
                total_loss += loss.item()
                num_batches += 1

                # Track predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(author_ids.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_predictions)

        # Get scale factor for unscaled loss
        scale_factor = 1.0
        if hasattr(self.model, "classification_head") and hasattr(self.model.classification_head, "s"):
            scale_factor = self.model.classification_head.s
        unscaled_loss = avg_loss / scale_factor

        # Top-5 accuracy if we have enough classes
        num_classes = len(set(all_labels))
        if num_classes >= 5:
            all_logits_np = np.vstack(all_logits)
            top5_accuracy = top_k_accuracy_score(all_labels, all_logits_np, k=5)
        else:
            top5_accuracy = accuracy

        return {
            "val_loss": avg_loss,
            "val_loss_unscaled": unscaled_loss,
            "val_accuracy": accuracy,
            "val_top5_accuracy": top5_accuracy,
        }

    def validate_epoch(self, epoch: int) -> dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_similarities = []
        all_labels = []
        all_embeddings1 = []
        all_embeddings2 = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both tuple format (external data) and dict format (internal data)
                if isinstance(batch, (list, tuple)):
                    # External data format: (features1, features2, labels)
                    features1, features2, labels = batch
                    features1 = features1.to(self.device)
                    features2 = features2.to(self.device)
                    labels = labels.to(self.device)

                    # Store labels consistently for downstream use
                    labels_tensor = labels
                    labels_np = labels.cpu().numpy()

                    # Split combined features back into semantic and style using config
                    semantic_dim = self.config.model.embedding_dim
                    style_dim = self.config.model.style_feature_dim

                    semantic_features1 = features1[:, :semantic_dim]
                    style_features1 = features1[:, semantic_dim : semantic_dim + style_dim]
                    semantic_features2 = features2[:, :semantic_dim]
                    style_features2 = features2[:, semantic_dim : semantic_dim + style_dim]

                else:
                    # Internal data format: dict
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    semantic_features1 = batch["semantic_features1"]
                    style_features1 = batch["style_features1"]
                    semantic_features2 = batch["semantic_features2"]
                    style_features2 = batch["style_features2"]

                    # Store labels consistently for downstream use
                    labels = batch["label"]
                    labels_np = labels.cpu().numpy()

                # Forward pass
                outputs = self.model(
                    semantic_features1,
                    style_features1,
                    semantic_features2,
                    style_features2,
                    labels,
                )

                total_loss += outputs["loss"].item()

                # Collect predictions for metrics
                similarities = outputs["similarity"].cpu().numpy()
                # Use the pre-computed labels_np from above

                # Collect embeddings for diversity analysis
                embeddings1 = outputs["embedding1"].cpu().numpy()
                embeddings2 = outputs["embedding2"].cpu().numpy()

                all_similarities.extend(similarities)
                all_labels.extend(labels_np)  # Use the pre-computed numpy labels
                all_embeddings1.extend(embeddings1)
                all_embeddings2.extend(embeddings2)

        avg_loss = total_loss / len(self.val_loader)

        # Find optimal threshold that maximizes F1 score
        # This avoids the median threshold issue that creates symmetric confusion matrices
        threshold = self._find_optimal_threshold(all_similarities, all_labels)
        predictions = [1 if sim > threshold else 0 for sim in all_similarities]
        accuracy = accuracy_score(all_labels, predictions)

        try:
            auc = roc_auc_score(all_labels, all_similarities)
        except ValueError:
            auc = 0.0  # Handle case where all labels are the same

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average="binary")

        # Calculate embedding diversity metrics
        all_embeddings1 = np.array(all_embeddings1)
        all_embeddings2 = np.array(all_embeddings2)
        all_embeddings = np.vstack([all_embeddings1, all_embeddings2])

        # Compute pairwise distances between embeddings (sample subset for efficiency)
        sample_size = min(200, len(all_embeddings))
        sample_indices = np.random.choice(len(all_embeddings), sample_size, replace=False)
        sample_embeddings = all_embeddings[sample_indices]

        # Compute mean pairwise L2 distance
        pairwise_distances = pdist(sample_embeddings, metric="euclidean")
        mean_pairwise_distance = np.mean(pairwise_distances)
        std_pairwise_distance = np.std(pairwise_distances)

        # Mode collapse detection
        sim_std = np.std(all_similarities)
        sim_mean = np.mean(all_similarities)

        # Check for mode collapse indicators
        if sim_std < 0.001:
            logger.warning(f"⚠️  POTENTIAL MODE COLLAPSE: Similarity std = {sim_std:.6f} < 0.001")
        if mean_pairwise_distance < 0.1:
            logger.warning(f"⚠️  POTENTIAL MODE COLLAPSE: Mean pairwise distance = {mean_pairwise_distance:.4f} < 0.1")
        if sim_mean > 0.99:
            logger.warning(f"⚠️  POTENTIAL MODE COLLAPSE: Similarity mean = {sim_mean:.6f} > 0.99")

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_auc": auc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_threshold": threshold,
            "val_sim_mean": sim_mean,
            "val_sim_std": sim_std,
            "val_embedding_dist_mean": mean_pairwise_distance,
            "val_embedding_dist_std": std_pairwise_distance,
        }

    def _find_optimal_threshold(self, similarities, labels, metric="f1"):
        """Find the optimal threshold that maximizes the specified metric.

        Args:
            similarities: List or array of similarity scores
            labels: List or array of ground truth labels
            metric: Metric to optimize ("f1", "accuracy", "balanced_accuracy")

        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

        similarities = np.array(similarities)
        labels = np.array(labels)

        # Try different thresholds from min to max similarity
        thresholds = np.linspace(similarities.min(), similarities.max(), 100)

        best_score = -1
        best_threshold = np.median(similarities)  # Fallback to median

        for thresh in thresholds:
            preds = (similarities > thresh).astype(int)

            if metric == "f1":
                score = f1_score(labels, preds)
            elif metric == "accuracy":
                score = accuracy_score(labels, preds)
            elif metric == "balanced_accuracy":
                # Balanced accuracy = (sensitivity + specificity) / 2
                cm = confusion_matrix(labels, preds)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    score = (sensitivity + specificity) / 2
                else:
                    score = 0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = thresh

        return best_threshold

    async def train_two_stage(self):
        """Two-stage training: classification then verification."""
        logger.info("Starting two-stage training...")

        # Stage 1: Classification training
        logger.info("=" * 60)
        logger.info("STAGE 1: Classification Training")
        logger.info("=" * 60)

        # Setup optimizer for classification stage
        self.setup_optimizer(stage="classification")

        classification_epochs = getattr(self.config.training, "classification_epochs", 5)
        logger.info(f"Training classification model for {classification_epochs} epochs...")

        best_val_acc = 0.0
        best_classification_epoch = -1
        best_classification_threshold = 0.5  # Default threshold

        for epoch in range(classification_epochs):
            logger.info(f"Classification Epoch {epoch + 1}/{classification_epochs}")

            train_metrics = self.train_classification_epoch(epoch)
            val_metrics = self.validate_classification_epoch(epoch)

            # Log metrics (with unscaled loss for clarity)
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

            # Track best model
            if val_metrics["val_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["val_accuracy"]
                best_classification_epoch = epoch + 1  # Store 1-based for logging
                best_classification_threshold = val_metrics.get("val_threshold", 0.5)
                self.save_checkpoint(
                    epoch, is_best=True, best_threshold=best_classification_threshold
                )  # Saves as best_model.pt (classification stage)
                logger.info(
                    f"New best classification accuracy: {best_val_acc:.4f} at epoch {best_classification_epoch}"
                )

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
            # Note: Don't touch self.optimizer - we'll rebuild it after creating verification model
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

        # Get dimensions
        semantic_dim = self.model.semantic_dim if hasattr(self.model, "semantic_dim") else 768
        style_dim = self.model.style_dim if hasattr(self.model, "style_dim") else 1169

        # Build verification model with pretrained encoder
        verification_model = self.build_verification_model(semantic_dim, style_dim)

        # Transfer encoder weights with proper attribute checking
        if hasattr(verification_model, "siamese_network") and hasattr(verification_model.siamese_network, "encoder"):
            verification_model.siamese_network.encoder.load_state_dict(pretrained_encoder.state_dict())
            logger.info("Transferred encoder weights from classification to verification model")
        else:
            logger.error("Could not transfer encoder weights - verification model structure unexpected")

        # Replace model
        self.model = verification_model.to(self.device)

        # Create verification data loaders
        logger.info("Creating verification data loaders...")

        # Check if pre-computed features are available and configured
        use_precomputed = False
        if hasattr(self.config, "precomputed_features") and self.config.precomputed_features.use_precomputed_features:
            train_path = self.config.precomputed_features.train_features_path
            val_path = self.config.precomputed_features.val_features_path

            if train_path and val_path:
                train_path_obj = Path(train_path)
                val_path_obj = Path(val_path)

                if train_path_obj.exists() and val_path_obj.exists():
                    logger.info("=" * 60)
                    logger.info("Using pre-computed features for fast verification training!")
                    logger.info(f"Train features: {train_path}")
                    logger.info(f"Val features: {val_path}")
                    logger.info("=" * 60)

                    # Use MPS-compatible settings
                    use_gpu_cache = False  # MPS doesn't support CUDA caching
                    num_workers = 0  # MPS requires 0 workers
                    pin_memory = False  # MPS doesn't support memory pinning

                    self.train_loader, self.val_loader = create_precomputed_data_loaders(
                        train_h5_path=train_path,
                        val_h5_path=val_path,
                        batch_size=self.config.training.batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        use_gpu_cache=use_gpu_cache,
                        device=self.device,
                    )
                    use_precomputed = True

                    # Load extractors from pre-computed features if available
                    from data.precomputed_dataset import load_extractors_from_h5

                    semantic_ext, style_ext, email_ext = load_extractors_from_h5(Path(train_path))
                    if style_ext is not None:
                        self.semantic_extractor = semantic_ext
                        self.style_extractor = style_ext
                        self.email_extractor = email_ext
                        logger.info("✓ Loaded fitted extractors from pre-computed features")
                    else:
                        logger.warning("No extractors found in pre-computed features - inference will fail!")

                    logger.info("Loaded pre-computed features successfully!")
                else:
                    logger.warning("Pre-computed feature files not found, falling back to on-the-fly computation")

        # Fall back to on-the-fly feature extraction if not using pre-computed
        if not use_precomputed:
            # Check if we have the required data
            if not hasattr(self, "train_texts_by_author") or not self.train_texts_by_author:
                logger.error("Two-stage training requires train_texts_by_author data")
                raise ValueError("Cannot run two-stage training without author text data")

            logger.info(
                f"Processing {len(self.train_texts_by_author)} authors with "
                f"{sum(len(texts) for texts in self.train_texts_by_author.values())} training texts"
            )

            # Initialize extractors for on-the-fly feature extraction
            from features.extractors import StyleFeatureExtractor, SemanticFeatureExtractor
            from features.email_patterns import EmailPatternExtractor
            from data.dataset import AuthorshipPairDataset

            self.semantic_extractor = SemanticFeatureExtractor()
            self.style_extractor = StyleFeatureExtractor()
            self.email_extractor = EmailPatternExtractor()

            # Fit style extractor on training texts
            all_train_texts = []
            for texts in self.train_texts_by_author.values():
                all_train_texts.extend(texts)

            logger.info(f"Fitting style extractor on {len(all_train_texts)} texts...")
            self.style_extractor.fit(all_train_texts)
            logger.info("Style extractor fitted successfully")

            logger.info("Generating Reddit pairs for verification training...")

            # Generate Reddit pairs using the new pair generator
            from data.reddit_pairs import RedditPairGenerator

            pair_config = self.config.external_data.reddit_pairs_config
            if pair_config is None:
                # Use default config if not set
                from config.config import RedditPairsConfig

                pair_config = RedditPairsConfig()

            pair_generator = RedditPairGenerator(
                cache_dir=pair_config.cache_dir,
                positive_ratio=pair_config.positive_ratio,
                max_pairs_per_author=pair_config.max_pairs_per_author,
                min_samples_per_author=pair_config.min_samples_per_author,
                seed=pair_config.seed,
            )

            # Generate deterministic pairs
            train_pairs = pair_generator.generate_pairs(
                self.train_texts_by_author, "train", pair_config.force_regenerate
            )
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
            num_workers = 0 if torch.backends.mps.is_available() else 4
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,
            )
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
            )

        logger.info(f"Verification training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Verification validation samples: {len(self.val_loader.dataset)}")
        logger.info("Data loaders ready for verification training!")

        # Reinitialize optimizer for verification training
        self.setup_optimizer(stage="verification")

        # Stage 2 training
        verification_epochs = getattr(self.config.training, "verification_epochs", 3)
        logger.info(f"Fine-tuning verification model for {verification_epochs} epochs...")

        best_val_auc = 0.0
        best_verification_threshold = 0.5  # Default threshold

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
            }
            self.training_history.append(epoch_metrics)

            # Track best model
            if val_metrics["val_auc"] > best_val_auc:
                best_val_auc = val_metrics["val_auc"]
                best_verification_threshold = val_metrics.get("val_threshold", 0.5)
                # Note: This will overwrite Stage 1's best_model.pt - document this behavior
                self.save_checkpoint(
                    classification_epochs + epoch, is_best=True, best_threshold=best_verification_threshold
                )  # Overwrites best_model.pt with verification model
                logger.info(
                    f"New best verification AUC: {best_val_auc:.4f} with threshold: {best_verification_threshold:.4f}"
                )
                # Note: best_model.pt now contains the verification model, not classification

            # Save periodic checkpoints during verification
            if (classification_epochs + epoch) % self.config.training.save_checkpoint_every == 0:
                self.save_checkpoint(classification_epochs + epoch, is_best=False)

        logger.info("=" * 60)
        logger.info("Two-stage training completed!")
        logger.info(f"Best classification accuracy: {best_val_acc:.4f}")
        logger.info(f"Best verification AUC: {best_val_auc:.4f}")
        logger.info("=" * 60)

        # Training history is already saved after each epoch

    async def train(self, num_epochs: int = None):
        """Main training loop."""
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

        logger.info(f"Starting training for {num_epochs} epochs...")

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

            # Log metrics based on training mode
            if self.config.training.training_mode == "classification":
                logger.info(
                    f"Train Loss: {train_metrics['train_loss']:.4f} "
                    f"(unscaled: {train_metrics.get('train_loss_unscaled', train_metrics['train_loss']):.4f}), "
                    f"Train Acc: {train_metrics['train_accuracy']:.4f}"
                )
                logger.info(
                    f"Val Loss: {val_metrics['val_loss']:.4f} "
                    f"(unscaled: {val_metrics.get('val_loss_unscaled', val_metrics['val_loss']):.4f}), "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f}"
                )
                if "val_top5_accuracy" in val_metrics:
                    logger.info(f"Val Top-5 Acc: {val_metrics['val_top5_accuracy']:.4f}")
            else:
                logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
                logger.info(
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                    f"Val AUC: {val_metrics['val_auc']:.4f}"
                )

            # Learning rate scheduling
            self._step_scheduler(val_metrics, epoch)

            # Early stopping check
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                best_threshold = val_metrics.get("val_threshold", 0.5)
                self.save_checkpoint(epoch, is_best=True, best_threshold=best_threshold)
            else:
                patience_counter += 1

            # Check for mode collapse only in verification mode
            if self.config.training.training_mode != "classification":
                if (val_metrics.get("val_sim_std", 1.0) < 0.001 and epoch > 5) or (
                    val_metrics.get("val_embedding_dist_mean", 1.0) < 0.05 and epoch > 5
                ):
                    logger.error(f"🚨 MODE COLLAPSE DETECTED! Stopping training at epoch {epoch + 1}")
                    logger.error(f"   Similarity std: {val_metrics.get('val_sim_std', 'N/A'):.6f}")
                    logger.error(f"   Embedding distance: {val_metrics.get('val_embedding_dist_mean', 'N/A'):.4f}")
                    break

            # Save regular checkpoint
            if epoch % self.config.training.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.info("Training completed!")
        self.save_training_history()

    def save_checkpoint(self, epoch: int, is_best: bool = False, best_threshold: float = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else self.config,
            "training_history": self.training_history,
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

        # Save regular checkpoint
        checkpoint_path = Path(self.config.model_save_path) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = Path(self.config.model_save_path) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved at epoch {epoch}")

    def save_training_history(self):
        """Save training history to JSON."""
        history_path = (
            Path(self.config.results_path) / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Convert any tensor values to lists for JSON serialization
        serializable_history = []
        for epoch_data in self.training_history:
            epoch_dict = {}
            for key, value in epoch_data.items():
                if hasattr(value, "item"):  # Tensor with single value
                    epoch_dict[key] = value.item()
                else:
                    epoch_dict[key] = value
            serializable_history.append(epoch_dict)

        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)

        logger.info(f"Training history saved to {history_path}")


async def main():
    """Main training function."""
    import argparse
    from config.experiment_configs import get_experiment_config

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train authorship verification model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment configuration name (e.g., reddit_classification, balanced, attention)",
    )
    parser.add_argument(
        "--use-gcs",
        action="store_true",
        help="Use GCS dataset instead of database",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default=None,
        help="GCS bucket for dataset",
    )

    args = parser.parse_args()

    # Load experiment config if specified
    config_override = None
    if args.config:
        logger.info(f"Loading experiment config: {args.config}")
        config_override = get_experiment_config(args.config)

    # Initialize trainer with config
    trainer = AuthorshipTrainer(config_override=config_override)

    # Prepare data
    await trainer.prepare_data(use_gcs=args.use_gcs, gcs_bucket=args.gcs_bucket)

    # Build model
    trainer.build_model()

    # Setup optimizer
    trainer.setup_optimizer()

    # Train model
    await trainer.train()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
