"""Base class for baseline authorship verification models."""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, f1_score


class BaselineModel(ABC):
    """Abstract base class for baseline authorship verification models."""

    def __init__(self, model_name: str):
        """Initialize the baseline model.

        Args:
            model_name: Human-readable name for the model
        """
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Args:
            text: Input text to encode

        Returns:
            Embedding vector as numpy array
        """
        pass

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into embedding vectors.

        Args:
            texts: list of input texts to encode

        Returns:
            Batch of embedding vectors as numpy array (n_texts, embedding_dim)
        """
        embeddings = []
        for text in texts:
            embedding = self.encode_text(text)
            embeddings.append(embedding)
        return np.array(embeddings)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine") -> float:
        """Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ("cosine" or "euclidean")

        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
        elif metric == "euclidean":
            # Convert euclidean distance to similarity (higher = more similar)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def predict_pair(self, text1: str, text2: str, threshold: float = 0.5) -> Tuple[float, bool]:
        """Predict if two texts are by the same author.

        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold for same author classification

        Returns:
            Tuple of (similarity_score, is_same_author)
        """
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)
        similarity = self.compute_similarity(embedding1, embedding2)
        is_same_author = similarity > threshold
        return similarity, is_same_author

    def find_optimal_threshold(self, similarities: np.ndarray, labels: np.ndarray, metric: str = "f1") -> float:
        """Find the optimal threshold that maximizes the specified metric.

        Args:
            similarities: Array of similarity scores
            labels: Array of ground truth labels
            metric: Metric to optimize ("f1", "accuracy", "balanced_accuracy")

        Returns:
            Optimal threshold value
        """
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

    def get_threshold(self) -> Optional[float]:
        """Get the threshold to use for predictions.

        Returns:
            The saved threshold if available, otherwise None.
        """
        # Check if this model has a saved threshold (e.g., from training)
        if hasattr(self, "optimal_threshold") and self.optimal_threshold is not None:
            return self.optimal_threshold
        return None

    def evaluate(
        self,
        text_pairs: list[Tuple[str, str]],
        labels: list[int],
        calibration_pairs: Optional[list[Tuple[str, str]]] = None,
        calibration_labels: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Evaluate the model on a set of text pairs.

        Args:
            text_pairs: list of (text1, text2) pairs
            labels: list of ground truth labels (1 = same author, 0 = different author)

        Returns:
            dictionary containing evaluation metrics
        """
        similarities = []

        for text1, text2 in text_pairs:
            embedding1 = self.encode_text(text1)
            embedding2 = self.encode_text(text2)
            similarity = self.compute_similarity(embedding1, embedding2)
            similarities.append(similarity)

        similarities = np.array(similarities)
        labels = np.array(labels)

        # Check if model has a saved threshold from training
        saved_threshold = self.get_threshold()

        if saved_threshold is not None:
            # Use the saved threshold from training
            threshold = saved_threshold
        elif calibration_pairs is not None and calibration_labels is not None:
            # Use separate calibration set for threshold finding (no data leakage)
            calib_similarities = []
            for text1, text2 in calibration_pairs:
                embedding1 = self.encode_text(text1)
                embedding2 = self.encode_text(text2)
                similarity = self.compute_similarity(embedding1, embedding2)
                calib_similarities.append(similarity)

            calib_similarities = np.array(calib_similarities)
            calib_labels = np.array(calibration_labels)

            # Find optimal threshold on calibration set using balanced accuracy
            threshold = self.find_optimal_threshold(calib_similarities, calib_labels, metric="balanced_accuracy")
        else:
            # Fallback: use median threshold (will be symmetric but stable)
            threshold = np.median(similarities)

        predictions = (similarities > threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)

        try:
            auc = roc_auc_score(labels, similarities)
        except ValueError:
            auc = 0.0  # Handle case where all labels are the same

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        # Calculate additional statistics
        sim_mean = np.mean(similarities)
        sim_std = np.std(similarities)

        return {
            "model_name": self.model_name,
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": threshold,
            "similarity_mean": sim_mean,
            "similarity_std": sim_std,
            "n_pairs": len(text_pairs),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }
