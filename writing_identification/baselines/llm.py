"""LLM-based baseline model for authorship verification."""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Tuple, Any, Literal
import numpy as np
from pydantic import BaseModel, Field

# Add experiments directory to Python path for common module access
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.instruct_llm import ainstruct_llm, set_async_instructor_client
from config.llm_settings import llm_settings
from .base import BaselineModel

logger = logging.getLogger(__name__)


class AuthorshipVerificationResponse(BaseModel):
    """Response model for LLM authorship verification."""

    prediction: Literal["same", "different"] = Field(
        description="Whether the two texts were written by the same author or different authors"
    )
    reasoning: str = Field(description="Detailed explanation of the reasoning behind the prediction")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, rate_per_minute: int):
        self.rate_per_minute = rate_per_minute
        self.tokens = rate_per_minute
        self.max_tokens = rate_per_minute
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * (self.rate_per_minute / 60))
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            # Wait until we have a token
            wait_time = (1 - self.tokens) * (60 / self.rate_per_minute)
            logger.debug(f"Rate limited, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            self.tokens = 0


class LLMBaseline(BaselineModel):
    """LLM baseline model for authorship verification."""

    def __init__(self, model_name: str):
        """Initialize LLM model.

        Args:
            model_name: Anthropic model name (e.g., claude-3-5-haiku-20241022)
        """
        display_name = self._get_display_name(model_name)
        super().__init__(display_name)

        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(llm_settings.max_concurrent_requests)
        self.rate_limiter = RateLimiter(llm_settings.rate_limit_per_minute)
        self._client_initialized = False
        self._cache: dict[Tuple[str, str], AuthorshipVerificationResponse] = {}

    def _get_display_name(self, model_name: str) -> str:
        """Convert model name to display name."""
        if "haiku" in model_name:
            return "Claude Haiku 3.5"
        elif "sonnet-4" in model_name:
            return "Claude Sonnet 4"
        elif "opus-4" in model_name:
            return "Claude Opus 4.1"
        else:
            return f"Claude ({model_name})"

    def _ensure_client_initialized(self):
        """Ensure the async instructor client is initialized."""
        if not self._client_initialized:
            set_async_instructor_client(llm_model=self.model_name, api_key=llm_settings.anthropic_api_key)
            self._client_initialized = True

    def encode_text(self, text: str) -> np.ndarray:
        """Not used for LLM baseline - texts are compared directly."""
        # Return dummy embedding for compatibility with base class
        return np.array([0.0])

    async def _compare_texts_llm(self, text1: str, text2: str) -> AuthorshipVerificationResponse:
        """Compare two texts using LLM with rate limiting and retries."""

        # Check cache first
        cache_key = (text1[:100], text2[:100])  # Use first 100 chars as cache key
        if cache_key in self._cache:
            cached_response = self._cache[cache_key]
            return AuthorshipVerificationResponse(
                prediction=cached_response.prediction,
                reasoning=f"Cached result: {cached_response.reasoning}",
                confidence=cached_response.confidence,
            )

        async with self.semaphore:
            await self.rate_limiter.acquire()

            for attempt in range(llm_settings.max_retries):
                try:
                    self._ensure_client_initialized()

                    system_prompt = """You are an expert in authorship verification. Your task is to determine if two text samples were written by the same person or different people.

Consider these factors:
- Writing style and voice
- Vocabulary choices and complexity
- Sentence structure and length patterns
- Punctuation habits
- Tone and formality level
- Topic knowledge and expertise level
- Grammar patterns and common errors

Be thorough in your analysis but concise in your reasoning."""

                    user_prompt = f"""Analyze these two text samples and determine if they were written by the same author:

TEXT 1:
{text1}

TEXT 2:
{text2}

Provide your prediction ("same" or "different"), detailed reasoning, and a confidence score between 0.0 and 1.0."""

                    response = await ainstruct_llm(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_model=AuthorshipVerificationResponse,
                        llm_model=self.model_name,
                        temperature=llm_settings.temperature,
                        max_tokens=llm_settings.max_tokens,
                    )

                    # Cache the full response
                    self._cache[cache_key] = response

                    return response

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < llm_settings.max_retries - 1:
                        wait_time = llm_settings.retry_delay * (llm_settings.backoff_factor**attempt)
                        await asyncio.sleep(wait_time)
                    else:
                        raise

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine") -> float:
        """Not used for LLM baseline - similarity computed in async method."""
        return 0.5  # Dummy value

    async def evaluate_async(
        self,
        text_pairs: list[Tuple[str, str]],
        labels: list[int],
        return_details: bool = False,
        original_pairs: list[Tuple[str, str]] = None,
    ) -> dict[str, Any]:
        """Async evaluation method for LLM baseline."""
        logger.info(f"Evaluating {self.model_name} on {len(text_pairs)} pairs...")

        valid_results = []
        detailed_results = []  # Store full details for CSV output
        failed_count = 0

        # Process in batches to avoid overwhelming the API
        batch_size = min(50, len(text_pairs))

        for i in range(0, len(text_pairs), batch_size):
            batch = text_pairs[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            # Create tasks for parallel processing
            tasks = []
            for text1, text2 in batch:
                task = self._compare_texts_llm(text1, text2)
                tasks.append(task)

            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                pair_idx = i + j
                if isinstance(result, Exception):
                    logger.error(f"Error in batch processing for pair {pair_idx}: {result}")
                    failed_count += 1
                    if return_details:
                        detail_entry = {
                            "pair_idx": pair_idx,
                            "text1_sanitized": batch[j][0][:100] + "..." if len(batch[j][0]) > 100 else batch[j][0],
                            "text2_sanitized": batch[j][1][:100] + "..." if len(batch[j][1]) > 100 else batch[j][1],
                            "true_label": batch_labels[j],
                            "predicted_label": None,
                            "confidence": None,
                            "reasoning": f"Error: {str(result)}",
                        }
                        # Add original text if provided
                        if original_pairs:
                            detail_entry["text1_raw"] = (
                                original_pairs[pair_idx][0][:100] + "..."
                                if len(original_pairs[pair_idx][0]) > 100
                                else original_pairs[pair_idx][0]
                            )
                            detail_entry["text2_raw"] = (
                                original_pairs[pair_idx][1][:100] + "..."
                                if len(original_pairs[pair_idx][1]) > 100
                                else original_pairs[pair_idx][1]
                            )
                        detailed_results.append(detail_entry)
                else:
                    valid_results.append((result, batch_labels[j]))
                    if return_details:
                        detail_entry = {
                            "pair_idx": pair_idx,
                            "text1_sanitized": batch[j][0][:100] + "..." if len(batch[j][0]) > 100 else batch[j][0],
                            "text2_sanitized": batch[j][1][:100] + "..." if len(batch[j][1]) > 100 else batch[j][1],
                            "true_label": batch_labels[j],
                            "predicted_label": 1 if result.prediction == "same" else 0,
                            "confidence": result.confidence,
                            "reasoning": result.reasoning,
                        }
                        # Add original text if provided
                        if original_pairs:
                            detail_entry["text1_raw"] = (
                                original_pairs[pair_idx][0][:100] + "..."
                                if len(original_pairs[pair_idx][0]) > 100
                                else original_pairs[pair_idx][0]
                            )
                            detail_entry["text2_raw"] = (
                                original_pairs[pair_idx][1][:100] + "..."
                                if len(original_pairs[pair_idx][1]) > 100
                                else original_pairs[pair_idx][1]
                            )
                        detailed_results.append(detail_entry)

            logger.info(f"Processed batch {i // batch_size + 1}/{(len(text_pairs) + batch_size - 1) // batch_size}")

        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} pairs out of {len(text_pairs)}")

        if not valid_results:
            raise ValueError("No valid results obtained from LLM evaluation")

        # Split into separate lists for processing
        responses, valid_labels = zip(*valid_results)

        # Extract data for analysis
        confidences = [r.confidence for r in responses]
        predictions = [r.prediction for r in responses]
        pred_labels = [1 if pred == "same" else 0 for pred in predictions]

        # Convert to numpy arrays
        confidences = np.array(confidences)
        valid_labels = np.array(valid_labels)
        pred_labels = np.array(pred_labels)

        # Calculate metrics on ALL predictions (raw)
        raw_metrics = self._calculate_metrics(valid_labels, pred_labels, confidences, "raw")

        # Calculate metrics on high-confidence predictions only
        high_confidence_threshold = 0.7  # Can be made configurable
        high_conf_mask = confidences >= high_confidence_threshold

        if np.sum(high_conf_mask) > 0:
            high_conf_labels = valid_labels[high_conf_mask]
            high_conf_predictions = pred_labels[high_conf_mask]
            high_conf_confidences = confidences[high_conf_mask]

            high_conf_metrics = self._calculate_metrics(
                high_conf_labels, high_conf_predictions, high_conf_confidences, "high_confidence"
            )

            logger.info(
                f"High confidence predictions: {np.sum(high_conf_mask)}/{len(valid_results)} "
                f"({100 * np.sum(high_conf_mask) / len(valid_results):.1f}%)"
            )
        else:
            logger.warning("No high-confidence predictions found")
            high_conf_metrics = {}

        # Return standard metrics for compatibility with BaselineEvaluator
        # Use raw metrics as the primary metrics
        result = {
            "model_name": self.model_name,
            "accuracy": raw_metrics["raw_accuracy"],
            "auc": raw_metrics["raw_auc"],
            "precision": raw_metrics["raw_precision"],
            "recall": raw_metrics["raw_recall"],
            "f1": raw_metrics["raw_f1"],
            "threshold": 0.5,  # LLMs use binary predictions, not similarity threshold
            "similarity_mean": raw_metrics["raw_confidence_mean"],
            "similarity_std": raw_metrics["raw_confidence_std"],
            "n_pairs": raw_metrics["raw_n_pairs"],
            "confusion_matrix": raw_metrics["raw_confusion_matrix"],
            # Additional LLM-specific metrics
            "n_pairs_total": len(text_pairs),
            "n_pairs_successful": len(valid_results),
            "n_pairs_failed": failed_count,
            "high_confidence_threshold": high_confidence_threshold,
            **{k: v for k, v in raw_metrics.items() if k not in ["raw_confusion_matrix"]},
            **high_conf_metrics,
        }

        # Add detailed results if requested
        if return_details:
            result["detailed_predictions"] = detailed_results

        return result

    def _calculate_metrics(
        self, labels: np.ndarray, predictions: np.ndarray, confidences: np.ndarray, prefix: str
    ) -> dict[str, Any]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)

        try:
            auc = roc_auc_score(labels, confidences)
        except ValueError:
            auc = 0.0

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge case where only one class is present
            if len(np.unique(labels)) == 1:
                if labels[0] == 0:  # All negative
                    tn, fp, fn, tp = len(labels), 0, 0, 0
                else:  # All positive
                    tn, fp, fn, tp = 0, 0, 0, len(labels)
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

        # Additional statistics
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences)

        return {
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_auc": auc,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": f1,
            f"{prefix}_confidence_mean": conf_mean,
            f"{prefix}_confidence_std": conf_std,
            f"{prefix}_n_pairs": len(labels),
            f"{prefix}_confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

    def evaluate(
        self,
        text_pairs: list[Tuple[str, str]],
        labels: list[int],
        return_details: bool = False,
        original_pairs: list[Tuple[str, str]] = None,
    ) -> dict[str, Any]:
        """Synchronous evaluate method that calls async version."""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.evaluate_async(text_pairs, labels, return_details, original_pairs))
                    )
                    return future.result()
            else:
                # No running loop, can use asyncio.run
                return asyncio.run(self.evaluate_async(text_pairs, labels, return_details, original_pairs))
        except RuntimeError:
            # No event loop, use asyncio.run
            return asyncio.run(self.evaluate_async(text_pairs, labels, return_details, original_pairs))


# Factory functions for each model
def create_haiku_baseline() -> LLMBaseline:
    """Create Claude Haiku 3.5 baseline."""
    return LLMBaseline("claude-3-5-haiku-20241022")


def create_sonnet_baseline() -> LLMBaseline:
    """Create Claude Sonnet 4 baseline."""
    return LLMBaseline("claude-sonnet-4-20250514")


def create_opus_baseline() -> LLMBaseline:
    """Create Claude Opus 4.1 baseline."""
    return LLMBaseline("claude-opus-4-1-20250805")
