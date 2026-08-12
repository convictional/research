"""Evaluation script for baseline authorship verification models."""

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Optional
import random

from baselines import LUARBaseline, ModernBERTBaseline, create_haiku_baseline, create_sonnet_baseline, create_opus_baseline, CustomModelBaseline
from baselines.text_sanitizer import TextSanitizer
from data.extract_training_data import AuthorshipDataExtractor

logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Evaluator for baseline authorship verification models."""

    def __init__(self, results_dir: str = "baselines/results", selected_models: list[str] = None,
                 sanitize_text: bool = True, custom_checkpoint_path: str = None):
        """Initialize the evaluator.

        Args:
            results_dir: Directory to save evaluation results
            selected_models: list of model names to load (default: load all)
            sanitize_text: Whether to sanitize text to remove metadata (default: True)
            custom_checkpoint_path: Path to custom model checkpoint (required if "custom" in selected_models)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.sanitize_text = sanitize_text
        self.custom_checkpoint_path = custom_checkpoint_path

        # Initialize baseline models
        self.models = {}
        self.selected_models = selected_models or ["luar", "modernbert", "llm-haiku", "llm-sonnet", "llm-opus"]
        self._load_baseline_models()

    def _load_baseline_models(self):
        """Load selected baseline models."""
        logger.info(f"Loading baseline models: {self.selected_models}")

        if "luar" in self.selected_models:
            try:
                logger.info("Loading LUAR model...")
                self.models["luar"] = LUARBaseline()
                logger.info("LUAR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LUAR model: {e}")
                self.models["luar"] = None

        if "modernbert" in self.selected_models:
            try:
                logger.info("Loading ModernBERT model...")
                self.models["modernbert"] = ModernBERTBaseline()
                logger.info("ModernBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ModernBERT model: {e}")
                self.models["modernbert"] = None

        if "llm-haiku" in self.selected_models:
            try:
                logger.info("Loading Claude Haiku 3.5 model...")
                self.models["llm-haiku"] = create_haiku_baseline()
                logger.info("Claude Haiku 3.5 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Claude Haiku 3.5 model: {e}")
                self.models["llm-haiku"] = None

        if "llm-sonnet" in self.selected_models:
            try:
                logger.info("Loading Claude Sonnet 4 model...")
                self.models["llm-sonnet"] = create_sonnet_baseline()
                logger.info("Claude Sonnet 4 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Claude Sonnet 4 model: {e}")
                self.models["llm-sonnet"] = None

        if "llm-opus" in self.selected_models:
            try:
                logger.info("Loading Claude Opus 4.1 model...")
                self.models["llm-opus"] = create_opus_baseline()
                logger.info("Claude Opus 4.1 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Claude Opus 4.1 model: {e}")
                self.models["llm-opus"] = None

        if "custom" in self.selected_models:
            try:
                if not self.custom_checkpoint_path:
                    raise ValueError("custom_checkpoint_path required when using custom model")
                logger.info(f"Loading custom Siamese model from {self.custom_checkpoint_path}...")
                self.models["custom"] = CustomModelBaseline(self.custom_checkpoint_path)
                logger.info("Custom Siamese model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load custom Siamese model: {e}")
                self.models["custom"] = None

    async def prepare_test_data(
        self,
        max_pairs_per_author: int | None = None,
        validation_split: float = 0.2,
        return_calibration_data: bool = False
    ) -> Tuple[list[Tuple[str, str]], list[int], Optional[list[Tuple[str, str]]], Optional[list[int]]]:
        """Prepare test data from internal database.

        Args:
            max_pairs_per_author: Maximum pairs to generate per author
            validation_split: Fraction of data to use for testing
            return_calibration_data: If True, also return calibration data from train split

        Returns:
            Tuple of (test_pairs, test_labels, calibration_pairs, calibration_labels)
        """
        logger.info("Loading internal training data...")

        # Load data using existing extractor
        extractor = AuthorshipDataExtractor()
        samples_by_author = await extractor.extract_training_samples()
        filtered_samples = extractor.filter_samples_for_training(samples_by_author)

        # Convert ContentSample objects to text strings
        texts_by_author = {}
        for author, samples in filtered_samples.items():
            texts_by_author[author] = [sample.content for sample in samples]

        logger.info(f"Loaded data for {len(texts_by_author)} authors")

        # Split into train/validation (use validation set for testing)
        test_texts_by_author = {}
        for author, texts in texts_by_author.items():
            n_val = max(1, int(len(texts) * validation_split))
            # Use the validation split for testing
            test_texts_by_author[author] = texts[-n_val:]

        logger.info(f"Using validation split for testing: {sum(len(texts) for texts in test_texts_by_author.values())} texts")

        # Generate test pairs
        text_pairs = []
        labels = []
        authors = list(test_texts_by_author.keys())

        # Generate positive pairs (same author)
        for author in authors:
            texts = test_texts_by_author[author]
            if len(texts) < 2:
                continue

            # Generate pairs within author
            pairs_generated = 0
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if max_pairs_per_author is not None and pairs_generated >= max_pairs_per_author:
                        break
                    text_pairs.append((texts[i], texts[j]))
                    labels.append(1)  # Same author
                    pairs_generated += 1
                if max_pairs_per_author is not None and pairs_generated >= max_pairs_per_author:
                    break

        n_positive = len(text_pairs)
        logger.info(f"Generated {n_positive} positive pairs")

        # Generate negative pairs (different authors)
        n_negative_needed = n_positive  # Balance the dataset
        negative_generated = 0

        while negative_generated < n_negative_needed and len(authors) > 1:
            # Pick two random authors
            author1, author2 = random.sample(authors, 2)

            # Pick random texts from each
            text1 = random.choice(test_texts_by_author[author1])
            text2 = random.choice(test_texts_by_author[author2])

            text_pairs.append((text1, text2))
            labels.append(0)  # Different authors
            negative_generated += 1

        logger.info(f"Generated {negative_generated} negative pairs")
        logger.info(f"Total test pairs: {len(text_pairs)}")

        # Generate calibration data from train split if requested
        calibration_pairs = None
        calibration_labels = None

        if return_calibration_data:
            logger.info("Generating calibration data from training split...")
            # Use train texts for calibration (completely separate from test)
            train_texts_by_author = {}
            for author_id, texts in texts_by_author.items():
                split_idx = int(len(texts) * (1 - validation_split))
                train_texts_by_author[author_id] = texts[:split_idx]

            # Generate calibration pairs (smaller set for efficiency)
            calibration_pairs = []
            calibration_labels = []
            authors_list = list(train_texts_by_author.keys())

            # Generate up to 100 calibration pairs (50 positive, 50 negative)
            # Positive pairs from train data
            positive_count = 0
            for author_id, texts in train_texts_by_author.items():
                if len(texts) >= 2 and positive_count < 50:
                    for i in range(min(2, len(texts) - 1)):
                        calibration_pairs.append((texts[i], texts[i + 1]))
                        calibration_labels.append(1)
                        positive_count += 1
                        if positive_count >= 50:
                            break

            # Negative pairs from train data
            negative_count = 0
            while negative_count < 50 and len(authors_list) >= 2:
                author1, author2 = random.sample(authors_list, 2)
                if train_texts_by_author[author1] and train_texts_by_author[author2]:
                    text1 = random.choice(train_texts_by_author[author1])
                    text2 = random.choice(train_texts_by_author[author2])
                    calibration_pairs.append((text1, text2))
                    calibration_labels.append(0)
                    negative_count += 1

            logger.info(f"Generated {len(calibration_pairs)} calibration pairs ({sum(calibration_labels)} positive, {len(calibration_labels) - sum(calibration_labels)} negative)")

        return text_pairs, labels, calibration_pairs, calibration_labels

    def evaluate_model(
        self,
        model_name: str,
        text_pairs: list[Tuple[str, str]],
        labels: list[int],
        original_pairs: list[Tuple[str, str]] = None,
        calibration_pairs: list[Tuple[str, str]] = None,
        calibration_labels: list[int] = None
    ) -> dict[str, Any]:
        """Evaluate a single baseline model.

        Args:
            model_name: Name of the model to evaluate
            text_pairs: list of (text1, text2) pairs
            labels: Ground truth labels

        Returns:
            dictionary containing evaluation metrics
        """
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not loaded")
            return {"error": f"Model {model_name} not available"}

        logger.info(f"Evaluating {model_name} on {len(text_pairs)} pairs...")

        try:
            # Check if model supports detailed predictions (LLM models)
            if model_name.startswith("llm-"):
                results = model.evaluate(text_pairs, labels, return_details=True, original_pairs=original_pairs)
            else:
                # Check if model needs calibration data (models without saved thresholds)
                if calibration_pairs is not None and not (hasattr(model, 'has_saved_threshold') and model.has_saved_threshold()):
                    results = model.evaluate(text_pairs, labels, calibration_pairs, calibration_labels)
                else:
                    results = model.evaluate(text_pairs, labels)

            logger.info(f"{model_name} evaluation completed")
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  AUC: {results['auc']:.4f}")
            logger.info(f"  F1: {results['f1']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {"error": str(e)}

    async def run_evaluation(self, max_pairs_per_author: int | None = None) -> dict[str, Any]:
        """Run complete baseline evaluation.

        Args:
            max_pairs_per_author: Maximum pairs to generate per author

        Returns:
            dictionary containing all evaluation results
        """
        logger.info("Starting baseline evaluation...")

        # Prepare test data and calibration data
        text_pairs, labels, calibration_pairs, calibration_labels = await self.prepare_test_data(
            max_pairs_per_author=max_pairs_per_author,
            return_calibration_data=True
        )

        # Keep original pairs for logging purposes
        original_pairs = text_pairs.copy() if self.sanitize_text else None

        # Sanitize text pairs if requested
        if self.sanitize_text:
            logger.info("Sanitizing text to remove identifying metadata...")
            text_pairs = TextSanitizer.sanitize_pairs(text_pairs)

        # Evaluate all models
        results = {
            "evaluation_date": datetime.now().isoformat(),
            "n_test_pairs": len(text_pairs),
            "n_positive_pairs": sum(labels),
            "n_negative_pairs": len(labels) - sum(labels),
            "models": {}
        }

        for model_name in self.models.keys():
            if self.models[model_name] is not None:
                model_results = self.evaluate_model(
                    model_name, text_pairs, labels, original_pairs,
                    calibration_pairs, calibration_labels
                )
                results["models"][model_name] = model_results

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()  # Convert numpy scalar to Python type
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj

        results = convert_numpy_types(results)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"baseline_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Save detailed predictions as CSV for models that support it
        for model_name, model_results in results["models"].items():
            if "detailed_predictions" in model_results:
                csv_file = self.results_dir / f"predictions_{model_name}_{timestamp}.csv"
                self._save_predictions_csv(csv_file, model_results["detailed_predictions"])
                logger.info(f"Detailed predictions saved to {csv_file}")

        return results

    def _save_predictions_csv(self, csv_file: Path, predictions: list[dict[str, Any]]):
        """Save detailed predictions to CSV file."""
        if not predictions:
            return

        # Check what fields are available in the predictions
        if predictions:
            sample_pred = predictions[0]
            if "text1_raw" in sample_pred:
                # Include both raw and sanitized text
                fieldnames = ["pair_idx", "text1_raw", "text2_raw", "text1_sanitized", "text2_sanitized",
                             "true_label", "predicted_label", "confidence", "reasoning"]
            else:
                # Only sanitized text available
                fieldnames = ["pair_idx", "text1_sanitized", "text2_sanitized",
                             "true_label", "predicted_label", "confidence", "reasoning"]

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.dictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(predictions)

    def print_comparison(self, results: dict[str, Any]):
        """Print a comparison table of results.

        Args:
            results: Results dictionary from run_evaluation
        """
        print("\n" + "="*80)
        print("BASELINE MODEL COMPARISON")
        print("="*80)

        print(f"Evaluation Date: {results['evaluation_date']}")
        print(f"Test Pairs: {results['n_test_pairs']} ({results['n_positive_pairs']} positive, {results['n_negative_pairs']} negative)")
        print()

        # Print metrics table
        metrics = ["accuracy", "auc", "precision", "recall", "f1"]

        print(f"{'Model':<15} " + " ".join([f"{metric:<10}" for metric in metrics]))
        print("-" * 80)

        for model_name, model_results in results["models"].items():
            if "error" in model_results:
                print(f"{model_name:<15} ERROR: {model_results['error']}")
            else:
                row = f"{model_name:<15} "
                for metric in metrics:
                    value = model_results.get(metric, 0.0)
                    row += f"{value:<10.4f} "
                print(row)

        print("="*80)
        print()

        # Print confusion matrices
        print("CONFUSION MATRICES")
        print("="*80)
        for model_name, model_results in results["models"].items():
            if "error" not in model_results and "confusion_matrix" in model_results:
                cm = model_results["confusion_matrix"]
                print(f"\n{model_name} Confusion Matrix:")
                print("                 Predicted")
                print("              Different  Same")
                print(f"Actual Different  {cm['true_negative']:4d}    {cm['false_positive']:4d}")
                print(f"       Same       {cm['false_negative']:4d}    {cm['true_positive']:4d}")

                # Calculate additional metrics from confusion matrix
                total = cm['true_negative'] + cm['false_positive'] + cm['false_negative'] + cm['true_positive']
                specificity = cm['true_negative'] / (cm['true_negative'] + cm['false_positive']) if (cm['true_negative'] + cm['false_positive']) > 0 else 0
                sensitivity = cm['true_positive'] / (cm['true_positive'] + cm['false_negative']) if (cm['true_positive'] + cm['false_negative']) > 0 else 0

                print(f"  Sensitivity (Recall): {sensitivity:.4f}")
                print(f"  Specificity:          {specificity:.4f}")
                print(f"  Total samples:        {total}")

        print("="*80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Baseline evaluation for authorship verification")

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["luar", "modernbert", "llm-haiku", "llm-sonnet", "llm-opus", "custom", "all"],
        default=["all"],
        help="Models to evaluate (default: all)"
    )

    parser.add_argument(
        "--max-pairs-per-author",
        type=int,
        default=None,
        help="Maximum pairs to generate per author (default: use all pairs)"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent LLM requests (default: 10)"
    )

    parser.add_argument(
        "--rate-limit",
        type=int,
        default=100,
        help="Rate limit per minute for LLM requests (default: 100)"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="baselines/results",
        help="Directory to save results (default: baselines/results)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable text sanitization (keep metadata)"
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to custom model checkpoint (required if 'custom' in models)"
    )

    return parser.parse_args()


async def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Determine which models to load
    if "all" in args.models:
        selected_models = ["luar", "modernbert", "llm-haiku", "llm-sonnet", "llm-opus"]
    else:
        selected_models = args.models

    logger.info(f"Selected models: {selected_models}")

    # Update LLM settings if provided
    from config.llm_settings import llm_settings
    if args.max_concurrent:
        llm_settings.max_concurrent_requests = args.max_concurrent
    if args.rate_limit:
        llm_settings.rate_limit_per_minute = args.rate_limit

    # Run evaluation
    evaluator = BaselineEvaluator(
        results_dir=args.results_dir,
        selected_models=selected_models,
        sanitize_text=not args.no_sanitize,
        custom_checkpoint_path=args.checkpoint_path
    )
    results = await evaluator.run_evaluation(max_pairs_per_author=args.max_pairs_per_author)

    # Print comparison
    evaluator.print_comparison(results)


if __name__ == "__main__":
    asyncio.run(main())
