import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import shutil
from sklearn.metrics import r2_score
from typing import Dict, List, Optional
from tqdm import tqdm

from ..helpers.async_helper import execute_tasks_with_manual_pbar
from ..helpers.io import load_checkpoint
from ..settings import settings, logger

# Initialize process pool for CPU-bound operations
N_PHYSICAL_CORES = multiprocessing.cpu_count() // 2  # Account for efficiency/performance cores
process_pool = ProcessPoolExecutor(max_workers=N_PHYSICAL_CORES)


@dataclass
class ModelParams:
    """Parameters for the entropy-driven generative model."""

    alpha_E: float  # Initial importance factor
    delta: float  # Decay constant for baseline interest
    alpha_global: float  # Weight for global entropy term
    alpha_local: float  # Weight for local entropy term
    mu: float  # Mean document length (facts)
    sigma: float  # Standard deviation of document length
    T: int  # Total time steps to simulate
    daily_doc_mu: float = 0.5
    daily_doc_sigma: float = 0.25
    alpha_docs: float = 10.0  # New parameter for doc-level variation
    use_actual_docs: bool = False  # New parameter to control doc count source
    actual_daily_docs: Dict[int, int] = None  # New parameter to store actual doc counts
    constant: float = 0.0  # New parameter for y-intercept


class EntropyDrivenModel:
    """Implements the entropy-driven model for fact generation."""

    def __init__(self, params: ModelParams):
        self.params = params
        self.documents: Dict[int, Dict[str, int]] = {}  # doc_id -> {fact_id: count}
        self.time_created: Dict[int, int] = {}  # doc_id -> creation time
        self.fact_counts: Dict[str, int] = {}  # fact_id -> total count
        self.doc_probabilities: Dict[int, float] = {}  # Store doc-level probabilities
        self.actual_daily_docs = params.actual_daily_docs if params.actual_daily_docs else {}

    def _compute_entropy(self, docs: Optional[List[int]] = None) -> float:
        """Compute entropy over specified documents or all documents."""
        if not self.documents:
            return 0.0

        # Use specified docs or all docs
        doc_set = set(docs) if docs else set(self.documents.keys())

        # Calculate probabilities
        total_facts = sum(sum(doc.values()) for doc_id, doc in self.documents.items() if doc_id in doc_set)
        if total_facts == 0:
            return 0.0

        probs = []
        for doc_id in doc_set:
            if doc_id in self.documents:
                doc_facts = sum(self.documents[doc_id].values())
                if doc_facts > 0:
                    probs.append(doc_facts / total_facts)

        return -sum(p * np.log2(p) for p in probs if p > 0) + self.params.constant

    def _sigmoid(self, x: float) -> float:
        """Compute sigmoid function."""
        # The logistic function is too aggressive, use an algebraic sigmoid
        return x / (1 + np.abs(x))

    def _compute_timestep_mean(self, time: int) -> tuple[float, float, float]:
        """Compute the time-step mean proportion (formerly _compute_proportion_probability)."""
        # Get current corpus size
        C = len(self.documents)
        if C == 0:
            baseline = self.params.alpha_E
            entropy_term = 0
            return self._sigmoid(baseline + entropy_term), baseline, entropy_term

        # Get documents from current time step
        current_docs = [d for d, t in self.time_created.items() if t == time - 1]

        # Compute global and local entropy
        H_global = self._compute_entropy()
        H_local = self._compute_entropy(current_docs) if current_docs else 0

        # Compute normalized entropies
        # H_global_norm = H_global / np.log2(max(C, 2))
        # H_local_norm = H_local / np.log2(max(len(current_docs), 2)) if current_docs else 0

        # Compute components
        baseline = self.params.alpha_E * np.exp(-self.params.delta * time)
        # add another param for local entropy
        entropy_term = (1 + self.params.alpha_local * H_local) / (1 + self.params.alpha_global * H_global)

        result = self._sigmoid(baseline + entropy_term)
        return result, baseline, entropy_term

    def _sample_doc_proportion(self, time_step_mean: float) -> float:
        """Sample document-level proportion from Beta distribution."""
        # Clamp time_step_mean to avoid numerical issues
        time_step_mean = min(max(time_step_mean, 0.001), 0.999)

        alpha_docs = self.params.alpha_docs + 1e-6  # avoid zero
        doc_p = np.random.beta(alpha_docs * time_step_mean, alpha_docs * (1.0 - time_step_mean))
        return doc_p

    def simulate(self, start_time: int = 0) -> Dict:
        """Run simulation and return results starting from specified time."""
        entropy_history = []
        baseline_component = []
        entropy_component = []
        fact_counts = []
        doc_lengths = []

        for time in tqdm(range(start_time, start_time + self.params.T)):
            # Compute time-step mean once for this time step
            time_step_mean, baseline, entropy_term = self._compute_timestep_mean(time)

            # Always use actual document counts when available
            if time in self.actual_daily_docs:
                daily_docs = self.actual_daily_docs[time]
            else:
                daily_docs = 1  # Fallback to minimum if no actual count

            for _ in range(daily_docs):
                doc_length = int(np.round(np.random.lognormal(self.params.mu, self.params.sigma)))
                doc_p = self._sample_doc_proportion(time_step_mean)

                doc_id = len(self.documents)
                self.documents[doc_id] = {}
                self.time_created[doc_id] = time
                self.doc_probabilities[doc_id] = doc_p

                n_facts = int(np.round(doc_length * doc_p))
                for _ in range(n_facts):
                    fact_id = f"fact_{len(self.fact_counts)}"
                    self.documents[doc_id][fact_id] = 1
                    self.fact_counts[fact_id] = self.fact_counts.get(fact_id, 0) + 1

            # Record metrics
            entropy_history.append(self._compute_entropy())
            baseline_component.append(baseline)
            entropy_component.append(entropy_term)
            fact_counts.append(len(self.fact_counts))
            doc_lengths.append(doc_length)

        return {
            "entropy_history": entropy_history,
            "baseline_component": baseline_component,
            "entropy_component": entropy_component,
            "fact_counts": fact_counts,
            "doc_lengths": doc_lengths,
            "start_time": start_time,
            "doc_probabilities": self.doc_probabilities,
        }


def _sanitize_path(name: str) -> str:
    """Convert a string into a safe path component by replacing unsafe chars with underscore."""
    return "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in name)


def plot_entropy_comparison(empirical_data, model_results, params, entity_name, output_dir, growth_patterns=None):
    """Plot comparison with burst analysis overlay."""
    plt.figure(figsize=(12, 8))

    # Create arrays for entropy components
    time_points = np.arange(len(model_results["entropy_history"]))
    baseline = np.array(model_results["baseline_component"])
    entropy_term = np.array(model_results["entropy_component"])
    total_entropy = np.array(model_results["entropy_history"])

    # Normalize components to sum to total entropy at each time point
    total_components = baseline + entropy_term
    normalized_baseline = np.where(total_components > 0, baseline / total_components * total_entropy, 0)
    normalized_entropy_term = np.where(total_components > 0, entropy_term / total_components * total_entropy, 0)

    # Plot stacked area for entropy components
    plt.fill_between(time_points, 0, normalized_baseline, alpha=0.3, color="blue", label="Baseline-Driven")
    plt.plot(time_points, normalized_baseline, color="blue", linewidth=1, alpha=0.8)  # Baseline boundary

    plt.fill_between(
        time_points,
        normalized_baseline,
        normalized_baseline + normalized_entropy_term,
        alpha=0.3,
        color="green",
        label="Entropy-Driven",
    )
    plt.plot(
        time_points, normalized_baseline + normalized_entropy_term, color="green", linewidth=1, alpha=0.8
    )  # Total boundary

    # Plot empirical entropy curve
    plt.plot(np.arange(len(empirical_data)), empirical_data, "k-", label="Empirical", linewidth=2, zorder=5)

    # Highlight single-document period if available
    if growth_patterns and growth_patterns["single_doc_period"] > 0:
        plt.axvspan(0, growth_patterns["single_doc_period"], color="gray", alpha=0.2, label="Single-doc period")

    # Add parameter details to plot
    param_text = (
        f"Model Parameters:\n"
        f"α_E = {params.alpha_E:.3f}\n"
        f"δ = {params.delta:.3f}\n"
        f"α_global = {params.alpha_global:.3f}\n"
        f"α_local = {params.alpha_local:.3f}\n"
        f"μ = {params.mu:.3f}\n"
        f"σ = {params.sigma:.3f}\n"
        f"c = {params.constant:.3f}"  # Added constant to parameter display
    )
    plt.text(
        0.02,
        0.98,
        param_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Entropy (bits)", fontsize=12)
    plt.title(f"Entropy Evolution: {entity_name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.savefig(
        output_dir / f"entropy_comparison_{_sanitize_path(entity_name.lower())}_.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_burstiness_analysis(empirical_data, growth_patterns, entity_name, output_dir):
    """Create detailed visualization of growth patterns."""
    plt.figure(figsize=(15, 10))

    # Main entropy curve
    plt.subplot(211)
    plt.plot(empirical_data, "k-", label="Entropy")

    # Highlight bursts
    if growth_patterns["bursts"]:
        burst_days = [day for day, _ in growth_patterns["bursts"]]
        plt.scatter(
            burst_days, [empirical_data[d] for d in burst_days], color="red", s=100, marker="*", label="Bursts"
        )

    plt.axvspan(0, growth_patterns["single_doc_period"], color="gray", alpha=0.2, label="Single-doc period")
    plt.legend()
    plt.title(f"Growth Pattern Analysis: {entity_name}")

    # Growth rate analysis
    plt.subplot(212)
    differences = np.diff(empirical_data)
    plt.plot(differences, "b-", label="Daily Change", alpha=0.5)
    plt.axhline(y=0, color="k", linestyle=":")

    # Highlight significant jumps
    jump_threshold = 0.1  # Same as burst threshold
    jumps = differences > jump_threshold
    plt.scatter(np.where(jumps)[0], differences[jumps], color="red", s=50, label="Significant Jumps")

    plt.legend()
    plt.title("Daily Entropy Changes")
    plt.tight_layout()

    plt.savefig(
        output_dir / f"burstiness_analysis_{_sanitize_path(entity_name.lower())}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def fit_model_parameters(empirical_entropy, initial_params=None, n_restarts=40) -> ModelParams:
    """Fit model parameters to match empirical entropy curve with multiple random restarts for robustness."""
    if initial_params is None:
        initial_params = ModelParams(
            alpha_E=1.0,
            delta=0.01,
            alpha_global=1.0,
            alpha_local=1.0,
            alpha_docs=1.0,
            mu=2.0,
            sigma=0.5,
            T=len(empirical_entropy),
        )

    def objective(params_array):
        """Objective function for parameter fitting."""
        # base_alpha = params_array[0]  # Single parameter controlling both alphas
        test_params = ModelParams(
            alpha_E=params_array[0],  # Direct use
            delta=params_array[1],
            alpha_global=params_array[2],
            alpha_local=params_array[3],
            alpha_docs=params_array[4],
            constant=params_array[5],  # New constant parameter
            mu=initial_params.mu,  # Keep document length params fixed
            sigma=initial_params.sigma,
            T=len(empirical_entropy),
        )

        model = EntropyDrivenModel(test_params)
        results = model.simulate(start_time=0)  # Always start from 0
        simulated_entropy = results["entropy_history"]

        # Compute mean squared error
        min_len = min(len(empirical_entropy), len(simulated_entropy))
        return np.mean((np.array(empirical_entropy[:min_len]) - np.array(simulated_entropy[:min_len])) ** 2)

    best_params = None
    best_value = float("inf")

    for restart in range(n_restarts):
        # Could randomize initial guesses here:
        x0 = np.array(
            [
                np.random.uniform(0.001, 100),  # alpha_E
                np.random.uniform(0.0, 1.0),  # delta
                np.random.uniform(0.001, 100),  # alpha_global
                np.random.uniform(0.001, 100),  # alpha_local
                np.random.uniform(0.001, 100),  # alpha_docs
                np.random.uniform(0, 10),  # constant
            ]
        )

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=[(0, 100), (0, 10), (0, 100), (0, 100), [0, 100], (0, 10)],  # Added bounds for constant
            tol=1e-6,
            options={"maxiter": 15000},
        )
        logger.info(
            f"Restart {restart + 1}, success={result.success}, message={result.message}, fval={result.fun:.4f}"
        )

        if result.fun < best_value:
            best_value = result.fun
            best_params = result.x

    if best_params is None:
        logger.warning("No successful optimization run found, using defaults.")
        return initial_params

    logger.info(f"Best MSE: {best_value:.4f} with params {best_params}")

    # base_alpha = best_params[0]
    return ModelParams(
        alpha_E=best_params[0],
        delta=best_params[1],
        alpha_global=best_params[2],
        alpha_local=best_params[3],
        alpha_docs=best_params[4],
        constant=best_params[5],  # New constant parameter
        mu=initial_params.mu,
        sigma=initial_params.sigma,
        T=len(empirical_entropy),
    )


def calculate_empirical_entropy_curve(entity):
    """Calculate empirical entropy curve for an entity, normalized to start at first non-zero entropy."""
    if not entity.facts:
        return [], 0

    # Sort facts by creation time
    sorted_facts = sorted(entity.facts, key=lambda x: x.created_at)
    start_time = sorted_facts[0].created_at
    end_time = sorted_facts[-1].created_at
    total_days = (end_time - start_time).days + 1

    entropy_curve = []
    current_docs = {}
    first_nonzero_idx = None

    for day in range(total_days):
        current_time = start_time + timedelta(days=day)

        # Add new facts for this day
        for fact in sorted_facts:
            if fact.created_at.date() <= current_time.date():
                doc_id = str(fact.source_id)
                if doc_id not in current_docs:
                    current_docs[doc_id] = 0
                current_docs[doc_id] += 1

        # Calculate entropy for current state
        if current_docs:
            total_facts = sum(current_docs.values())
            probs = [count / total_facts for count in current_docs.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            entropy_curve.append(entropy)

            # Track first non-zero entropy
            if first_nonzero_idx is None and entropy > 0:
                first_nonzero_idx = day
        else:
            entropy_curve.append(0)

    # If we found non-zero entropy, shift the curve
    if first_nonzero_idx is not None:
        entropy_curve = entropy_curve[first_nonzero_idx:]
        total_days = len(entropy_curve)
    else:
        # If no non-zero entropy was found, return empty curve
        entropy_curve = []
        total_days = 0

    return entropy_curve, total_days


def estimate_daily_doc_distribution(doc_timestamps: List):
    """
    Estimate lognormal parameters (mu, sigma) from the distribution of daily doc counts
    inferred from doc_timestamps, which are datetime objects.
    """
    # Compute daily counts
    daily_counts = {}
    for ts in doc_timestamps:
        day = ts.date()
        daily_counts[day] = daily_counts.get(day, 0) + 1

    # Convert counts to a list
    counts_arr = np.array(list(daily_counts.values()))
    counts_arr = counts_arr[counts_arr > 0]  # Filter out zeros to avoid log(0)

    # Fit lognormal by taking log and computing mean, std
    log_counts = np.log(counts_arr)
    mu_hat = np.mean(log_counts)
    sigma_hat = np.std(log_counts)
    return mu_hat, sigma_hat


def estimate_factlength_distribution(entities) -> tuple[float, float]:
    """
    Estimate lognormal parameters (mu, sigma) for the distribution of facts-per-document.
    """
    doc_fact_counts = {}
    for entity in entities:
        for fact in entity.facts:
            doc_id = fact.source_id
            doc_fact_counts[doc_id] = doc_fact_counts.get(doc_id, 0) + 1

    if not doc_fact_counts:
        return 1.0, 0.1  # Fall back to a small default

    lengths = np.array(list(doc_fact_counts.values()))
    log_lengths = np.log(lengths[lengths > 0])
    mu_hat = np.mean(log_lengths)
    sigma_hat = np.std(log_lengths)
    return mu_hat, sigma_hat


def evaluate_lognormal_fit(data: np.ndarray, mu: float, sigma: float):
    """
    Perform K-S test to see how well a lognormal(mu, sigma) fits the data.
    Returns (statistic, p_value).
    """
    # shape = sigma, loc=0, scale=exp(mu)
    shape, loc, scale = sigma, 0, np.exp(mu)
    ks_statistic, p_value = stats.kstest(data, "lognorm", args=(shape, loc, scale))
    return ks_statistic, p_value


def plot_distribution_fit(data: np.ndarray, mu: float, sigma: float, name: str, output_dir: Path):
    """Plot empirical distribution against fitted lognormal."""
    plt.figure(figsize=(10, 6))

    # Plot empirical distribution
    plt.hist(data, bins=50, density=True, alpha=0.7, label="Empirical")

    # Plot fitted distribution
    x = np.linspace(min(data), max(data), 100)
    pdf = stats.lognorm.pdf(x, sigma, loc=0, scale=np.exp(mu))
    plt.plot(x, pdf, "r-", label="Fitted Lognormal")

    plt.xlabel(f"{name} Count")
    plt.ylabel("Density")
    plt.title(f"{name} Distribution Fit")
    plt.legend()
    plt.savefig(output_dir / f"{_sanitize_path(name.lower())}_distribution_fit.png")
    plt.close()


def analyze_growth_patterns(entropy_curve: List[float], window: int = 10) -> Dict:
    """Analyze growth patterns in entropy curve."""
    if not entropy_curve:
        return {}

    # Find bursts (significant increases after periods of stability)
    bursts = []
    stable_threshold = 0.01  # What we consider "stable" entropy
    burst_threshold = 0.1  # What we consider a "burst" increase

    for i in range(window, len(entropy_curve)):
        pre_window = entropy_curve[i - window : i]
        if (
            max(pre_window) - min(pre_window) < stable_threshold  # Was stable
            and entropy_curve[i] - entropy_curve[i - 1] > burst_threshold
        ):  # Then jumped
            bursts.append((i, entropy_curve[i] - entropy_curve[i - 1]))

    # Analyze stepwise vs smooth growth
    differences = np.diff(entropy_curve)
    smoothness = np.std(differences)  # Lower = more smooth

    # Find initial single-doc period
    single_doc_period = 0
    for val in entropy_curve:
        if val == 0:
            single_doc_period += 1
        else:
            break

    return {
        "bursts": bursts,
        "smoothness": smoothness,
        "single_doc_period": single_doc_period,
        "total_jumps": len([d for d in differences if d > burst_threshold]),
    }


def calculate_prediction_metrics(true_values, predicted_values):
    """Calculate various metrics for prediction quality."""
    # Convert inputs to numpy arrays and ensure they're the same length
    true_arr = np.array(true_values)
    pred_arr = np.array(predicted_values)

    # Trim arrays to same length if needed
    min_len = min(len(true_arr), len(pred_arr))
    true_arr = true_arr[:min_len]
    pred_arr = pred_arr[:min_len]

    # Basic metrics
    mse = np.mean((true_arr - pred_arr) ** 2)
    mae = np.mean(np.abs(true_arr - pred_arr))

    # R² calculation using sklearn (handles edge cases better)
    # Will return negative values when model performs worse than horizontal line
    r2 = r2_score(true_arr, pred_arr)

    # Clamp extremely negative R² values to -1 for better interpretability
    r2 = max(-1.0, r2)

    return {"mse": mse, "mae": mae, "r2": r2}


def partial_fit_and_evaluate(
    empirical_curve,
    full_length,
    intervals,
    daily_doc_mu,
    daily_doc_sigma,
    doc_mu,
    doc_sigma,
):
    """Fit model on partial data and evaluate predictions."""
    results = {}
    EVAL_WINDOW = 90  # Fixed 90-day evaluation window

    # Convert empirical curve to numpy array if it isn't already
    empirical_curve = np.array(empirical_curve)

    for interval in intervals:
        # Check if we have enough data for training + 90 day evaluation
        if interval + EVAL_WINDOW >= full_length:
            logger.debug(f"Skipping interval {interval} - insufficient evaluation data")
            continue

        # Split data into training and evaluation sets
        partial_data = empirical_curve[:interval]
        holdout_data = empirical_curve[interval : interval + EVAL_WINDOW]  # Only take 90 days for evaluation

        # Initialize and fit on partial data
        partial_init = ModelParams(
            alpha_E=1.0,
            delta=0.01,
            alpha_global=0.5,
            alpha_local=1.0,
            alpha_docs=1.0,
            mu=doc_mu,
            sigma=doc_sigma,
            T=interval,
            daily_doc_mu=daily_doc_mu,
            daily_doc_sigma=daily_doc_sigma,
        )

        try:
            # Fit parameters on partial data
            partial_params = fit_model_parameters(partial_data, partial_init)

            # Simulate trajectory for training + evaluation period
            partial_params.T = interval + EVAL_WINDOW
            model = EntropyDrivenModel(partial_params)
            sim_results = model.simulate()
            sim_entropy = np.array(sim_results["entropy_history"])  # Convert to numpy array

            # Evaluate predictions on 90-day holdout period
            if len(sim_entropy) >= interval + EVAL_WINDOW:
                predicted_holdout = sim_entropy[interval : interval + EVAL_WINDOW]
                metrics = calculate_prediction_metrics(holdout_data, predicted_holdout)

                logger.debug(f"Successful fit for interval {interval}")
                logger.debug(f"Metrics: {metrics}")

                results[interval] = {
                    "metrics": metrics,
                    "params": partial_params,
                    "predicted": predicted_holdout.tolist(),  # Convert numpy array to list
                    "true": holdout_data.tolist(),  # Convert numpy array to list
                }
            else:
                logger.warning(f"Simulation too short for interval {interval}")
                results[interval] = {"error": "Simulation too short"}
        except Exception as e:
            logger.error(f"Error in interval {interval}: {str(e)}")
            results[interval] = {"error": str(e)}

    return results


def plot_fit_summary(fit_summaries, output_dir):
    """Create summary visualizations of model fits."""
    if not fit_summaries:
        logger.warning("No fit summaries provided for plotting")
        return

    logger.info(f"Number of fit summaries received: {len(fit_summaries)}")

    # Get intervals that have metrics data
    all_metrics = {}

    # First pass: collect valid intervals and their metrics
    for summary in fit_summaries:
        if not summary:
            continue

        logger.info(f"Processing summary for entity: {summary.get('entity', 'unknown')}")

        if "partial_fits" not in summary:
            logger.warning(f"No partial_fits in summary for {summary.get('entity', 'unknown')}")
            continue

        for interval, results in summary["partial_fits"].items():
            if "metrics" not in results:
                logger.warning(f"No metrics for interval {interval} in {summary.get('entity', 'unknown')}")
                continue

            logger.info(f"Found metrics for interval {interval}: {results['metrics']}")

            if interval not in all_metrics:
                all_metrics[interval] = {"mse": [], "r2": []}

            all_metrics[interval]["mse"].append(results["metrics"]["mse"])
            all_metrics[interval]["r2"].append(results["metrics"]["r2"])

    # Sort intervals and ensure we have data
    intervals = sorted(all_metrics.keys())

    logger.info(f"Found valid intervals: {intervals}")
    logger.info(f"Metrics collected: {all_metrics}")

    if not intervals:
        logger.warning("No valid intervals found for plotting")
        return

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Prepare data for plotting
    mse_data = [all_metrics[i]["mse"] for i in intervals]

    # Check if we have valid data
    interval_counts = {i: len(all_metrics[i]["mse"]) for i in intervals}

    if any(len(d) > 0 for d in mse_data):
        ax1.boxplot(mse_data, labels=intervals)
        ax1.set_xlabel("Training Period (days)")
        ax1.set_ylabel("MSE")
        title = "Distribution of MSE by Training Period\n"
        title += f"90-day Prediction Window (n={', '.join(f'{i}d: {interval_counts[i]}' for i in intervals)})"
        ax1.set_title(title)
    else:
        ax1.text(0.5, 0.5, "No MSE data available", ha="center", va="center")

    # Plot RMSE in second subplot
    if any(len(d) > 0 for d in mse_data):
        # Convert MSE to RMSE for better interpretability
        rmse_data = [np.sqrt(np.array(d)) for d in mse_data]
        ax2.boxplot(rmse_data, labels=intervals)
        ax2.set_xlabel("Training Period (days)")
        ax2.set_ylabel("RMSE")
        title = "Distribution of RMSE by Training Period\n"
        title += f"90-day Prediction Window (n={', '.join(f'{i}d: {interval_counts[i]}' for i in intervals)})"
        ax2.set_title(title)
    else:
        ax2.text(0.5, 0.5, "No RMSE data available", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "fit_summary.png")
    plt.close()

    # Log summary statistics
    logger.info("\nFit Summary Statistics:")
    for interval in intervals:
        if all_metrics[interval]["mse"]:
            mse_values = all_metrics[interval]["mse"]
            r2_values = all_metrics[interval]["r2"]
            logger.info(f"\nTraining Period: {interval} days")
            logger.info(
                f"  MSE - Mean: {np.mean(mse_values):.4f}, Median: {np.median(mse_values):.4f}, "
                f"Std: {np.std(mse_values):.4f}, Count: {len(mse_values)}"
            )
            if r2_values:
                logger.info(
                    f"  R² - Mean: {np.mean(r2_values):.4f}, Median: {np.median(r2_values):.4f}, "
                    f"Std: {np.std(r2_values):.4f}, Count: {len(r2_values)}"
                )


def plot_fit_metrics_scatter(fit_summaries, output_dir):
    """Create scatter plot of entity entropy vs RMSE for 90-day predictions."""
    plt.figure(figsize=(12, 8))

    # Define intervals we're looking for
    training_periods = [30, 60, 90]
    colors = plt.cm.viridis(np.linspace(0, 1, len(training_periods)))

    # Initialize counts
    period_counts = {period: 0 for period in training_periods}

    # First pass to validate data and count valid entries
    for summary in fit_summaries:
        if not summary or "partial_fits" not in summary:
            continue

        for period in training_periods:
            if (
                period in summary["partial_fits"]
                and "metrics" in summary["partial_fits"][period]
                and "mse" in summary["partial_fits"][period]["metrics"]
            ):
                period_counts[period] += 1

    # Plot data for each training period
    for period, color in zip(training_periods, colors):
        x_vals = []  # Max entropy values
        y_vals = []  # RMSE values

        for summary in fit_summaries:
            if not summary:
                continue

            # Skip if we don't have the required data
            if "max_entropy" not in summary:
                continue

            # Check if we have valid partial fit results for this period
            partial_fit = summary.get("partial_fits", {}).get(period, {})
            if not partial_fit or "metrics" not in partial_fit:
                continue

            metrics = partial_fit["metrics"]
            if "mse" not in metrics:
                continue

            max_entropy = summary["max_entropy"]
            rmse = np.sqrt(metrics["mse"])

            x_vals.append(max_entropy)
            y_vals.append(rmse)

        if x_vals:  # Only plot if we have data
            plt.scatter(x_vals, y_vals, c=[color], label=f"{period} days", alpha=0.6)

            # Add trend line
            if len(x_vals) > 1:  # Need at least 2 points for a trend line
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(x_vals), max(x_vals), 100)
                plt.plot(x_range, p(x_range), "--", color=color, alpha=0.5)

    plt.xlabel("Maximum Entity Entropy (bits)")
    plt.ylabel("RMSE (90-day predictions)")  # Updated label
    plt.title(
        "Model Fit Quality vs Entity Complexity\n"
        f"90-day Prediction Window\n"
        f"(n={', '.join(f'{p}d: {period_counts[p]}' for p in training_periods)})"
    )
    plt.legend(title="Training Period")
    plt.grid(True, alpha=0.3)

    plt.savefig(output_dir / "fit_metrics_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


async def run_cpu_bound(func, *args):
    """Run CPU-bound function in process pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(process_pool, func, *args)


async def analyze_single_entity(entity, empirical_curve, doc_mu, doc_sigma, daily_doc_mu, daily_doc_sigma, output_dir):
    """Analyze a single entity and return its results."""
    if not empirical_curve:
        return None

    # Calculate actual daily document counts
    actual_daily_docs = {}
    doc_creation_times = {}
    for fact in entity.facts:
        if fact.source_id not in doc_creation_times:
            doc_creation_times[fact.source_id] = fact.created_at

    start_date = min(doc_creation_times.values()).date()
    for doc_id, created_at in doc_creation_times.items():
        day_index = (created_at.date() - start_date).days
        actual_daily_docs[day_index] = actual_daily_docs.get(day_index, 0) + 1

    # Fit parameters using actual document counts
    fitted_params = await run_cpu_bound(
        fit_model_parameters,
        empirical_curve,
        ModelParams(
            alpha_E=1.0,
            delta=0.1,
            alpha_global=1.0,
            alpha_local=1.0,
            alpha_docs=1.0,
            mu=doc_mu,
            sigma=doc_sigma,
            T=len(empirical_curve),
            daily_doc_mu=daily_doc_mu,
            daily_doc_sigma=daily_doc_sigma,
            actual_daily_docs=actual_daily_docs,
        ),
    )

    # Run simulation with actual document counts
    model = EntropyDrivenModel(fitted_params)
    results = await run_cpu_bound(model.simulate)

    # Run growth pattern analysis
    growth_patterns = await run_cpu_bound(analyze_growth_patterns, empirical_curve)

    # Plot results using original plot_entropy_comparison
    await run_cpu_bound(
        plot_entropy_comparison,
        empirical_curve,
        results,
        fitted_params,
        entity.name,
        output_dir,
        growth_patterns,
    )

    # Calculate metrics
    metrics = await run_cpu_bound(calculate_prediction_metrics, empirical_curve, results["entropy_history"])

    # Run partial fits and log the results
    partial_results = await run_cpu_bound(
        partial_fit_and_evaluate,
        empirical_curve,
        len(empirical_curve),
        [30, 60, 90],
        daily_doc_mu,
        daily_doc_sigma,
        doc_mu,
        doc_sigma,
    )

    logger.info(f"Partial fit results for {entity.name}: {partial_results}")

    return {
        "entity": entity.name,
        "full_fit_metrics": metrics,
        "partial_fits": partial_results,  # Make sure this is being passed correctly
        "params": fitted_params,
        "max_entropy": max(empirical_curve) if empirical_curve else 0,  # Add this line
        "mean_entropy": np.mean(empirical_curve) if empirical_curve else 0,  # Add this line too
    }


async def run_analysis(fact_length_cutoff: int = 250, max_concurrent_tasks: int = N_PHYSICAL_CORES * 10):
    """Run analysis comparing empirical data with entropy-driven model."""
    try:
        # Load empirical data
        entities = load_checkpoint("entity_store_final_entities")
        if not entities:
            raise ValueError("No checkpoint found")

        # Create output directory
        output_dir = settings.output_path / "entropy_model_analysis"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Collect the earliest creation time for each document
        doc_creation_times = {}
        for entity in entities:
            for fact in entity.facts:
                doc_id = fact.source_id
                if doc_id not in doc_creation_times or fact.created_at < doc_creation_times[doc_id]:
                    doc_creation_times[doc_id] = fact.created_at

        # Convert to a list of timestamps and fit daily doc distribution
        doc_timestamps = list(doc_creation_times.values())
        daily_doc_mu, daily_doc_sigma = estimate_daily_doc_distribution(doc_timestamps)
        logger.info(f"Fitted daily doc distribution: mu={daily_doc_mu:.3f}, sigma={daily_doc_sigma:.3f}")

        # Get the daily counts for K-S test
        daily_counts = {}
        for ts in doc_timestamps:
            day = ts.date()
            daily_counts[day] = daily_counts.get(day, 0) + 1
        daily_counts_array = np.array(list(daily_counts.values()))

        # Evaluate fits
        ks_stat, p_val = evaluate_lognormal_fit(daily_counts_array + 1e-9, daily_doc_mu, daily_doc_sigma)
        logger.info(f"Daily doc K-S test: statistic={ks_stat:.4f}, p_value={p_val:.4f}")

        # Fit fact-length distribution
        doc_mu, doc_sigma = estimate_factlength_distribution(entities)
        logger.info(f"Fitted fact-length distribution: mu={doc_mu:.3f}, sigma={doc_sigma:.3f}")

        # Build doc_fact_counts for K-S test
        doc_fact_counts = {}
        for entity in entities:
            for fact in entity.facts:
                doc_id = fact.source_id
                doc_fact_counts[doc_id] = doc_fact_counts.get(doc_id, 0) + 1

        fact_counts_array = np.array(list(doc_fact_counts.values()))
        ks_stat, p_val = evaluate_lognormal_fit(fact_counts_array + 1e-9, doc_mu, doc_sigma)
        logger.info(f"Fact length K-S test: statistic={ks_stat:.4f}, p_value={p_val:.4f}")

        # After K-S tests, add visualization
        plot_distribution_fit(fact_counts_array, doc_mu, doc_sigma, "Fact Length", output_dir)
        plot_distribution_fit(daily_counts_array, daily_doc_mu, daily_doc_sigma, "Daily Document", output_dir)

        # Find high-entropy entities for comparison
        high_entropy_entities = []

        for entity in tqdm(entities, desc="Finding high-entropy entities"):
            if len(entity.facts) < fact_length_cutoff:  # Skip entities with too few facts
                continue

            entropy_curve, _ = calculate_empirical_entropy_curve(entity)
            if entropy_curve and max(entropy_curve) > 0.0:  # Threshold for "high entropy"
                high_entropy_entities.append((entity, entropy_curve))

        logger.info(f"Found {len(high_entropy_entities)} high-entropy entities")

        # Create async tasks for each entity with rate limiting
        tasks = []
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        chunk_size = 10  # Process multiple entities per worker

        async def process_entity_batch(entities_batch):
            async with semaphore:
                return await asyncio.gather(
                    *[
                        analyze_single_entity(
                            entity,
                            empirical_curve,
                            doc_mu,
                            doc_sigma,
                            daily_doc_mu,
                            daily_doc_sigma,
                            output_dir,
                        )
                        for entity, empirical_curve in entities_batch
                    ]
                )

        # Split entities into batches
        entity_batches = [
            high_entropy_entities[i : i + chunk_size] for i in range(0, len(high_entropy_entities), chunk_size)
        ]

        tasks = [process_entity_batch(batch) for batch in entity_batches]

        best_fit_summaries = []
        for batch_results in await execute_tasks_with_manual_pbar(tasks, desc="Analyzing entity batches"):
            best_fit_summaries.extend(batch_results)

        # Clean up process pool
        process_pool.shutdown()

        # Generate summary visualizations
        plot_fit_summary(best_fit_summaries, output_dir)
        plot_fit_metrics_scatter(best_fit_summaries, output_dir)  # Add this line

        # Prepare evaluation results for CSV
        eval_results = []
        for summary in best_fit_summaries:
            if not summary:  # Skip None results
                continue

            entity_name = summary["entity"]
            metrics = summary.get("full_fit_metrics", {})
            params = summary.get("params", None)

            # Find the entity object to get metadata
            entity_obj = next((e for e, _ in high_entropy_entities if e.name == entity_name), None)
            if not entity_obj:
                continue

            # Calculate empirical curve for metadata
            emp_curve, _ = calculate_empirical_entropy_curve(entity_obj)

            result = {
                "entity_name": entity_name,
                "num_facts": len(entity_obj.facts),
                "num_sources": len({f.source_id for f in entity_obj.facts}),
                "time_span_days": (entity_obj.facts[-1].created_at - entity_obj.facts[0].created_at).days,
                "max_entropy": max(emp_curve) if emp_curve else 0,
                "mean_entropy": np.mean(emp_curve) if emp_curve else 0,
                "mse": metrics.get("mse", None),
                "mae": metrics.get("mae", None),
                "r2": metrics.get("r2", None),
            }

            # Add model parameters
            if params:
                result.update(
                    {
                        "alpha_E": params.alpha_E,
                        "delta": params.delta,
                        "alpha_global": params.alpha_global,
                        "alpha_local": params.alpha_local,
                        "alpha_docs": params.alpha_docs,
                        "mu": params.mu,
                        "sigma": params.sigma,
                    }
                )

            # Add partial fit metrics
            for interval in [30, 60, 90]:
                partial_data = summary.get("partial_fits", {}).get(interval, {})
                if "metrics" in partial_data:
                    result[f"partial_mse_{interval}"] = partial_data["metrics"].get("mse")
                    result[f"partial_mae_{interval}"] = partial_data["metrics"].get("mae")
                    result[f"partial_r2_{interval}"] = partial_data["metrics"].get("r2")

            eval_results.append(result)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(eval_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"model_evaluation_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Evaluation results saved to: {csv_path}")

        # Log aggregate results
        logger.info("\nAggregate Results:")

        # Get all possible intervals from all summaries
        all_intervals = set()
        for summary in best_fit_summaries:
            if "partial_fits" in summary:
                all_intervals.update(summary["partial_fits"].keys())

        intervals = sorted(list(all_intervals))

        for interval in intervals:
            valid_results = []
            for summary in best_fit_summaries:
                if (
                    "partial_fits" in summary
                    and interval in summary["partial_fits"]
                    and "metrics" in summary["partial_fits"][interval]
                    and "mse" in summary["partial_fits"][interval]["metrics"]
                ):
                    valid_results.append(summary["partial_fits"][interval]["metrics"]["mse"])

            if valid_results:  # Only log if we have valid results
                logger.info(f"\nTraining Period: {interval} days")
                logger.info(f"  Mean MSE: {np.mean(valid_results):.4f}")
                logger.info(f"  Median MSE: {np.median(valid_results):.4f}")
                logger.info(f"  Std MSE: {np.std(valid_results):.4f}")
                logger.info(f"  Valid samples: {len(valid_results)}")

    finally:
        process_pool.shutdown()


def plot_entropy_comparison_dual(
    empirical_data,
    sampling_results,
    actual_results,
    sampling_params,
    actual_params,
    entity_name,
    output_dir,
    growth_patterns=None,
):
    """Plot comparison showing both sampling-based and actual document count results."""
    plt.figure(figsize=(12, 8))

    # Plot empirical data
    plt.plot(np.arange(len(empirical_data)), empirical_data, "k-", label="Empirical", linewidth=2, zorder=5)

    # Plot sampling-based results
    plt.plot(
        np.arange(len(sampling_results["entropy_history"])),
        sampling_results["entropy_history"],
        "b--",
        label="Model (Sampled Docs)",
        alpha=0.7,
    )

    # Plot actual-docs results
    plt.plot(
        np.arange(len(actual_results["entropy_history"])),
        actual_results["entropy_history"],
        "r--",
        label="Model (Actual Docs)",
        alpha=0.7,
    )

    # Highlight single-document period if available
    if growth_patterns and growth_patterns["single_doc_period"] > 0:
        plt.axvspan(0, growth_patterns["single_doc_period"], color="gray", alpha=0.2, label="Single-doc period")

    # Add parameter details to plot
    param_text = (
        f"Sampling Params | Actual Params\n"
        f"α_E: {sampling_params.alpha_E:.3f} | {actual_params.alpha_E:.3f}\n"
        f"δ: {sampling_params.delta:.3f} | {actual_params.delta:.3f}\n"
        f"α_global: {sampling_params.alpha_global:.3f} | {actual_params.alpha_global:.3f}\n"
        f"α_local: {sampling_params.alpha_local:.3f} | {actual_params.alpha_local:.3f}\n"
        f"c: {sampling_params.constant:.3f} | {actual_params.constant:.3f}"  # Added constant to parameter display
    )

    plt.text(
        0.02,
        0.98,
        param_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Entropy (bits)", fontsize=12)
    plt.title(f"Entropy Evolution: {entity_name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.savefig(
        output_dir / f"entropy_comparison_{_sanitize_path(entity_name.lower())}_.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
