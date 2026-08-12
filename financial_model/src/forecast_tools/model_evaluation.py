import json
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import base64
from pathlib import Path

from ..instruct_llm import ainstruct_llm
from ..helpers.print_section import print_section, Colors
from .forecasting_models import ForecastResult, plot_forecast


class ModelSelectionResult(BaseModel):
    """Structure for LLM model selection results"""

    selected_model: str
    selected_config: int  # Index of the best config for the selected model
    reasoning: str
    comparative_analysis: str
    confidence_score: float  # 0-1 indicating confidence in selection
    considerations: List[str]
    next_iteration_suggestions: Dict[str, List[str]]  # Suggestions for each method's next configs


class ModelEvaluator:
    """Agent responsible for evaluating and comparing different forecasting approaches"""

    def __init__(self):
        self.historical_suggestions = []  # Track all historical suggestions
        self.historical_performances = []  # Track performance of each suggestion
        self.initial_exploration_iterations = 3  # Number of iterations for heavy exploration
        self.plot_dir = Path(__file__).parent.parent.parent / "src/config/output/plots"
        self.current_experiment_id = None  # Add experiment_id tracking

    def _calculate_exploration_rate(self, current_iteration: int) -> float:
        """Calculate exploration rate based on iteration number"""
        if current_iteration <= self.initial_exploration_iterations:
            # High exploration rate (0.8) during initial iterations
            return 0.8
        else:
            # Exponential decay for exploration rate after initial iterations
            decay_rate = 0.5
            min_exploration = 0.1
            exploration_rate = 0.8 * np.exp(-decay_rate * (current_iteration - self.initial_exploration_iterations))
            return max(exploration_rate, min_exploration)

    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate various error metrics"""
        # Ensure we're comparing the same number of points
        n = min(len(actual), len(predicted))
        actual = actual[:n]
        predicted = predicted[:n]

        # Avoid division by zero
        mask = actual != 0
        if not any(mask):
            return {"mape": float("inf"), "rmse": float("inf"), "mae": float("inf")}

        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))

        return {"mape": mape, "rmse": rmse, "mae": mae}

    def _evaluate_uncertainty(self, forecast: ForecastResult) -> Dict[str, float]:
        """Evaluate the forecast uncertainty using confidence intervals"""
        ci_widths = [upper - lower for lower, upper in forecast.confidence_interval]
        return {
            "avg_ci_width": np.mean(ci_widths),
            "ci_width_std": np.std(ci_widths),
            "uncertainty_score": np.mean(ci_widths) / np.mean(forecast.forecast),
        }

    def _analyze_trend(self, forecast: ForecastResult) -> Dict[str, Any]:
        """Analyze the trend in the forecast"""
        values = np.array(forecast.forecast)
        trend = np.polyfit(np.arange(len(values)), values, 1)
        return {
            "trend_direction": "increasing" if trend[0] > 0 else "decreasing",
            "trend_strength": abs(trend[0]),
            "trend_coefficient": trend[0],
        }

    def _format_data_summary(self, validation_data: pd.Series) -> str:
        """Format data characteristics for LLM context"""
        return {
            "data_points": len(validation_data),
            "date_range": f"{validation_data.index.min()} to {validation_data.index.max()}",
            "mean_value": validation_data.mean(),
            "std_value": validation_data.std(),
            "min_value": validation_data.min(),
            "max_value": validation_data.max(),
        }

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return self._convert_to_serializable(obj.tolist())
        return obj

    def _load_plot_images(self) -> List[Dict[str, Any]]:
        """Load forecast plot images from the current iteration"""
        if not self.current_experiment_id:
            return []

        images = []
        current_iteration = len(self.historical_performances)
        start_idx = current_iteration * 12
        end_idx = start_idx + 12

        # Updated to use experiment_id in filename
        for idx in range(start_idx, end_idx):
            plot_path = self.plot_dir / f"{self.current_experiment_id}_forecast_plot_{idx:03d}.jpg"
            if plot_path.exists():
                with open(plot_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    images.append(
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}}
                    )

        return images

    async def evaluate_forecasts(
        self,
        forecasts: Dict[str, List[ForecastResult]],
        validation_data: pd.Series,
        all_results_history: List[Dict] = None,
        experiment_id: str = None,  # Add experiment_id parameter
    ) -> Dict[str, Any]:
        """Evaluate and compare different forecasting approaches using validation data"""
        self.current_experiment_id = experiment_id  # Store experiment_id

        print_section("Model Evaluation", "Evaluating forecast results against validation data", Colors.BLUE)

        # Handle empty forecasts
        if not forecasts:
            return {
                "best_model": None,
                "best_config": 0,
                "model_evaluations": {},
                "reasoning": "No forecasts to evaluate",
                "comparative_analysis": "No data available",
                "selection_confidence": 0.0,
                "key_considerations": [],
                "next_iteration_suggestions": {},
                "comparative_metrics": {},
            }

        # Use validation data for evaluation
        validation_values = validation_data.values
        target_horizon = len(validation_values)
        evaluation_results = {}

        for method, forecast_list in forecasts.items():
            method_results = []
            for forecast in forecast_list:
                # Take only the first n forecast values where n is the length of validation data
                forecast_values = np.array(forecast.forecast[: len(validation_values)])

                # Calculate metrics using validation data
                metrics = self._calculate_metrics(validation_values, forecast_values)

                # Analyze uncertainty
                uncertainty = self._evaluate_uncertainty(forecast)

                # Analyze trend
                trend_analysis = self._analyze_trend(forecast)

                # Store comprehensive evaluation
                method_results.append(
                    {
                        "metrics": metrics,
                        "uncertainty": uncertainty,
                        "trend_analysis": trend_analysis,
                        "forecast_length": len(forecast.forecast),
                        "avg_forecast_value": np.mean(forecast.forecast),
                    }
                )
            evaluation_results[method] = method_results

        # Ensure we have at least one valid result before proceeding
        if not any(evaluation_results.values()):
            return {
                "best_model": list(forecasts.keys())[0] if forecasts else None,
                "best_config": 0,
                "model_evaluations": evaluation_results,
                "reasoning": "No valid evaluation results",
                "comparative_analysis": "Insufficient data for comparison",
                "selection_confidence": 0.0,
                "key_considerations": [],
                "next_iteration_suggestions": {},
                "comparative_metrics": {},
            }

        # Prepare data summary for LLM context
        data_summary = self._format_data_summary(validation_data)

        # Calculate current exploration rate
        current_iteration = len(self.historical_performances)
        exploration_rate = self._calculate_exploration_rate(current_iteration)

        # Create LLM prompts
        system_prompt = f"""
# Role and Context
You are a seasoned forecasting expert with:
- Training under Nate Silver's methodology
- Extensive experience in business forecasting across domains:
  - Stock prices
  - Inventory levels
  - Buying patterns
  - Product growth

# Primary Objectives
1. Analyze forecasting model evaluation results
2. Select the most appropriate model based on:
   - Accuracy metrics
   - Uncertainty measures
   - Trend analysis
3. Provide structured analysis explaining your selection

# Visual Analysis Guidelines
Examine provided forecast plots for:
- Trend capture accuracy
- Seasonality patterns
- Outlier handling
- Uncertainty bounds appropriateness
- Over/under-fitting patterns
- Configuration effectiveness

# Iteration Strategy
Current Iteration: {current_iteration + 1}
Exploration Rate: {exploration_rate:.2f}
Current Strategy: {"EXPLORATION" if exploration_rate > 0.5 else "EXPLOITATION"}

## Early Phase (Iterations 1-3)
Focus on exploration:
- Diverse parameter combinations
- Extreme parameter values
- Various seasonal patterns
- Wide parameter ranges

## Later Phase (After Iteration 3)
Shift to exploitation:
- Fine-tune successful configurations
- Smaller parameter adjustments
- Focus on promising models
- Narrow parameter ranges

# Available Forecasting Tools

The tools available for forecasting are:
        1. prophet:
           - seasonality_mode: 'additive' or 'multiplicative'
           - yearly_seasonality: 'auto', True, False, or int
           - weekly_seasonality: 'auto', True, False, or int
           - daily_seasonality: 'auto', True, False, or int
           - changepoint_prior_scale: float (default 0.05)
           - seasonality_prior_scale: float (default 10.0)
           - holidays_prior_scale: float (default 10.0)
           - changepoint_range: float (default 0.8)
           - add_seasonalities: list of dicts with name, period, fourier_order
           - holidays: string (SINGLE country code)
           - forecast_periods: {target_horizon}

        2. exponential_smoothing (Holt-Winters):
           - window: int (default 7, represents seasonal_periods for Holt-Winters method)
           - trend: str (default 'add', options: 'add' or 'mul')
           - seasonal: str (default 'add', options: 'add' or 'mul')
           - forecast_periods: {target_horizon}
           Note: Uses Holt-Winters exponential smoothing with trend and seasonal components

        3. regression:
           - features: list of ['trend', 'month', 'day_of_week', 'week_of_year', 'quarter']
           - forecast_periods: {target_horizon}


# Configuration Requirements
Provide FOUR configurations per method:
1. Conservative/baseline
2. Aggressive/complex
3. Balanced
4. Experimental

# Recommendation Strategy
Balance exploration/exploitation by:
1. Including variations of best-performing configurations
2. Adding one significantly different configuration per method
3. Exploring under-utilized parameter spaces
4. Maintaining parameter combination diversity

# Selection Criteria
Base tool and parameter choices on:
- Data characteristics and patterns
- Business context and seasonality
- Trade-off between interpretability and accuracy
        """

        # Track historical performance and suggestions
        current_performance = {
            model: [config["metrics"] for config in configs] for model, configs in evaluation_results.items()
        }
        self.historical_performances.append(current_performance)

        # Enhanced context for LLM
        historical_context = self._convert_to_serializable(
            {
                "past_suggestions": self.historical_suggestions,
                "past_performances": self.historical_performances,
                "performance_trends": self._analyze_performance_trends(),
                "exploration_metrics": self._calculate_exploration_metrics(evaluation_results),
            }
        )

        evaluation_results = self._convert_to_serializable(evaluation_results)
        all_results_history = self._convert_to_serializable(all_results_history) if all_results_history else None

        for method, forecast_list in forecasts.items():
            print_section(f"Plots for {method}", f"Showing {len(forecast_list)} configurations", Colors.BLUE)
            for i, forecast in enumerate(forecast_list):
                plot_forecast(forecast, validation_data, method, i, self.current_experiment_id)
                print("\n")  # Add space between plots

        print_section("Plots Complete", "All plots have been generated and saved", Colors.GREEN)

        user_prompt = f"""
        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Historical Context:
        {json.dumps(historical_context, indent=2)}

        Current Evaluation Results:
        {json.dumps(evaluation_results, indent=2)}

        Previous Results History:
        {json.dumps(all_results_history, indent=2)}
        ...
        """

        # Load plot images
        plot_images = self._load_plot_images()

        # Construct message content with images
        message_content = []

        # Add plot images with labels
        for i, img in enumerate(plot_images):
            message_content.extend(
                [
                    {
                        "type": "text",
                        "text": f"\nPlot {i+1} - "
                        f"Model: {list(forecasts.keys())[i // 4]} "
                        f"Configuration: {(i % 4) + 1}:",
                    },
                    img,
                ]
            )

        # Add evaluation data and context
        message_content.append({"type": "text", "text": user_prompt})

        result = await ainstruct_llm(
            messages=[{"role": "user", "content": system_prompt}, {"role": "user", "content": message_content}],
            response_model=ModelSelectionResult,
        )

        selection = result[0]  # Extract selection from tuple

        # Validate selection before returning
        if selection.selected_model not in forecasts:
            # Default to first available model if selected model is invalid
            selection.selected_model = list(forecasts.keys())[0]

        if selection.selected_config >= len(evaluation_results[selection.selected_model]):
            # Default to first config if selected config is invalid
            selection.selected_config = 0

        report = {
            "best_model": selection.selected_model,
            "best_config": selection.selected_config,
            "model_evaluations": evaluation_results,
            "reasoning": selection.reasoning,
            "comparative_analysis": selection.comparative_analysis,
            "selection_confidence": selection.confidence_score,
            "key_considerations": selection.considerations,
            "next_iteration_suggestions": selection.next_iteration_suggestions,
            "comparative_metrics": {
                f"{model}_config_{i}": results["metrics"]
                for model, configs in evaluation_results.items()
                for i, results in enumerate(configs)
            },
        }

        print_section(
            "Evaluation Results",
            f"Best Model: {selection.selected_model} (Config {selection.selected_config})\n"
            f"Reasoning: {selection.reasoning}\n"
            f"Confidence: {selection.confidence_score}",
            Colors.BLUE,
        )

        print_section("Model Comparison", report["comparative_metrics"], Colors.BLUE)

        print_section(
            "Next Iteration Suggestions",
            "\n".join(
                [
                    f"{method}:\n" + "\n".join(f"- {suggestion}" for suggestion in suggestions)
                    for method, suggestions in selection.next_iteration_suggestions.items()
                ]
            ),
            Colors.GREEN,
        )

        # Plot forecasts for each method and configuration
        print_section("Generating Forecast Plots", "Plotting results to terminal and saving to files", Colors.YELLOW)

        # Store suggestions for next iteration
        self.historical_suggestions.append(selection.next_iteration_suggestions)

        return report

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze trends in historical performance"""
        if not self.historical_performances:
            return {}

        trends = {}
        for model in self.historical_performances[0].keys():
            model_metrics = []
            for perf in self.historical_performances:
                if model in perf:
                    best_metric = min(config["mape"] for config in perf[model])
                    model_metrics.append(best_metric)

            trends[model] = {
                "improving": len(model_metrics) > 1 and model_metrics[-1] < min(model_metrics[:-1]),
                "best_score": min(model_metrics) if model_metrics else float("inf"),
                "trend": np.polyfit(range(len(model_metrics)), model_metrics, 1)[0] if len(model_metrics) > 1 else 0,
            }

        return trends

    def _calculate_exploration_metrics(self, current_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate metrics about parameter space exploration"""
        exploration_metrics = {}

        for model, configs in current_results.items():
            param_ranges = {}
            param_combinations = set()

            # Analyze parameter ranges and combinations
            for config in configs:
                # Convert nested dictionaries to tuples of key-value pairs for hashing
                flattened_config = []
                for param_name, param_value in config.items():
                    if isinstance(param_value, dict):
                        # Sort dict items to ensure consistent hashing
                        flattened_config.extend(sorted(param_value.items()))
                    else:
                        flattened_config.append((param_name, param_value))

                param_combinations.add(tuple(sorted(flattened_config)))

                for param, value in flattened_config:
                    if param not in param_ranges:
                        param_ranges[param] = []
                    param_ranges[param].append(value)

            exploration_metrics[model] = {
                "unique_combinations": len(param_combinations),
                "param_diversity": {
                    param: len(set(str(v) for v in values)) / len(values) for param, values in param_ranges.items()
                },
            }

        return exploration_metrics
