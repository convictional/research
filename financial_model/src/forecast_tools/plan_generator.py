import json
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any
from .forecasting_models import run_prophet, run_moving_average, run_regression, ForecastResult
from ..instruct_llm import ainstruct_llm
from ..helpers.print_section import print_section, Colors
import base64
from pathlib import Path


class MethodConfig(BaseModel):
    params: Dict[str, Any]
    reasoning: str


class AnalysisStep(BaseModel):
    tool: str
    configs: List[MethodConfig]
    reasoning: str


class AnalysisPlan(BaseModel):
    steps: List[AnalysisStep]
    explanation: str
    iteration: int = 1  # Track which iteration we're on


class FinancialAnalyst:
    """Lead agent that coordinates the analysis and forecasting process"""

    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame, business_context: str, target_column: str):
        self.train_data = train_data
        self.val_data = val_data
        self.business_context = business_context
        self.target_column = target_column
        self.iteration_count = 0
        self.initial_exploration_iterations = 3
        self.tools = {
            "prophet": self._run_prophet,
            "exponential_smoothing": self._run_moving_average,
            "regression": self._run_regression,
        }
        self.current_experiment_id = None  # Add experiment_id tracking

    def _calculate_exploration_rate(self) -> float:
        """Calculate exploration rate based on iteration number"""
        if self.iteration_count <= self.initial_exploration_iterations:
            return 0.8
        else:
            decay_rate = 0.5
            min_exploration = 0.1
            exploration_rate = 0.8 * np.exp(-decay_rate * (self.iteration_count - self.initial_exploration_iterations))
            return max(exploration_rate, min_exploration)

    def _load_plot_images(self) -> List[Dict[str, Any]]:
        """Load forecast plot images from the current iteration"""
        if not self.current_experiment_id:
            return []

        images = []
        plot_dir = Path(__file__).parent.parent.parent / "src/config/output/plots"
        current_iteration = self.iteration_count - 1
        start_idx = current_iteration * 12
        end_idx = start_idx + 12

        # Updated to use experiment_id in filename
        for idx in range(start_idx, end_idx):
            plot_path = plot_dir / f"{self.current_experiment_id}_forecast_plot_{idx:03d}.jpg"
            if plot_path.exists():
                with open(plot_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    images.append(
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}}
                    )

        return images

    async def create_forecast_plan(
        self,
        target_column: str,
        target_horizon: int,
        previous_results: Dict[str, Any] = None,
        experiment_id: str = None,
    ) -> AnalysisPlan:
        """Create a plan for analyzing and forecasting the target column"""
        self.current_experiment_id = experiment_id  # Store experiment_id
        self.iteration_count += 1
        exploration_rate = self._calculate_exploration_rate()

        print_section("Creating Forecast Plan", Colors.YELLOW)

        # Load plot images if available
        plot_images = self._load_plot_images()

        # Construct message content with images
        message_content = []

        # Add system prompt
        system_prompt = f"""
You are an expert data scientist specializing in time series forecasting. You are a student of Nate Silver and have a knack for translating data into actionable insights for business.
You have also had a long and varied career forecasting just about everything in business, from
stock prices, inventory levels, buying patterns, product growth, etc. Given data and business context,
create a step-by-step plan for forecasting. You have access to these forecasting tools:
Current Iteration: {self.iteration_count}
Exploration Rate: {exploration_rate:.2f}

Strategy for this iteration:
{
    "HEAVY EXPLORATION - Focus on diverse parameter combinations and wide parameter ranges"
    if exploration_rate > 0.5 else
    "FOCUSED EXPLOITATION - Fine-tune around successful configurations with smaller adjustments"
}

Parameter selection strategy:
- Exploration configurations: {int(exploration_rate * 100)}% of suggestions
- Exploitation configurations: {int((1 - exploration_rate) * 100)}% of suggestions

{
    "During this exploration phase, prioritize:"
    if exploration_rate > 0.5 else
    "During this exploitation phase, prioritize:"
}
{
    "- Testing diverse parameter combinations\n- Using wider parameter ranges\n- Trying extreme values"
    if exploration_rate > 0.5 else
    "- Fine-tuning successful configurations\n- Making smaller parameter adjustments\n- Focusing on proven models"
}

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
    Important:
    - Multiplicative trend/seasonality requires strictly positive data
    - If data contains zero or negative values, use additive components only
    - The model will automatically fall back to additive components if needed

3. regression:
    - features: list of ['trend', 'month', 'day_of_week', 'week_of_year', 'quarter']
    - forecast_periods: {target_horizon}

Choose appropriate tools and parameters based on:
- Data characteristics and patterns
- Business context and seasonality
- The need for interpretability vs accuracy
For each forecasting method, provide FOUR different parameter configurations that explore different aspects:
1. Conservative/baseline configuration
2. Aggressive/complex configuration
3. Balanced configuration
4. Experimental configuration

Base your new configurations on any previous results if provided.

Return a structured plan with reasoning for each step.

Exploration Strategy:
1. For each method, ensure your configurations include:
    - One configuration close to the previous best (exploitation)
    - One configuration that explores new parameter ranges (exploration)
    - One configuration that combines successful parameters from different methods
    - One configuration that tests extreme parameter values

## When Using Previous Results
- Analyze unexplored parameter ranges
- Identify performance-sensitive parameters
- Combine successful parameters across iterations
- Avoid exact configuration repetition

# Selection Criteria
Base choices on:
1. Data characteristics and patterns
2. Business context and seasonality
3. Interpretability vs accuracy trade-offs

# Exploration Balance
Maintain ratio between:
- Exploitation: 70% of suggestions
  - Fine-tune known good parameters
  - Small adjustments to successful configs
- Exploration: 30% of suggestions
  - Test new parameter combinations
  - Probe unexplored parameter spaces
"""
        # Add plot images with labels
        for i, img in enumerate(plot_images):
            message_content.extend([{"type": "text", "text": f"Image {i+1}:"}, img])

        # Add user prompt with data
        message_content.append(
            {
                "type": "text",
                "text": f"""
# Business Context:
{self.business_context}


# Data Summary:
{str(self._get_data_summary(target_column))}


# Target Column:
{str(target_column)}


# Previous Results:
{json.dumps(previous_results) if previous_results else "None"}


# Previous Results Analysis:
{json.dumps(self._analyze_previous_results(previous_results), indent=2)}


# Parameter Space Coverage:
{json.dumps(self._analyze_parameter_coverage(previous_results), indent=2)}


Create a detailed forecasting plan specifying:
1. Which tools to use
2. FOUR parameter configurations for each tool
3. Reasoning for each configuration
""",
            }
        )

        messages = [{"role": "user", "content": system_prompt}, {"role": "user", "content": message_content}]

        result = await ainstruct_llm(messages=messages, response_model=AnalysisPlan)

        plan = result[0]

        print_section(
            f"Generated Forecast Plan - Iteration {plan.iteration}",
            f"Explanation: {plan.explanation}\n\nSteps:\n"
            + "\n".join([f"- {step.tool}: {len(step.configs)} configs" for step in plan.steps]),
            Colors.YELLOW,
        )

        # Add detailed config printing
        print_section(
            f"Generated Forecast Plan - Iteration {plan.iteration}",
            f"Explanation: {plan.explanation}",
            Colors.YELLOW,
        )

        for step in plan.steps:
            print_section(
                f"Configurations for {step.tool}",
                "\n".join(
                    [
                        f"Config {i+1}:"
                        f"\nParameters: {json.dumps(config.params, indent=2)}"
                        f"\nReasoning: {config.reasoning}\n"
                        for i, config in enumerate(step.configs)
                    ]
                ),
                Colors.YELLOW,
            )

        return plan

    async def execute_plan(self, plan: AnalysisPlan, forecast_periods: int) -> Dict[str, List[ForecastResult]]:
        """Execute all configurations in the plan"""
        forecasts = {}

        for step in plan.steps:
            forecasts[step.tool] = []

            for config in step.configs:
                params = config.params.copy()  # Make a copy to avoid modifying original
                # Remove 'forecast_periods' from params if present
                params.pop("forecast_periods", None)

                # Initialize result as None
                result = None

                try:
                    if step.tool == "prophet":
                        df = self.train_data.copy()
                        df = df.reset_index()
                        df = df.rename(columns={"date_id": "ds", self.target_column: "y"})
                        result = run_prophet(df, params, forecast_periods, self.target_column)
                    elif step.tool == "exponential_smoothing":
                        result = run_moving_average(self.train_data, params, forecast_periods, self.target_column)
                    elif step.tool == "regression":
                        result = run_regression(self.train_data, params, forecast_periods, self.target_column)

                    # Only add reasoning if result was successfully created
                    if result:
                        result.config_reasoning = config.reasoning
                        forecasts[step.tool].append(result)
                    else:
                        print(f"Warning: Failed to generate forecast for {step.tool} with config: {params}")

                except Exception as e:
                    print(f"Error executing {step.tool} with config {params}: {str(e)}")
                    continue

        return forecasts

    def _get_data_summary(self, target_column: str) -> str:
        """Generate a summary of the relevant data characteristics"""
        summary = {
            "rows": len(self.train_data),
            "time_range": f"{self.train_data.index.min()} to {self.train_data.index.max()}",
            "target_stats": self.train_data[target_column].describe().to_dict(),
        }
        return str(summary)

    def _run_prophet(self, params: Dict[str, Any]) -> ForecastResult:
        return run_prophet(self.train_data, params)

    def _run_moving_average(self, params: Dict[str, Any]) -> ForecastResult:
        return run_moving_average(self.train_data, params)

    def _run_regression(self, params: Dict[str, Any]) -> ForecastResult:
        return run_regression(self.train_data, params)

    def _analyze_previous_results(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze previous results to guide parameter selection"""
        if not previous_results:
            return {}

        analysis = {
            "best_performing_params": {},
            "unexplored_ranges": {},
            "performance_patterns": {},
            "parameter_sensitivity": {},
        }

        # Define default parameter ranges for each method
        default_ranges = {
            "prophet": {
                "changepoint_prior_scale": (0.001, 0.5),
                "seasonality_prior_scale": (0.01, 100),
                "holidays_prior_scale": (0.01, 100),
                "changepoint_range": (0.4, 0.95),
            },
            "exponential_smoothing": {"window": (1, 30)},
            "regression": {"features": [["trend"], ["month"], ["day_of_week"], ["week_of_year"], ["quarter"]]},
        }

        # Analyze each method's historical performance
        for method in self.tools.keys():
            if method in previous_results:
                configs = previous_results[method]

                # Find best performing parameters
                best_config = min(configs, key=lambda x: x["metrics"]["mape"])
                analysis["best_performing_params"][method] = best_config["params"]

                # Identify unexplored parameter ranges
                if method in default_ranges:
                    used_params = set()
                    for config in configs:
                        for param, value in config["params"].items():
                            if isinstance(value, (int, float)):
                                used_params.add((param, value))
                            elif isinstance(value, list):
                                used_params.add((param, tuple(value)))

                    unexplored = {}
                    for param, range_val in default_ranges[method].items():
                        if isinstance(range_val, tuple):
                            used_values = [v for p, v in used_params if p == param]
                            if used_values:
                                min_used, max_used = min(used_values), max(used_values)
                                unexplored[param] = {
                                    "below_min": (range_val[0], min_used) if min_used > range_val[0] else None,
                                    "above_max": (max_used, range_val[1]) if max_used < range_val[1] else None,
                                }
                        elif isinstance(range_val, list):
                            used_combinations = set(tuple(v) for p, v in used_params if p == param)
                            unexplored[param] = [combo for combo in range_val if tuple(combo) not in used_combinations]

                    analysis["unexplored_ranges"][method] = unexplored

                # Analyze parameter sensitivity
                if len(configs) > 1:
                    sensitivity = {}
                    for param in best_config["params"].keys():
                        values = [c["params"].get(param) for c in configs]
                        metrics = [c["metrics"]["mape"] for c in configs]
                        if all(v is not None for v in values) and len(set(values)) > 1:
                            correlation = np.corrcoef(values, metrics)[0, 1]
                            sensitivity[param] = abs(correlation)
                    analysis["parameter_sensitivity"][method] = sensitivity

        return analysis

    def _analyze_parameter_coverage(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well the parameter space has been covered"""
        if not previous_results:
            return {}

        coverage = {}
        for method in self.tools.keys():
            if method in previous_results:
                params_used = set()
                param_values = {}

                for config in previous_results[method]:
                    param_key = frozenset(config["params"].items())
                    params_used.add(param_key)

                    # Track all values used for each parameter
                    for param, value in config["params"].items():
                        if param not in param_values:
                            param_values[param] = []
                        param_values[param].append(value)

                coverage[method] = {
                    "unique_configs": len(params_used),
                    "parameter_frequencies": self._calculate_param_frequencies(param_values),
                    "coverage_score": self._calculate_coverage_score(param_values),
                }

        return coverage

    def _calculate_param_frequencies(self, param_values: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate the frequency of different parameter values"""
        frequencies = {}

        for param, values in param_values.items():
            if not values:
                continue

            if isinstance(values[0], (int, float)):
                # For numeric parameters, create bins
                if len(values) > 1:
                    hist, bins = np.histogram(values, bins="auto")
                    frequencies[param] = {
                        f"{bins[i]:.2f}-{bins[i+1]:.2f}": count / len(values) for i, count in enumerate(hist)
                    }
            else:
                # For categorical parameters, count occurrences
                value_counts = {}
                total = len(values)
                for value in values:
                    if isinstance(value, list):
                        value = tuple(value)  # Make list hashable
                    value_counts[str(value)] = value_counts.get(str(value), 0) + 1 / total
                frequencies[param] = value_counts

        return frequencies

    def _calculate_coverage_score(self, param_values: Dict[str, List[Any]]) -> float:
        """Calculate a score representing how well the parameter space is covered"""
        if not param_values:
            return 0.0

        scores = []
        for values in param_values.values():
            if not values:
                continue

            if isinstance(values[0], (int, float)):
                # For numeric parameters, look at the spread
                unique_values = len(set(values))
                range_coverage = (max(values) - min(values)) / (max(values) + 1e-10)  # Avoid division by zero
                scores.append((unique_values / len(values)) * range_coverage)
            else:
                # For categorical parameters, look at unique value ratio
                scores.append(len(set(map(str, values))) / len(values))

        return np.mean(scores) if scores else 0.0
