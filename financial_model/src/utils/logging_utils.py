from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Tuple, Union
import uuid


class AccuracyMetrics(BaseModel):
    metrics: Dict[str, float]
    uncertainty: Dict[str, float]
    trend_analysis: Dict[str, str | float]


class ForecastLog(BaseModel):
    @staticmethod
    def _get_current_time():
        return datetime.now()

    experiment_id: str
    iteration_id: str
    forecast_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=_get_current_time)
    forecast_model_name: str
    config_index: int
    config_params: Dict[str, Any]
    config_reasoning: str
    forecast_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    accuracy_metrics: AccuracyMetrics
    target_column: str
    forecast_horizon: int
    data_points_used: int


class IterationLog(BaseModel):
    @staticmethod
    def _get_current_time():
        return datetime.now()

    experiment_id: str
    iteration_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=_get_current_time)
    iteration_number: int
    llm_prompt: str
    llm_response: str
    best_model_name: str
    best_config_index: int
    best_model_metrics: AccuracyMetrics
    iteration_duration: float
    models_attempted: List[str]
    total_configs_tested: int


class InterpretationResult(BaseModel):
    key_insights: List[str]
    recommendations: List[str]
    metrics: Dict[str, Dict[str, float]]


class ExperimentLog(BaseModel):
    @staticmethod
    def _get_current_time():
        return datetime.now()

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=_get_current_time)
    llm_model: str
    target_column: str
    business_context: str
    data_path: str
    total_iterations: int
    total_duration: float
    final_interpretation: InterpretationResult
    best_model_summary: Dict[str, Any]
    final_accuracy_metrics: AccuracyMetrics
    total_forecasts_generated: int
    experiment_params: Dict[str, Any]


class ExperimentVisualizer:
    def __init__(self, base_path: Path):
        self.plots_path = base_path / "visualizations"
        self.plots_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_metric_value(row, metric_name):
        """Get specific metric value from row data"""
        try:
            # Convert single quotes to double quotes for valid JSON
            metrics_str = row.best_model_metrics.replace("'", '"')
            metrics_dict = json.loads(metrics_str)
            return metrics_dict.get("metrics", {}).get(metric_name, float("nan"))
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error processing row metrics: {e}")
            return float("nan")

    @staticmethod
    def get_mape_value(metrics_data):
        """Get MAPE value from metrics data"""
        try:
            # Convert single quotes to double quotes for valid JSON
            metrics_str = metrics_data.replace("'", '"') if isinstance(metrics_data, str) else json.dumps(metrics_data)
            metrics_dict = json.loads(metrics_str)
            return metrics_dict.get("metrics", {}).get("mape", float("nan"))
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error processing metrics data: {e}")
            return float("nan")

    def plot_iteration_accuracies(self, experiment_id: str, iteration_data: pd.DataFrame) -> Path:
        """Plot accuracy metrics across iterations for a single experiment"""
        plt.figure(figsize=(12, 6))
        metrics = ["mape", "rmse", "mae"]

        for metric in metrics:
            values = iteration_data.apply(lambda x: self.get_metric_value(x, metric), axis=1)
            plt.plot(iteration_data.iteration_number, values, marker="o", label=metric.upper())

        plt.xlabel("Iteration")
        plt.ylabel("Error Metric Value")
        plt.title(f"Forecast Accuracy Across Iterations\nExperiment: {experiment_id}")
        plt.legend()
        plt.grid(True)

        plot_path = self.plots_path / f"accuracy_progression_{experiment_id}.png"
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def plot_model_comparison(self, experiment_id: str, forecast_data: pd.DataFrame) -> Path:
        """Plot comparison of different models' performance"""
        plt.figure(figsize=(12, 6))

        # Fix column name - try both possible names
        model_name_col = "forecast_model_name"
        if model_name_col not in forecast_data.columns:
            print(f"Available columns: {forecast_data.columns}")
            raise KeyError("'forecast_model_name' not found in columns")

        model_metrics = forecast_data.groupby(model_name_col).agg(
            {"accuracy_metrics": lambda x: [self.get_mape_value(m) for m in x]}
        )

        # Filter out NaN values before plotting
        model_metrics_filtered = {
            model: [x for x in metrics if not np.isnan(x)]
            for model, metrics in model_metrics["accuracy_metrics"].items()
        }

        # Only plot models with valid metrics
        valid_models = [model for model, metrics in model_metrics_filtered.items() if metrics]
        if not valid_models:
            print("No valid metrics found for any model")
            return None

        plt.boxplot([model_metrics_filtered[model] for model in valid_models], labels=valid_models)

        plt.xlabel("Model")
        plt.ylabel("MAPE")
        plt.title(f"Model Performance Comparison\nExperiment: {experiment_id}")
        plt.xticks(rotation=45)
        plt.grid(True)

        plot_path = self.plots_path / f"model_comparison_{experiment_id}.png"
        plt.savefig(plot_path)
        plt.close()

        return plot_path


class ForecastLogger:
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(__file__).parent.parent.parent / "src/config/output/experiment_logs"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.forecasts_path = self.base_path / "forecast_logs.csv"
        self.iterations_path = self.base_path / "iteration_logs.csv"
        self.experiments_path = self.base_path / "experiment_logs.csv"

        # Initialize files if they don't exist
        self._initialize_files()
        self.visualizer = ExperimentVisualizer(self.base_path)

    def _initialize_files(self):
        """Create files with headers if they don't exist"""

        if not self.forecasts_path.exists():
            pd.DataFrame(columns=ForecastLog.__annotations__.keys()).to_csv(self.forecasts_path, index=False)

        if not self.iterations_path.exists():
            pd.DataFrame(columns=IterationLog.__annotations__.keys()).to_csv(self.iterations_path, index=False)

        if not self.experiments_path.exists():
            pd.DataFrame(columns=ExperimentLog.__annotations__.keys()).to_csv(self.experiments_path, index=False)

    def log_forecast(self, forecast: ForecastLog):
        """Log a single forecast"""
        df = pd.DataFrame([forecast.model_dump()])
        df.to_csv(self.forecasts_path, mode="a", header=False, index=False)

    def log_iteration(self, iteration: IterationLog):
        """Log an iteration"""
        df = pd.DataFrame([iteration.model_dump()])
        df.to_csv(self.iterations_path, mode="a", header=False, index=False)

    def log_experiment(self, experiment: ExperimentLog):
        """Log an experiment"""
        df = pd.DataFrame([experiment.model_dump()])
        df.to_csv(self.experiments_path, mode="a", header=False, index=False)

    def get_experiment_data(self, experiment_id: str) -> Dict[str, pd.DataFrame]:
        """Retrieve all data for a specific experiment"""
        forecasts = pd.read_csv(self.forecasts_path)
        iterations = pd.read_csv(self.iterations_path)
        experiments = pd.read_csv(self.experiments_path)

        exp_forecasts = forecasts[forecasts.experiment_id == experiment_id]
        exp_iterations = iterations[iterations.experiment_id == experiment_id]
        exp_experiment = experiments[experiments.experiment_id == experiment_id]

        return {"forecasts": exp_forecasts, "iterations": exp_iterations, "experiment": exp_experiment}

    def create_experiment_visualizations(self, experiment_id: str) -> Dict[str, Path]:
        """Generate and save all visualizations for an experiment"""
        data = self.get_experiment_data(experiment_id)

        plots = {
            "accuracy_progression": self.visualizer.plot_iteration_accuracies(experiment_id, data["iterations"]),
            "model_comparison": self.visualizer.plot_model_comparison(experiment_id, data["forecasts"]),
        }

        return plots
