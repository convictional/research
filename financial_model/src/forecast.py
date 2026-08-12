import asyncio
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import time
import uuid

from .helpers.print_section import print_section, Colors
from .forecast_tools.data_preprocess import DataPreprocessor
from .forecast_tools.plan_generator import FinancialAnalyst
from .forecast_tools.model_evaluation import ModelEvaluator
from .forecast_tools.results_interpreter import ResultsInterpreter
from .utils.logging_utils import ForecastLogger, ForecastLog, IterationLog, ExperimentLog, AccuracyMetrics
from .utils.gifteval_data import GiftEvalDataLoader
from .config.experiment_settings import settings


async def run_multiple_experiments(
    data_source: str = "csv",
    data_path: str | Path | None = None,
    target_column: str = None,
    business_context: str = "",
    target_horizon: int = 20,
    num_iterations: int = 2,
    llm_model: str = settings.llm_model,
    n_series: int | None = None,
) -> List[Dict[str, Any]]:
    """Run forecast experiments on multiple time series"""

    # Load data based on source type
    if data_source == "csv":
        results = [
            await run_forecast_pipeline(
                data_path=data_path,
                target_column=target_column,
                business_context=business_context,
                target_horizon=target_horizon,
                num_iterations=num_iterations,
                llm_model=llm_model,
                data_source="csv",
            )
        ]
    elif data_source == "gifteval":
        loader = GiftEvalDataLoader()
        all_series = loader.load_timeseries(n_series=n_series)

        # Set default target column for GiftEval data if not provided
        if not target_column:
            target_column = "value"

        results = []
        for series_idx in range(len(all_series)):
            print_section(
                f"Running Experiment for Time Series {series_idx + 1}/{len(all_series)}",
                f"Series ID: {series_idx}",
                Colors.CYAN,
            )

            # Extract single time series
            series_data = all_series.iloc[[series_idx]]

            # Run forecast pipeline for this series
            result = await run_forecast_pipeline(
                data_source="gifteval",
                target_column=target_column,
                business_context=f"{business_context} (Series {series_idx + 1})",
                target_horizon=target_horizon,
                num_iterations=num_iterations,
                llm_model=llm_model,
                _input_data=series_data,  # New parameter to pass pre-loaded data
            )

            results.append(result)

    return results


async def run_forecast_pipeline(
    data_path: str | Path | None = None,
    target_column: str = None,
    business_context: str = "",
    target_horizon: int = 20,
    num_iterations: int = 2,
    llm_model: str = settings.llm_model,
    data_source: str = "csv",
    n_series: int | None = None,
    _input_data: pd.DataFrame = None,  # New parameter for pre-loaded data
) -> Dict[str, Any]:
    """Main function to orchestrate the forecasting pipeline with iterative hyperparameter optimization"""

    # Initialize experiment tracking
    experiment_start_time = time.time()
    experiment_id = str(uuid.uuid4())
    logger = ForecastLogger(Path(__file__).parent / "logs")

    print_section(
        "Starting Forecast Pipeline",
        f"Data Path: {data_path}\nTarget Column: {target_column}\nIterations: {num_iterations}\nExperiment ID: {experiment_id}",
        Colors.WHITE,
    )

    # Load data based on source type
    if _input_data is not None:
        data = _input_data
    elif data_source == "csv":
        if not data_path:
            raise ValueError("data_path is required for CSV data source")
        data = pd.read_csv(data_path)
    elif data_source == "gifteval":
        loader = GiftEvalDataLoader()
        data = loader.load_timeseries(n_series=n_series)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")

    # Continue with existing pipeline
    preprocessor = DataPreprocessor(data_source=data_source)
    train_data, val_data = await preprocessor.prepare_data(data, target_column)

    # Calculate total forecast length needed (validation period + target horizon)
    total_forecast_length = len(val_data) + target_horizon

    # Initialize agents
    analyst = FinancialAnalyst(train_data, val_data, business_context, target_column)
    evaluator = ModelEvaluator()

    # Store all forecasts and evaluations across iterations
    all_forecasts = {}
    iteration_evaluations = []
    iteration_plans = []

    # Iterate through refinement process
    for iteration in range(num_iterations):
        iteration_start_time = time.time()
        iteration_id = str(uuid.uuid4())

        print_section(
            f"Starting Iteration {iteration + 1}/{num_iterations}", "Generating forecast configurations", Colors.CYAN
        )

        # Create and execute plan using previous results if available
        all_previous_results = iteration_evaluations
        current_plan = await analyst.create_forecast_plan(
            target_column, total_forecast_length, previous_results=all_previous_results, experiment_id=experiment_id
        )
        iteration_plans.append(current_plan)

        # Remove target_column argument as it's not needed in execute_plan
        current_forecasts = await analyst.execute_plan(current_plan, total_forecast_length)

        # Combine results with previous iterations
        for method, forecasts in current_forecasts.items():
            if method in all_forecasts:
                all_forecasts[method].extend(forecasts)
            else:
                all_forecasts[method] = forecasts

        # Evaluate current iteration
        current_evaluation = await evaluator.evaluate_forecasts(
            current_forecasts, val_data[target_column], all_previous_results, experiment_id=experiment_id
        )
        iteration_evaluations.append(current_evaluation)

        # Log each forecast
        for method, forecasts in current_forecasts.items():
            for idx, forecast in enumerate(forecasts):
                forecast_log = ForecastLog(
                    experiment_id=experiment_id,
                    iteration_id=iteration_id,
                    forecast_model_name=method,
                    config_index=idx,
                    config_params=current_plan.steps[0].configs[idx].params,
                    config_reasoning=forecast.config_reasoning,
                    forecast_values=forecast.forecast,
                    confidence_intervals=forecast.confidence_interval,
                    accuracy_metrics=AccuracyMetrics(
                        metrics=current_evaluation["model_evaluations"][method][idx]["metrics"],
                        uncertainty=current_evaluation["model_evaluations"][method][idx]["uncertainty"],
                        trend_analysis=current_evaluation["model_evaluations"][method][idx]["trend_analysis"],
                    ),
                    target_column=target_column,
                    forecast_horizon=target_horizon,
                    data_points_used=len(train_data),
                )
                logger.log_forecast(forecast_log)

        # Add defensive checks when creating IterationLog
        best_model = current_evaluation.get("best_model")
        best_config = current_evaluation.get("best_config", 0)

        # Default metrics if we can't get them
        default_metrics = AccuracyMetrics(
            metrics={"mape": float("inf"), "rmse": float("inf"), "mae": float("inf")},
            uncertainty={
                "avg_ci_width": float("inf"),
                "ci_width_std": float("inf"),
                "uncertainty_score": float("inf"),
            },
            trend_analysis={
                "trend_direction": "unknown",
                "trend_strength": float("inf"),
                "trend_coefficient": float("inf"),
            },
        )

        try:
            best_metrics = (
                AccuracyMetrics(
                    metrics=current_evaluation["model_evaluations"][best_model][best_config]["metrics"],
                    uncertainty=current_evaluation["model_evaluations"][best_model][best_config]["uncertainty"],
                    trend_analysis=current_evaluation["model_evaluations"][best_model][best_config]["trend_analysis"],
                )
                if best_model and best_model in current_evaluation["model_evaluations"]
                else default_metrics
            )
        except (KeyError, IndexError):
            best_metrics = default_metrics

        iteration_log = IterationLog(
            experiment_id=experiment_id,
            iteration_number=iteration,
            llm_prompt=current_plan.explanation,
            llm_response=str(current_plan),
            best_model_name=best_model,
            best_config_index=best_config,
            best_model_metrics=best_metrics,
            iteration_duration=time.time() - iteration_start_time,
            models_attempted=[step.tool for step in current_plan.steps],
            total_configs_tested=sum(len(step.configs) for step in current_plan.steps),
        )
        logger.log_iteration(iteration_log)

    # Get best overall forecast from all iterations
    final_evaluation = await evaluator.evaluate_forecasts(all_forecasts, val_data[target_column])
    best_method = final_evaluation["best_model"]
    best_config_idx = final_evaluation["best_config"]
    best_forecast = all_forecasts[best_method][best_config_idx]

    # Interpret final results
    interpreter = ResultsInterpreter()
    interpretation = await interpreter.interpret_results(
        best_forecast,
        business_context,
        final_evaluation["model_evaluations"][best_method][best_config_idx],
        train_data[target_column],
    )

    # Log experiment
    experiment_log = ExperimentLog(
        experiment_id=experiment_id,
        llm_model=llm_model,
        target_column=target_column,
        business_context=business_context,
        data_path=str(data_path),
        total_iterations=num_iterations,
        total_duration=time.time() - experiment_start_time,
        final_interpretation=interpretation,  # interpretation is already a dict from ResultsInterpreter
        best_model_summary={"model": best_method, "config": best_config_idx},
        final_accuracy_metrics=AccuracyMetrics(
            metrics=final_evaluation["model_evaluations"][best_method][best_config_idx]["metrics"],
            uncertainty=final_evaluation["model_evaluations"][best_method][best_config_idx]["uncertainty"],
            trend_analysis=final_evaluation["model_evaluations"][best_method][best_config_idx]["trend_analysis"],
        ),
        total_forecasts_generated=sum(len(f) for f in all_forecasts.values()),
        experiment_params={
            "target_horizon": target_horizon,
            "num_iterations": num_iterations,
            "data_shape": train_data.shape,
        },
    )
    logger.log_experiment(experiment_log)

    # Generate visualizations
    visualization_paths = logger.create_experiment_visualizations(experiment_id)

    print_section(
        "Pipeline Complete",
        f"Best Model: {best_method} (config {best_config_idx})\n"
        f"Interpretation: {interpretation}\n"
        f"Total Configurations Tested: {sum(len(f) for f in all_forecasts.values())}\n"
        f"Visualizations saved to:\n" + "\n".join(f"- {k}: {v}" for k, v in visualization_paths.items()),
        Colors.GREEN,
    )

    return {
        "data_shape": train_data.shape,
        "iteration_plans": iteration_plans,
        "all_forecasts": all_forecasts,
        "iteration_evaluations": iteration_evaluations,
        "final_evaluation": final_evaluation,
        "interpretation": interpretation,
        "best_forecast": best_forecast,
        "visualization_paths": visualization_paths,
    }


def main():
    print_section("Financial Forecasting Pipeline", "Starting pipeline execution...", Colors.CYAN)

    # Example usage with CSV
    # DATA_PATH = Path(__file__).parent.parent / "src/config/input/sample_data.csv"
    # BUSINESS_CONTEXT = """
    # This data represents buyer-seller transaction GMV (Gross Merchandise Value) over time.
    # We want to forecast future GMV trends for specific buyer-seller pairs.
    # """

    # Run experiments
    # csv_results = asyncio.run(
    #     run_multiple_experiments(
    #         data_source="csv", data_path=DATA_PATH, target_column="buyer_gmv_usd", business_context=BUSINESS_CONTEXT
    #     )
    # )

    gifteval_results = asyncio.run(
        run_multiple_experiments(
            data_source="gifteval",
            business_context="Forecasting time series from GiftEval M4 Daily dataset.",
            n_series=5,
            num_iterations=10,
            target_horizon=0,
        )
    )

    # Print summary for all experiments
    print_section(
        "Multi-Experiment Summary",
        # f"CSV Experiments: {len(csv_results)}\n"
        f"GiftEval Experiments: {len(gifteval_results)}\n"
        # f"Best models CSV: {[r['final_evaluation']['best_model'] for r in csv_results]}\n"
        f"Best models GiftEval: {[r['final_evaluation']['best_model'] for r in gifteval_results]}",
        Colors.CYAN,
    )
