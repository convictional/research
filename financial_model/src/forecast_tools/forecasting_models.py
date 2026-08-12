import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotext as plt
import matplotlib.pyplot as mplt
from pathlib import Path

from ..helpers.print_section import print_section, Colors


class ForecastResult(BaseModel):
    forecast: List[float]
    confidence_interval: List[tuple]
    method_used: str
    explanation: str
    config_reasoning: str = ""


def run_prophet(
    train_data: pd.DataFrame, params: Dict[str, Any], forecast_periods: int, target_column: str
) -> ForecastResult:
    """Run Prophet forecasting model"""
    print_section("Running Prophet Model", params, Colors.BLUE)

    # Prepare data for Prophet
    df = train_data.reset_index()
    df = df.rename(columns={"date_id": "ds", target_column: "y"})

    # Get parameters from the plan, with defaults
    prophet_params = {
        "seasonality_mode": params.get("seasonality_mode", "additive"),
        "yearly_seasonality": params.get("yearly_seasonality", "auto"),
        "weekly_seasonality": params.get("weekly_seasonality", "auto"),
        "daily_seasonality": params.get("daily_seasonality", "auto"),
        "changepoint_prior_scale": params.get("changepoint_prior_scale", 0.05),
        "seasonality_prior_scale": params.get("seasonality_prior_scale", 10.0),
        "holidays_prior_scale": params.get("holidays_prior_scale", 10.0),
        "changepoint_range": params.get("changepoint_range", 0.8),
    }

    # Initialize Prophet with dynamic parameters
    model = Prophet(**{k: v for k, v in prophet_params.items() if v is not None})

    # Add any additional seasonalities if specified
    if params.get("add_seasonalities"):
        for seasonality in params["add_seasonalities"]:
            model.add_seasonality(**seasonality)

    # Add any holidays if specified
    if params.get("holidays"):
        model.add_country_holidays(country_name=params["holidays"])

    # Fit the model
    model.fit(df)

    # Make future dataframe for predictions
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)

    # Extract forecasted values beyond the training data
    forecast_values = forecast[len(df) :]  # Skip the training data predictions
    forecast_yhat = forecast_values["yhat"].tolist()
    forecast_ci = list(zip(forecast_values["yhat_lower"].tolist(), forecast_values["yhat_upper"].tolist()))

    print_section(
        "Prophet Results",
        f"Forecast Length: {len(forecast_yhat)}\nFirst few values: {forecast_yhat[:5]}",
        Colors.BLUE,
    )

    forecast_result = ForecastResult(
        forecast=forecast_yhat,
        confidence_interval=forecast_ci,
        method_used="prophet",
        explanation=f"Forecast generated using Prophet with {prophet_params['seasonality_mode']} seasonality",
    )

    return forecast_result


def run_moving_average(
    train_data: pd.DataFrame, params: Dict[str, Any], forecast_periods: int, target_column: str
) -> ForecastResult:
    """Run Holt-Winters exponential smoothing with trend and seasonality"""
    print_section("Running Holt-Winters Exponential Smoothing", params, Colors.YELLOW)

    # Get the series and ensure it has proper frequency
    series = train_data[target_column].copy()

    # Validate and process window parameter
    window = params.get("window", 7)
    try:
        # Convert to integer, rounding if necessary
        window = int(round(window))
        if window < 1:
            print(f"Warning: Invalid window size {window}, using default value of 7")
            window = 7
    except (ValueError, TypeError):
        print(f"Warning: Invalid window value {window}, using default value of 7")
        window = 7

    # Update params with validated window
    params["window"] = window

    # Ensure datetime index and daily frequency
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    # Validate that data is daily
    if series.index.to_series().diff().median().days != 1:
        raise ValueError("Input data must be daily frequency")

    # Set daily frequency explicitly
    series = series.asfreq("D")

    # Fill any gaps that might exist in the daily data
    series = series.ffill()

    # Validate parameters
    valid_trends = ["add", "mul", None]
    valid_seasonals = ["add", "mul", None]

    trend = params.get("trend", "add")
    seasonal = params.get("seasonal", "add")

    if trend not in valid_trends:
        print(f"Warning: Invalid trend type '{trend}'. Using 'add' instead.")
        trend = "add"
    if seasonal not in valid_seasonals:
        print(f"Warning: Invalid seasonal type '{seasonal}'. Using 'add' instead.")
        seasonal = "add"

    # Check if using multiplicative components
    is_multiplicative = trend == "mul" or seasonal == "mul"

    try:
        if is_multiplicative:
            # For multiplicative components, ensure strictly positive values
            if (series <= 0).any():
                print("Warning: Data contains non-positive values, switching to additive components")
                trend = "add"
                seasonal = "add"
                is_multiplicative = False
            else:
                # If using multiplicative components and all values are positive,
                # scale to preserve positivity using log transform
                scaled_series = np.log1p(series)
        else:
            # For additive components, use standard scaling
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            scaled_series = pd.Series(scaled_data, index=series.index)

        # Fit Holt-Winters model
        model = ExponentialSmoothing(
            scaled_series,
            seasonal_periods=params.get("window", 7),
            trend=trend,
            seasonal=seasonal,
            initialization_method="estimated",
        ).fit(use_boxcox=None)

        # Get future forecast
        scaled_forecast = model.forecast(forecast_periods)

        # Inverse transform the forecasted values
        if is_multiplicative:
            forecast_values = np.expm1(scaled_forecast)
        else:
            forecast_values = scaler.inverse_transform(scaled_forecast.values.reshape(-1, 1)).flatten()

        # Calculate confidence intervals
        residuals = model.resid
        if is_multiplicative:
            # For multiplicative models, work with relative errors
            relative_error = np.std(np.exp(residuals) - 1)
            conf_int = [(f / (1 + 1.96 * relative_error), f * (1 + 1.96 * relative_error)) for f in forecast_values]
        else:
            # For additive models, use standard approach
            std_resid = np.std(residuals) * (series.max() - series.min())
            conf_int = [(f - 1.96 * std_resid, f + 1.96 * std_resid) for f in forecast_values]

    except Exception as e:
        print(f"Warning: Error in Holt-Winters calculation: {str(e)}")
        print("Falling back to simple moving average")

        # Fallback to simple moving average
        window = params.get("window", 7)
        ma = series.rolling(window=window, min_periods=1).mean()
        last_value = ma.iloc[-1]
        forecast_values = [last_value] * forecast_periods
        std_resid = series.std()
        conf_int = [(f - 1.96 * std_resid, f + 1.96 * std_resid) for f in forecast_values]

    print_section(
        "Moving Average Results",
        f"Forecast Length: {len(forecast_values)}\nFuture values: {forecast_values[:5]}",
        Colors.YELLOW,
    )

    forecast_result = ForecastResult(
        forecast=forecast_values.tolist() if isinstance(forecast_values, np.ndarray) else forecast_values,
        confidence_interval=conf_int,
        method_used="exponential_smoothing",
        explanation=f"Forecast using Holt-Winters exponential smoothing with {trend} trend and {seasonal} seasonality",
    )

    return forecast_result


def run_regression(
    train_data: pd.DataFrame, params: Dict[str, Any], forecast_periods: int, target_column: str
) -> ForecastResult:
    """Run linear regression with time-based features"""
    print_section("Running Regression Model", params, Colors.RED)

    df = train_data.copy()

    # Create time-based features
    df["trend"] = np.arange(len(df))
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["week_of_year"] = df.index.isocalendar().week
    df["quarter"] = df.index.quarter

    # Define available features and ensure uniqueness
    available_features = ["trend", "month", "day_of_week", "week_of_year", "quarter"]

    # Use only unique features from the requested list
    requested_features = params.get("features", ["trend", "month", "day_of_week"])
    features = list(dict.fromkeys([f for f in requested_features if f in available_features]))

    # If no valid features, use default features
    if not features:
        features = ["trend", "month", "day_of_week"]

    # Prepare features
    X = df[features]
    y = df[target_column]

    # Scale features while preserving column names
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Get historical predictions
    historical_forecast = model.predict(X_scaled)

    # Prepare future features for prediction
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq="D")

    # Create future features DataFrame with same column names
    future_data = pd.DataFrame(index=future_dates)

    # Add features one by one to ensure proper length
    last_trend = len(df)
    for feature in features:
        if feature == "trend":
            future_data[feature] = np.arange(last_trend + 1, last_trend + forecast_periods + 1)
        elif feature == "month":
            future_data[feature] = future_dates.month
        elif feature == "day_of_week":
            future_data[feature] = future_dates.dayofweek
        elif feature == "week_of_year":
            future_data[feature] = future_dates.isocalendar().week
        elif feature == "quarter":
            future_data[feature] = future_dates.quarter

    # Rest of the function remains unchanged...
    future_X_scaled = pd.DataFrame(scaler.transform(future_data), columns=future_data.columns, index=future_data.index)

    forecast = model.predict(future_X_scaled)
    forecast_values = forecast.tolist()

    # Calculate confidence intervals using prediction std
    std_dev = np.std(y - historical_forecast)
    conf_int = [(f - 1.96 * std_dev, f + 1.96 * std_dev) for f in forecast_values]

    print_section(
        "Regression Results",
        f"Forecast Length: {len(forecast_values)}\nFuture values: {forecast_values[:5]}\nFeatures: {features}",
        Colors.RED,
    )

    forecast_result = ForecastResult(
        forecast=forecast_values,
        confidence_interval=conf_int,
        method_used="regression",
        explanation=f"Forecast using linear regression with features: {', '.join(features)}",
    )

    return forecast_result


def plot_forecast(
    forecast: ForecastResult,
    validation_data: pd.Series,
    method_name: str,
    config_idx: int,
    experiment_id: str = None,  # Add experiment_id parameter
) -> None:
    """Plot forecast results with confidence intervals"""
    plot_dir = Path(__file__).parent.parent.parent / "src/config/output/plots"
    plot_dir.mkdir(exist_ok=True)

    # Terminal plot using plotext with more compact settings
    plt.clear_data()
    plt.plotsize(70, 15)

    # Get validation period and future period lengths
    val_len = len(validation_data)
    full_forecast = np.array(forecast.forecast)

    # Split forecast into validation period and future period
    validation_forecast = full_forecast[:val_len]
    future_forecast = full_forecast[val_len:]

    # For terminal plot, reduce points if needed
    step = max(1, val_len // 50)  # Show max ~50 points in terminal
    val_data = validation_data.values[::step]
    val_forecast = validation_forecast[::step]
    x_indices = list(range(len(val_data)))

    # Plot validation data and forecast using numerical x-axis
    plt.scatter(x_indices, val_data, label="Actual", color="red")
    plt.plot(x_indices, val_forecast, label="Historical Forecast", color="blue")

    # Plot future forecast
    if len(future_forecast) > 0:
        future_x = list(range(len(x_indices), len(x_indices) + len(future_forecast)))
        plt.plot(future_x, future_forecast, label="Future Forecast", color="green")

    plt.title(f"{method_name} (Config {config_idx})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

    # Add a small delay to ensure terminal display
    import time

    time.sleep(0.5)

    # Save matplotlib plot to file
    fig, ax = mplt.subplots(figsize=(12, 6))

    # Plot actual values
    ax.plot(validation_data.index, validation_data.values, label="Actual", color="red")

    # Plot historical forecast
    ax.plot(validation_data.index, validation_forecast, label="Historical Forecast", color="blue")

    # Plot future forecast
    if len(future_forecast) > 0:
        freq = pd.infer_freq(validation_data.index) or "D"
        future_dates = pd.date_range(
            start=validation_data.index[-1] + pd.Timedelta(1, unit=freq), periods=len(future_forecast), freq=freq
        )
        ax.plot(future_dates, future_forecast, label="Future Forecast", color="green", linestyle="--")

        # Add confidence intervals for future forecast
        ci_lower = [ci[0] for ci in forecast.confidence_interval[val_len:]]
        ci_upper = [ci[1] for ci in forecast.confidence_interval[val_len:]]
        ax.fill_between(future_dates, ci_lower, ci_upper, color="blue", alpha=0.1)

    ax.set_title(f"{method_name} (Config {config_idx}) Forecast vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    # Update filename to include experiment_id
    iteration_num = forecast.iteration if hasattr(forecast, "iteration") else 0
    plot_idx = (
        iteration_num * 12 + (["prophet", "exponential_smoothing", "regression"].index(method_name) * 4) + config_idx
    )

    # Use experiment_id in filename if provided
    filename = (
        f"{experiment_id}_forecast_plot_{plot_idx:03d}.jpg" if experiment_id else f"forecast_plot_{plot_idx:03d}.jpg"
    )
    fig.savefig(plot_dir / filename)
    mplt.close(fig)
