import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from scipy import stats

from ..settings import settings


def load_token_comparison_data() -> pd.DataFrame:
    """Load token comparison data from CSV file."""
    csv_file = settings.output_path / "token_count_comparison" / "token_comparison.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Token comparison CSV not found at: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} records from {csv_file}")
    return df


def save_plot(filename: str) -> None:
    """Save the current matplotlib figure as PNG to the output directory."""
    output_dir = settings.output_path / "token_count_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_file = output_dir / f"{filename}.png"
    plt.savefig(str(plot_file), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_file}")


def plot_tiktoken_vs_gpt4(df: pd.DataFrame) -> None:
    """Create scatter plot with GPT-4 counts on y-axis, tiktoken counts on x-axis."""
    plt.figure(figsize=(10, 8))
    plt.scatter(df['num_tiktoken_tokens'], df['num_gpt_4_tokens'], alpha=0.6, s=20)

    # Add diagonal line for perfect correlation
    max_tokens = max(df['num_tiktoken_tokens'].max(), df['num_gpt_4_tokens'].max())
    plt.plot([0, max_tokens], [0, max_tokens], 'r--', linewidth=2, label='Perfect correlation')

    plt.xlabel('Tiktoken Token Count', fontsize=14)
    plt.ylabel('GPT-4 API Token Count', fontsize=14)
    plt.title('GPT-4 vs Tiktoken Token Counts', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)


def plot_sonnet_vs_tiktoken(df: pd.DataFrame, regression_results: dict = None) -> None:
    """Create scatter plot with Sonnet counts on y-axis, tiktoken counts on x-axis."""
    plt.figure(figsize=(10, 8))
    plt.scatter(df['num_tiktoken_tokens'], df['num_sonnet_tokens'], alpha=0.6, s=20)

    # Add diagonal line for perfect correlation
    max_tokens = max(df['num_tiktoken_tokens'].max(), df['num_sonnet_tokens'].max())
    plt.plot([0, max_tokens], [0, max_tokens], 'r--', linewidth=2, label='Perfect correlation')

    # Add regression line if provided
    if regression_results is not None:
        x_range = np.array([df['num_tiktoken_tokens'].min(), df['num_tiktoken_tokens'].max()])
        y_regression = regression_results['slope'] * x_range + regression_results['intercept']
        plt.plot(x_range, y_regression, 'g-', linewidth=2, label=f'Line of best fit (R² = {regression_results["r_squared"]:.4f})')

    plt.xlabel('Tiktoken Token Count', fontsize=14)
    plt.ylabel('Sonnet API Token Count', fontsize=14)
    plt.title('Sonnet vs Tiktoken Token Counts', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)


def plot_sonnet_vs_gpt4(df: pd.DataFrame) -> None:
    """Create scatter plot with Sonnet counts on y-axis, GPT-4 counts on x-axis."""
    plt.figure(figsize=(10, 8))
    plt.scatter(df['num_gpt_4_tokens'], df['num_sonnet_tokens'], alpha=0.6, s=20)

    # Add diagonal line for perfect correlation
    max_tokens = max(df['num_gpt_4_tokens'].max(), df['num_sonnet_tokens'].max())
    plt.plot([0, max_tokens], [0, max_tokens], 'r--', linewidth=2, label='Perfect correlation')

    plt.xlabel('GPT-4 API Token Count', fontsize=14)
    plt.ylabel('Sonnet API Token Count', fontsize=14)
    plt.title('Sonnet vs GPT-4 Token Counts', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)


def perform_linear_regression_sonnet_tiktoken(df: pd.DataFrame) -> dict:
    """Perform linear regression analysis on sonnet vs tiktoken token counts."""
    # Prepare data
    X = df['num_tiktoken_tokens'].values.reshape(-1, 1)  # Independent variable
    y = df['num_sonnet_tokens'].values  # Dependent variable

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get predictions
    y_pred = model.predict(X)

    # Calculate basic coefficients
    slope = model.coef_[0]
    intercept = model.intercept_

    # Calculate R²
    r_squared = model.score(X, y)

    # Calculate RMSE
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # Calculate confidence intervals using scipy.stats
    n = len(X)

    # Standard errors for confidence intervals
    # Calculate residual sum of squares
    rss = np.sum((y - y_pred) ** 2)
    mse_residual = rss / (n - 2)  # degrees of freedom = n - 2 for simple linear regression

    # Calculate standard error of slope
    x_mean = np.mean(X)
    ss_xx = np.sum((X.flatten() - x_mean) ** 2)
    se_slope = np.sqrt(mse_residual / ss_xx)

    # Calculate standard error of intercept
    se_intercept = np.sqrt(mse_residual * (1/n + x_mean**2/ss_xx))

    # Calculate confidence intervals
    # 95% CI (alpha = 0.05, two-tailed)
    t_95 = stats.t.ppf(0.975, n - 2)  # 97.5th percentile for 95% CI
    slope_95_ci = (slope - t_95 * se_slope, slope + t_95 * se_slope)
    intercept_95_ci = (intercept - t_95 * se_intercept, intercept + t_95 * se_intercept)

    # 67% CI (alpha = 0.33, two-tailed)
    t_67 = stats.t.ppf(0.835, n - 2)  # 83.5th percentile for 67% CI
    slope_67_ci = (slope - t_67 * se_slope, slope + t_67 * se_slope)
    intercept_67_ci = (intercept - t_67 * se_intercept, intercept + t_67 * se_intercept)

    # Package results
    results = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'rmse': rmse,
        'slope_95_ci': slope_95_ci,
        'intercept_95_ci': intercept_95_ci,
        'slope_67_ci': slope_67_ci,
        'intercept_67_ci': intercept_67_ci,
        'slope_sigma': se_slope,
        'intercept_sigma': se_intercept,
        'model': model  # Store model for plotting
    }

    return results


def save_regression_analysis_report(results: dict) -> None:
    """Save detailed regression analysis to text file."""
    output_dir = settings.output_path / "token_count_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "sonnet_tiktoken_regression_analysis.txt"

    with open(report_file, 'w') as f:
        f.write("Sonnet vs Tiktoken Token Counts - Linear Regression Analysis\n")
        f.write("=" * 65 + "\n\n")

        f.write(f"Slope: {results['slope']:.6f}\n")
        f.write(f"Intercept: {results['intercept']:.6f}\n")
        f.write(f"R²: {results['r_squared']:.6f}\n")
        f.write(f"RMSE: {results['rmse']:.6f}\n\n")

        f.write("95% Confidence Intervals:\n")
        f.write(f"  Slope: [{results['slope_95_ci'][0]:.6f}, {results['slope_95_ci'][1]:.6f}]\n")
        f.write(f"  Intercept: [{results['intercept_95_ci'][0]:.6f}, {results['intercept_95_ci'][1]:.6f}]\n\n")

        f.write("67% Confidence Intervals:\n")
        f.write(f"  Slope: [{results['slope_67_ci'][0]:.6f}, {results['slope_67_ci'][1]:.6f}]\n")
        f.write(f"  Intercept: [{results['intercept_67_ci'][0]:.6f}, {results['intercept_67_ci'][1]:.6f}]\n\n")

        f.write("Standard Errors (Sigma):\n")
        f.write(f"  Slope: {results['slope_sigma']:.6f}\n")
        f.write(f"  Intercept: {results['intercept_sigma']:.6f}\n")

    print(f"Regression analysis report saved to: {report_file}")


def save_regression_parameters(results: dict) -> None:
    """Save regression parameters to JSON file for future use."""
    output_dir = settings.output_path / "token_count_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    params_file = output_dir / "sonnet_tiktoken_regression_params.json"

    params = {
        "slope": results['slope'],
        "intercept": results['intercept']
    }

    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Regression parameters saved to: {params_file}")


def run_token_counting_analysis():
    """Run token counting analysis and generate plots."""
    # Check if analysis has already been run
    output_dir = settings.output_path / "token_count_comparison"
    sonnet_tiktoken_plot = output_dir / "sonnet_vs_tiktoken_comparison.png"

    if sonnet_tiktoken_plot.exists():
        print("Token counting analysis already completed (sonnet_vs_tiktoken_comparison.png exists)")
        print("Skipping token counting analysis.")
        return

    print("Starting token counting analysis...")

    # Load data
    df = load_token_comparison_data()

    # Perform linear regression analysis on sonnet vs tiktoken
    print("Performing linear regression analysis...")
    regression_results = perform_linear_regression_sonnet_tiktoken(df)

    # Save regression analysis results
    save_regression_analysis_report(regression_results)
    save_regression_parameters(regression_results)

    # Create all three comparison plots
    plot_tiktoken_vs_gpt4(df)
    save_plot("gpt4_vs_tiktoken_comparison")

    # Create sonnet vs tiktoken plot with regression line
    plot_sonnet_vs_tiktoken(df, regression_results)
    save_plot("sonnet_vs_tiktoken_comparison")

    plot_sonnet_vs_gpt4(df)
    save_plot("sonnet_vs_gpt4_comparison")

    # Print summary statistics
    print(f"Linear Regression Summary:")
    print(f"  Slope: {regression_results['slope']:.6f}")
    print(f"  Intercept: {regression_results['intercept']:.6f}")
    print(f"  R²: {regression_results['r_squared']:.6f}")
    print(f"  RMSE: {regression_results['rmse']:.6f}")
