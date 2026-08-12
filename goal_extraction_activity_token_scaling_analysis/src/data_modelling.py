import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List

from .settings import settings
from .percentile_sampling import sample_from_percentiles

# Monte Carlo simulation parameters
MONTE_CARLO_ITERATIONS = 10000

# Common visualization constants
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
PLOT_FIGURE_SIZE = (12, 8)
PLOT_DPI = 200
PLOT_FONT_SIZE_TITLE = 18
PLOT_FONT_SIZE_AXIS = 16
PLOT_FONT_SIZE_LEGEND = 14
PLOT_FONT_SIZE_TICK = 14


def load_available_context_limit() -> int:
    """Load available context limit from prompt token counts JSON file."""
    prompt_tokens_file = settings.output_path / "app_data_analysis" / "prompt_token_counts.json"
    with open(prompt_tokens_file, 'r') as f:
        prompt_data = json.load(f)
    return prompt_data['leftover_tokens']


def extract_uncertainty_data_arrays(token_uncertainty_data: Dict, alpha: float, org_sizes: List[int], divide_by_context: bool = False) -> tuple:
    """Extract median, p16, p84 arrays for a given alpha value."""
    leftover_tokens = load_available_context_limit() if divide_by_context else 1

    medians = []
    p16_values = []
    p84_values = []

    for org_size in org_sizes:
        data = token_uncertainty_data[alpha][org_size]
        medians.append(data['median'] / leftover_tokens)
        p16_values.append(data['p16'] / leftover_tokens)
        p84_values.append(data['p84'] / leftover_tokens)

    return np.array(medians), np.array(p16_values), np.array(p84_values)


def calculate_error_bars(medians: np.ndarray, p16_values: np.ndarray, p84_values: np.ndarray) -> tuple:
    """Calculate error bar distances from median to percentiles."""
    lower_errors = medians - p16_values
    upper_errors = p84_values - medians
    return lower_errors, upper_errors


def apply_common_plot_styling(ax, xlabel: str, ylabel: str, title: str, legend_loc: str = 'upper left'):
    """Apply common styling to matplotlib axes."""
    ax.set_xlabel(xlabel, fontsize=PLOT_FONT_SIZE_AXIS)
    ax.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZE_AXIS)
    ax.set_title(title, fontsize=PLOT_FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=PLOT_FONT_SIZE_LEGEND, loc=legend_loc)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE_TICK)


def save_plot(filename: str, description: str):
    """Save plot with consistent settings and print confirmation."""
    output_dir = settings.output_path / "data_modelling"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_file = output_dir / filename
    plt.savefig(str(plot_file), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    print(f"{description} saved to: {plot_file}")


def create_uncertainty_visualization_base(
    token_uncertainty_data: Dict,
    org_sizes: List[int],
    alpha_values: List[float],
    config: Dict
) -> None:
    """Base function for creating uncertainty visualization plots with error bars."""

    # Set up single plot
    plt.figure(figsize=PLOT_FIGURE_SIZE)

    for i, alpha in enumerate(alpha_values):
        # Extract data arrays using shared utility
        medians, p16_values, p84_values = extract_uncertainty_data_arrays(
            token_uncertainty_data, alpha, org_sizes, config['divide_by_context']
        )

        # Calculate error bars using shared utility
        lower_errors, upper_errors = calculate_error_bars(medians, p16_values, p84_values)

        # Plot with error bars
        plt.errorbar(org_sizes, medians,
                    yerr=[lower_errors, upper_errors],
                    color=PLOT_COLORS[i],
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    capsize=3,
                    capthick=1,
                    label=f'α = {alpha}',
                    alpha=0.8)

    # Add reference line
    if config.get('reference_line_value'):
        plt.axhline(y=config['reference_line_value'],
                   color='red',
                   linestyle='--',
                   linewidth=2,
                   label=config['reference_line_label'],
                   alpha=0.8)

    # Apply styling using shared utility
    apply_common_plot_styling(
        plt.gca(),
        config['xlabel'],
        config['ylabel'],
        config['title'],
        config.get('legend_loc', 'upper left')
    )

    # Apply axis formatting
    ax = plt.gca()
    if config.get('y_formatter'):
        ax.yaxis.set_major_formatter(config['y_formatter'])

    # Set y-axis ticks
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    # Set axis limits
    plt.xlim(config['xlim'])
    plt.ylim(config['ylim'])

    # Save plot using shared utility
    save_plot(config['filename'], config['description'])


def load_baseline_org_size() -> int:
    """Load baseline organization size from convictional_org_size.json."""
    org_size_file = settings.input_data_path / "convictional_org_size.json"

    with open(org_size_file, 'r') as f:
        org_data = json.load(f)

    return org_data['count']


def load_token_distribution_percentiles() -> Dict[str, Dict]:
    """Load token distribution percentiles from input_data_analysis results."""
    statistics_files = {
        'tasks': 'task_statistics.json',
        'discussions': 'discussion_statistics.json',
        'meetings': 'meeting_statistics.json',
        'decisions': 'decision_statistics.json'
    }

    percentiles_data = {}

    for content_type, filename in statistics_files.items():
        file_path = settings.output_path / "app_data_analysis" / filename

        with open(file_path, 'r') as f:
            stats_data = json.load(f)

        # Extract percentiles dictionary
        percentiles_data[content_type] = stats_data['token_distribution']['percentiles']

    return percentiles_data


def load_baseline_monthly_counts() -> Dict[str, float]:
    """Load baseline monthly counts from statistics JSON files."""
    statistics_files = {
        'tasks': 'task_statistics.json',
        'discussions': 'discussion_statistics.json',
        'meetings': 'meeting_statistics.json',
        'decisions': 'decision_statistics.json'
    }

    baseline_counts = {}

    for content_type, filename in statistics_files.items():
        file_path = settings.output_path / "app_data_analysis" / filename

        with open(file_path, 'r') as f:
            stats_data = json.load(f)

        # Extract the appropriate field name for each content type
        field_name = f"avg_{content_type}_per_month"
        baseline_counts[content_type] = stats_data['monthly_averages'][field_name]

        print(f"Loaded baseline {content_type}: {baseline_counts[content_type]:.1f} per month")

    return baseline_counts


def power_law_scaling(y_o: float, x_o: int, x: int, alpha: float) -> float:
    """
    Calculate power law scaling: y = y_o * (x / x_o)^alpha

    Args:
        y_o: Baseline count (at baseline organization size)
        x_o: Baseline organization size
        x: Target organization size
        alpha: Scaling exponent

    Returns:
        Scaled count for target organization size
    """
    return y_o * ((x / x_o) ** alpha)


def sample_content_tokens(percentiles_dict: Dict, count: int) -> float:
    """
    Sample token counts for a specific content type.

    Args:
        percentiles_dict: Dictionary of percentiles for the content type
        count: Number of items to sample (e.g., number of tasks)

    Returns:
        Total token count for this content type
    """
    # Sample individual item token counts
    individual_tokens = sample_from_percentiles(percentiles_dict, n_samples=count)

    # Sum to get total tokens for this content type
    total_tokens = np.sum(individual_tokens)

    return total_tokens


def simulate_total_tokens(scaling_data: Dict, percentiles_data: Dict, org_size: int, alpha: float) -> np.ndarray:
    """
    Run Monte Carlo simulation to get distribution of total token counts.

    Args:
        scaling_data: Scaling data from calculate_scaling_data()
        percentiles_data: Token percentiles from load_token_distribution_percentiles()
        org_size: Organization size to simulate
        alpha: Alpha value for scaling

    Returns:
        Array of total token counts from all iterations
    """
    total_tokens_per_iteration = []
    content_types = ['tasks', 'discussions', 'meetings', 'decisions']

    for _ in range(MONTE_CARLO_ITERATIONS):
        iteration_total = 0

        for content_type in content_types:
            # Get the count for this content type at this org size and alpha
            org_size_index = scaling_data['org_sizes'].index(org_size)
            count = scaling_data['scaling_data'][content_type][alpha][org_size_index]

            # Sample tokens for this content type
            content_tokens = sample_content_tokens(percentiles_data[content_type], int(count))
            iteration_total += content_tokens

        total_tokens_per_iteration.append(iteration_total)

    return np.array(total_tokens_per_iteration)


def calculate_uncertainty_ranges(total_tokens_array: np.ndarray) -> tuple:
    """
    Calculate uncertainty ranges from Monte Carlo simulation results.

    Args:
        total_tokens_array: Array of total token counts from simulation

    Returns:
        Tuple of (p16, median, p84) percentiles
    """
    p16, median, p84 = np.percentile(total_tokens_array, [16, 50, 84])
    return p16, median, p84


def generate_org_size_range() -> List[int]:
    """Generate organization size range from 14 to 200 in increments of ~5."""
    return [14, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]


def calculate_token_scaling_with_uncertainty(scaling_data: Dict, percentiles_data: Dict) -> Dict:
    """
    Generate token scaling data with uncertainty ranges for all org sizes and alphas.

    Args:
        scaling_data: Results from calculate_scaling_data()
        percentiles_data: Results from load_token_distribution_percentiles()

    Returns:
        Dictionary: {alpha: {org_size: {'median': X, 'p16': Y, 'p84': Z}}}
    """
    print("Starting token uncertainty analysis...")

    token_uncertainty_data = {}
    org_sizes = scaling_data['org_sizes']
    alpha_values = scaling_data['alpha_values']

    total_combinations = len(alpha_values) * len(org_sizes)
    current_combination = 0

    for alpha in alpha_values:
        token_uncertainty_data[alpha] = {}

        for org_size in org_sizes:
            current_combination += 1
            print(f"Processing combination {current_combination}/{total_combinations}: α={alpha}, org_size={org_size}")

            # Run Monte Carlo simulation
            total_tokens_array = simulate_total_tokens(scaling_data, percentiles_data, org_size, alpha)

            # Calculate uncertainty ranges
            p16, median, p84 = calculate_uncertainty_ranges(total_tokens_array)

            token_uncertainty_data[alpha][org_size] = {
                'median': median,
                'p16': p16,
                'p84': p84
            }

    print("Token uncertainty analysis completed!")
    return token_uncertainty_data


def calculate_scaling_data() -> Dict:
    """
    Calculate scaled counts for each content type, alpha value, and org size.
    Loads all baseline data and performs power law scaling calculations.

    Returns:
        Dictionary containing:
        - scaling_data: {content_type: {alpha: [counts_for_each_org_size]}}
        - org_sizes: List of organization sizes used
        - alpha_values: List of alpha values used
        - baseline_counts: Original baseline counts per content type
        - baseline_org_size: Baseline organization size (14)
    """
    print("Starting power law scaling analysis...")

    # Load baseline data
    x_o = load_baseline_org_size()
    print(f"Baseline organization size: {x_o} people")

    baseline_counts = load_baseline_monthly_counts()

    # Define analysis parameters
    org_sizes = generate_org_size_range()
    alpha_values = [0.5, 1, 1.05, 1.15, 1.5, 2]

    print(f"Organization size range: {org_sizes[0]} to {org_sizes[-1]} people")
    print(f"Alpha values: {alpha_values}")

    # Calculate scaling data
    scaling_data = {}

    for content_type, y_o in baseline_counts.items():
        scaling_data[content_type] = {}

        for alpha in alpha_values:
            scaling_data[content_type][alpha] = [
                power_law_scaling(y_o, x_o, x, alpha) for x in org_sizes
            ]

    return {
        'scaling_data': scaling_data,
        'org_sizes': org_sizes,
        'alpha_values': alpha_values,
        'baseline_counts': baseline_counts,
        'baseline_org_size': x_o
    }


def create_scaling_visualization(data: Dict) -> None:
    """Create 4-panel matplotlib visualization of power law scaling."""

    # Extract data components
    scaling_data = data['scaling_data']
    org_sizes = data['org_sizes']
    alpha_values = data['alpha_values']

    # Set up 2x2 subplot layout
    _, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    # Content type configuration
    content_config = {
        'tasks': {'title': 'Monthly Task Count Scaling', 'color_base': 'blue'},
        'discussions': {'title': 'Monthly Discussion Count Scaling', 'color_base': 'red'},
        'meetings': {'title': 'Monthly Meeting Count Scaling', 'color_base': 'green'},
        'decisions': {'title': 'Monthly Decision Count Scaling', 'color_base': 'orange'}
    }

    # Color palette for different alpha values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':', '-', '--']

    content_types = ['tasks', 'discussions', 'meetings', 'decisions']

    for i, content_type in enumerate(content_types):
        ax = axes[i]

        # Plot curves for each alpha value
        for j, alpha in enumerate(alpha_values):
            counts = scaling_data[content_type][alpha]

            ax.plot(org_sizes, counts,
                   color=colors[j],
                   linestyle=line_styles[j],
                   linewidth=2,
                   label=f'α = {alpha}',
                   marker='o' if j % 2 == 0 else 's',
                   markersize=4,
                   alpha=0.8)

        # Styling
        ax.set_xlabel('Organization Size (People)', fontsize=16)
        ax.set_ylabel('Monthly Count', fontsize=16)
        ax.set_title(content_config[content_type]['title'], fontsize=18, fontweight='bold')
        ax.legend(fontsize=14, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Set reasonable y-limits with fine-grained ticks
        max_count = max([max(scaling_data[content_type][alpha]) for alpha in alpha_values])
        ax.set_ylim(0, max_count * 1.05)

        # Add more fine-grained y-axis ticks
        if max_count <= 1000:
            # For smaller ranges (decisions, discussions), use steps of 100
            tick_step = 100
        elif max_count <= 5000:
            # For medium ranges (meetings), use steps of 500
            tick_step = 500
        else:
            # For larger ranges (tasks), use steps of 2000
            tick_step = 2000

        y_ticks = np.arange(0, max_count * 1.05 + tick_step, tick_step)
        ax.set_yticks(y_ticks)

    # Overall plot formatting
    plt.tight_layout(pad=2.0)

    # Save plot
    output_dir = settings.output_path / "data_modelling"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_file = output_dir / "power_law_scaling_counts.png"
    plt.savefig(str(plot_file), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Power law scaling analysis plot saved to: {plot_file}")


def create_token_uncertainty_visualization_zoomed(token_uncertainty_data: Dict, org_sizes: List[int], alpha_values: List[float]) -> None:
    """Create zoomed-in visualization of total token scaling with uncertainty bars (250K limit)."""
    leftover_tokens = load_available_context_limit()

    config = {
        'divide_by_context': False,
        'reference_line_value': leftover_tokens,
        'reference_line_label': f'Available Context Limit ({leftover_tokens/1000:.0f}K)',
        'xlabel': 'Organization Size (People)',
        'ylabel': 'Total Monthly Tokens',
        'title': 'Token Scaling with Uncertainty Ranges (Zoomed)',
        'legend_loc': 'lower right',
        'y_formatter': plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'),
        'xlim': (min(org_sizes) - 5, 50),
        'ylim': (100000, 250000),
        'filename': 'token_scaling_uncertainty_zoomed.png',
        'description': 'Zoomed token uncertainty analysis plot'
    }

    create_uncertainty_visualization_base(token_uncertainty_data, org_sizes, alpha_values, config)


def create_token_uncertainty_visualization(token_uncertainty_data: Dict, org_sizes: List[int], alpha_values: List[float]) -> None:
    """Create visualization of total token scaling with uncertainty bars."""
    leftover_tokens = load_available_context_limit()

    config = {
        'divide_by_context': False,
        'reference_line_value': leftover_tokens,
        'reference_line_label': f'Available Context Limit ({leftover_tokens/1000:.0f}K)',
        'xlabel': 'Organization Size (People)',
        'ylabel': 'Total Monthly Tokens',
        'title': 'Token Scaling with Uncertainty Ranges',
        'legend_loc': 'upper left',
        'y_formatter': plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'),
        'xlim': (min(org_sizes) - 5, max(org_sizes) + 5),
        'ylim': (0, None),
        'filename': 'token_scaling_uncertainty.png',
        'description': 'Token uncertainty analysis plot'
    }

    create_uncertainty_visualization_base(token_uncertainty_data, org_sizes, alpha_values, config)


def create_context_windows_visualization(token_uncertainty_data: Dict, org_sizes: List[int], alpha_values: List[float]) -> None:
    """Create visualization of context windows scaling with uncertainty bars."""
    config = {
        'divide_by_context': True,
        'reference_line_value': 1.0,
        'reference_line_label': 'Single Context Window Limit',
        'xlabel': 'Organization Size (People)',
        'ylabel': 'Number of Context Windows',
        'title': 'Context Windows Scaling with Uncertainty Ranges',
        'legend_loc': 'upper left',
        'y_formatter': plt.FuncFormatter(lambda x, _: f'{x:.1f}'),
        'xlim': (min(org_sizes) - 5, max(org_sizes) + 5),
        'ylim': (0, None),
        'filename': 'context_windows_scaling_uncertainty.png',
        'description': 'Context windows uncertainty analysis plot'
    }

    create_uncertainty_visualization_base(token_uncertainty_data, org_sizes, alpha_values, config)


def create_context_windows_visualization_zoomed(token_uncertainty_data: Dict, org_sizes: List[int], alpha_values: List[float]) -> None:
    """Create zoomed-in visualization of context windows scaling with uncertainty bars."""
    config = {
        'divide_by_context': True,
        'reference_line_value': 1.0,
        'reference_line_label': 'Single Context Window Limit',
        'xlabel': 'Organization Size (People)',
        'ylabel': 'Number of Context Windows',
        'title': 'Context Windows Scaling with Uncertainty Ranges (Zoomed)',
        'legend_loc': 'lower right',
        'y_formatter': plt.FuncFormatter(lambda x, _: f'{x:.1f}'),
        'xlim': (min(org_sizes) - 5, 50),
        'ylim': (0.5, 1.5),
        'filename': 'context_windows_scaling_uncertainty_zoomed.png',
        'description': 'Zoomed context windows uncertainty analysis plot'
    }

    create_uncertainty_visualization_base(token_uncertainty_data, org_sizes, alpha_values, config)


def export_token_uncertainty_table(token_uncertainty_data: Dict) -> None:
    """Export context windows (LLM requests) uncertainty data as a formatted text table for presentation."""

    leftover_tokens = load_available_context_limit()

    # Define specific organization sizes to extract
    target_org_sizes = [14, 50, 100, 150, 200]

    # Reverse alpha values order (2.0 down to 0.5)
    alpha_values = [2.0, 1.5, 1.15, 1.05, 1.0, 0.5]

    # Create table headers
    header = ["Alpha"]
    for org_size in target_org_sizes:
        header.extend([f"{org_size}_p16", f"{org_size}_median", f"{org_size}_p84"])

    # Prepare table rows
    table_data = []

    for alpha in alpha_values:
        row = [f"{alpha}"]

        for org_size in target_org_sizes:
            if org_size in token_uncertainty_data[alpha]:
                data = token_uncertainty_data[alpha][org_size]
                # Convert to context windows (number of LLM requests)
                p16 = data['p16'] / leftover_tokens
                median = data['median'] / leftover_tokens
                p84 = data['p84'] / leftover_tokens
                # Format as decimal values with 2 decimal places
                row.extend([f"{p16:.2f}", f"{median:.2f}", f"{p84:.2f}"])
            else:
                # Handle case where org size not found
                row.extend(["N/A", "N/A", "N/A"])

        table_data.append(row)

    # Calculate column widths for proper alignment
    col_widths = []
    all_rows = [header] + table_data

    for col_idx in range(len(header)):
        max_width = max(len(str(row[col_idx])) for row in all_rows)
        col_widths.append(max_width + 2)  # Add padding

    # Create formatted table string
    table_lines = []

    # Add header
    header_line = "".join(cell.ljust(col_widths[i]) for i, cell in enumerate(header))
    table_lines.append(header_line)

    # Add separator line
    separator = "".join("-" * width for width in col_widths)
    table_lines.append(separator)

    # Add data rows
    for row in table_data:
        row_line = "".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        table_lines.append(row_line)

    # Join all lines
    table_text = "\n".join(table_lines)

    # Save to text file
    output_dir = settings.output_path / "data_modelling"
    output_dir.mkdir(parents=True, exist_ok=True)

    table_file = output_dir / "context_windows_uncertainty_table.txt"
    with open(table_file, 'w') as f:
        f.write("Context Windows (LLM Requests) Uncertainty Analysis - Key Organization Sizes\n")
        f.write("=" * 75 + "\n\n")
        f.write("Values represent number of LLM requests needed for monthly activity data\n")
        f.write("p16 = 16th percentile, median = 50th percentile, p84 = 84th percentile\n")
        f.write("Values > 1.0 require multiple LLM API calls\n\n")
        f.write(table_text)
        f.write("\n")

    print(f"Context windows uncertainty table saved to: {table_file}")


def calculate_and_visualize_token_uncertainty(scaling_data: Dict) -> None:
    """
    Calculate token scaling with uncertainty and create visualization.

    Args:
        scaling_data: Results from calculate_scaling_data()
    """
    # Load token distribution percentiles
    percentiles_data = load_token_distribution_percentiles()

    # Calculate token scaling with uncertainty ranges
    token_uncertainty_data = calculate_token_scaling_with_uncertainty(scaling_data, percentiles_data)

    # Create visualization with error bars
    create_token_uncertainty_visualization(token_uncertainty_data, scaling_data['org_sizes'], scaling_data['alpha_values'])

    # Create zoomed visualization with error bars
    create_token_uncertainty_visualization_zoomed(token_uncertainty_data, scaling_data['org_sizes'], scaling_data['alpha_values'])

    # Create context windows visualization with error bars
    create_context_windows_visualization(token_uncertainty_data, scaling_data['org_sizes'], scaling_data['alpha_values'])

    # Create zoomed context windows visualization with error bars
    create_context_windows_visualization_zoomed(token_uncertainty_data, scaling_data['org_sizes'], scaling_data['alpha_values'])

    # Export token uncertainty data as formatted text table
    export_token_uncertainty_table(token_uncertainty_data)


async def run_data_modelling():
    """
    Main function for power law scaling analysis and visualization.

    This function models how organizational activity counts (tasks, discussions,
    meetings, decisions) scale with organization size using power law relationships.
    """
    # Calculate scaling data for all content types and alpha values
    scaling_data = calculate_scaling_data()

    # Create visualization of power law scaling
    create_scaling_visualization(scaling_data)

    # Calculate and visualize token uncertainty
    calculate_and_visualize_token_uncertainty(scaling_data)

    print("Power law scaling analysis completed!")
