import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
from scipy import stats

from .settings import settings
from .percentile_sampling import sample_from_percentiles


def load_data():
    """
    Load all scaling data CSV files into separate pandas DataFrames.

    Returns the DataFrames for each data type.
    """
    print("Load input data")
    data_dir = settings.input_data_path

    system_prompt_df = pd.read_csv(data_dir / "system_prompt.csv")
    user_prompt_baseline_df = pd.read_csv(data_dir / "user_prompt_baseline.csv")
    decisions_df = pd.read_csv(data_dir / "decisions.csv")
    tasks_df = pd.read_csv(data_dir / "tasks.csv")
    discussions_df = pd.read_csv(data_dir / "discussions.csv")
    meetings_df = pd.read_csv(data_dir / "meetings.csv")

    print(f"Loaded {len(system_prompt_df)} system prompt records")
    print(f"Loaded {len(user_prompt_baseline_df)} user prompt baseline records")
    print(f"Loaded {len(decisions_df)} decisions")
    print(f"Loaded {len(tasks_df)} tasks")
    print(f"Loaded {len(discussions_df)} discussions")
    print(f"Loaded {len(meetings_df)} meetings")

    return (
        system_prompt_df,
        user_prompt_baseline_df,
        decisions_df,
        tasks_df,
        discussions_df,
        meetings_df
    )


def filter_data_by_date_range(decisions_df, tasks_df, discussions_df, meetings_df):
    """
    Filter dataframes to only include records with created_at between March 1, 2025 and July 31, 2025.
    """
    print("Filtering data by date range")

    start_date = datetime(2025, 3, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 7, 31, 23, 59, 59, tzinfo=timezone.utc)

    # Convert created_at columns to datetime
    decisions_df['created_at'] = pd.to_datetime(decisions_df['created_at'])
    tasks_df['created_at'] = pd.to_datetime(tasks_df['created_at'])
    discussions_df['created_at'] = pd.to_datetime(discussions_df['created_at'])
    meetings_df['created_at'] = pd.to_datetime(meetings_df['created_at'])

    # Filter each dataframe
    decisions_filtered = decisions_df[
        (decisions_df['created_at'] >= start_date) &
        (decisions_df['created_at'] <= end_date)
    ]

    tasks_filtered = tasks_df[
        (tasks_df['created_at'] >= start_date) &
        (tasks_df['created_at'] <= end_date)
    ]

    discussions_filtered = discussions_df[
        (discussions_df['created_at'] >= start_date) &
        (discussions_df['created_at'] <= end_date)
    ]

    meetings_filtered = meetings_df[
        (meetings_df['created_at'] >= start_date) &
        (meetings_df['created_at'] <= end_date)
    ]

    print(f"Filtered decisions: {len(decisions_df)} -> {len(decisions_filtered)}")
    print(f"Filtered tasks: {len(tasks_df)} -> {len(tasks_filtered)}")
    print(f"Filtered discussions: {len(discussions_df)} -> {len(discussions_filtered)}")
    print(f"Filtered meetings: {len(meetings_df)} -> {len(meetings_filtered)}")

    return decisions_filtered, tasks_filtered, discussions_filtered, meetings_filtered


def transform_openai_to_sonnet_tokens(num_openai_tokens, slope, intercept):
    """Transform OpenAI token counts to Sonnet token counts using linear regression parameters."""
    if pd.isna(num_openai_tokens):
        return pd.NA
    return slope * num_openai_tokens + intercept


def transform_tokens_to_sonnet(system_prompt_df, user_prompt_baseline_df, decisions_df, tasks_df, discussions_df, meetings_df):
    """
    Transform OpenAI token counts to Sonnet token counts using regression parameters.

    Returns updated dataframes with num_sonnet_tokens column added where num_open_ai_tokens exists.
    """
    print("Transforming OpenAI tokens to Sonnet tokens...")

    # Load regression parameters
    params_file = settings.output_path / "token_count_comparison" / "sonnet_tiktoken_regression_params.json"

    if not params_file.exists():
        raise FileNotFoundError(f"Regression parameters file not found at: {params_file}. Run token counting analysis first.")

    with open(params_file, 'r') as f:
        params = json.load(f)

    slope, intercept = params['slope'], params['intercept']
    print(f"Loaded regression parameters: slope={slope:.6f}, intercept={intercept:.6f}")

    # Create dictionary of all dataframes for processing
    dataframes_dict = {
        'system_prompt_df': system_prompt_df,
        'user_prompt_baseline_df': user_prompt_baseline_df,
        'decisions_df': decisions_df,
        'tasks_df': tasks_df,
        'discussions_df': discussions_df,
        'meetings_df': meetings_df
    }

    # Transform each dataframe
    updated_dfs = {}
    for name, df in dataframes_dict.items():
        if 'num_open_ai_tokens' in df.columns:
            df_copy = df.copy()
            df_copy['num_sonnet_tokens'] = df_copy['num_open_ai_tokens'].apply(
                lambda x: transform_openai_to_sonnet_tokens(x, slope, intercept)
            )
            print(f"Added num_sonnet_tokens column to {name} (transformed {len(df_copy)} rows)")
            updated_dfs[name] = df_copy
        else:
            print(f"No num_open_ai_tokens column found in {name}, skipping transformation")
            updated_dfs[name] = df

    return (
        updated_dfs['system_prompt_df'],
        updated_dfs['user_prompt_baseline_df'],
        updated_dfs['decisions_df'],
        updated_dfs['tasks_df'],
        updated_dfs['discussions_df'],
        updated_dfs['meetings_df']
    )


def analyze_and_plot_prompts(system_prompt_df, user_prompt_baseline_df):
    """
    Generate a stacked bar plot showing system and user prompt Sonnet tokens,
    plus the leftover tokens from 200,000 total.
    """
    print("Analyzing and plotting prompt token usage...")

    # Get token counts
    system_tokens = system_prompt_df['num_sonnet_tokens'].iloc[0] if len(system_prompt_df) > 0 else 0
    user_tokens = user_prompt_baseline_df['num_sonnet_tokens'].iloc[0] if len(user_prompt_baseline_df) > 0 else 0

    total_limit = 200000
    used_tokens = system_tokens + user_tokens
    leftover_tokens = total_limit - used_tokens

    print(f"System prompt tokens: {system_tokens:,.0f}")
    print(f"User prompt tokens: {user_tokens:,.0f}")
    print(f"Total used tokens: {used_tokens:,.0f}")
    print(f"Leftover tokens: {leftover_tokens:,.0f}")

    # Create stacked bar plot
    plt.figure(figsize=(10, 8))

    categories = ['Token Usage']
    system_values = [system_tokens]
    user_values = [user_tokens]
    leftover_values = [leftover_tokens]

    # Create stacked bars
    bar_width = 0.6
    plt.bar(categories, system_values, bar_width, label=f'System Prompt ({system_tokens:,.0f})', color='#2E86C1')
    plt.bar(categories, user_values, bar_width, bottom=system_values, label=f'User Prompt ({user_tokens:,.0f})', color='#F39C12')
    plt.bar(categories, leftover_values, bar_width, bottom=[system_tokens + user_tokens],
            label=f'Leftover ({leftover_tokens:,.0f})', color='#58D68D')

    # Add horizontal line at 200k limit
    plt.axhline(y=total_limit, color='red', linestyle='--', linewidth=2, label='200K Token Limit')

    # Formatting
    plt.ylabel('Number of Tokens', fontsize=16)
    plt.title('Sonnet Token Usage: System + User Prompts vs Available Capacity', fontsize=18)
    plt.legend(fontsize=14, loc='center right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Format y-axis to show values in thousands
    plt.ticklabel_format(style='plain', axis='y')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

    # Set y-axis limit to show full 200k range
    plt.ylim(0, total_limit * 1.05)

    # Save plot
    output_dir = settings.output_path / "app_data_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_file = output_dir / "prompt_token_usage_analysis.png"
    plt.savefig(str(plot_file), dpi=300, bbox_inches='tight')
    plt.close()

    # Save token counts data to JSON
    token_data = {
        "system_prompt_tokens": int(system_tokens),
        "user_prompt_tokens": int(user_tokens),
        "total_used_tokens": int(used_tokens),
        "leftover_tokens": int(leftover_tokens),
        "token_limit": total_limit
    }

    data_file = output_dir / "prompt_token_counts.json"
    with open(data_file, 'w') as f:
        json.dump(token_data, f, indent=2)

    print(f"Prompt analysis plot saved to: {plot_file}")
    print(f"Token counts data saved to: {data_file}")


def calculate_token_statistics(token_counts):
    """Calculate comprehensive token statistics including percentiles."""
    if len(token_counts) == 0:
        return None

    median_tokens = token_counts.median()
    p16 = np.percentile(token_counts, 16)
    p84 = np.percentile(token_counts, 84)

    # Calculate comprehensive percentiles for JSON export
    percentile_values = [1, 10, 16, 20, 30, 40, 50, 60, 70, 80, 84, 90, 99]
    percentiles = {f"p{p}": np.percentile(token_counts, p) for p in percentile_values}

    return {
        'median': median_tokens,
        'p16': p16,
        'p84': p84,
        'percentiles': percentiles,
        'x_min': 0,
        'x_max': np.percentile(token_counts, 99)
    }


def create_histogram_panel(ax, token_counts, config, stats):
    """Create histogram with percentile lines and styling."""
    # Create histogram
    n, bins, patches = ax.hist(token_counts, bins=config['bins'], alpha=0.7,
                              color=config['hist_color'], edgecolor='black', density=True)

    # Set axis limits to focus on main distribution
    ax.set_xlim(stats['x_min'], stats['x_max'])

    # Add percentile-based distribution sampling overlay
    try:
        sampled_values = sample_from_percentiles(
            stats['percentiles'],
            n_samples=100000
        )

        # Create histogram from sampled values for density calculation
        # Use same number of bins as the original histogram
        hist_counts, bin_edges = np.histogram(
            sampled_values,
            bins=config['bins'],  # Match the original histogram bins
            range=(stats['x_min'], stats['x_max']),
            density=True
        )

        # Calculate bin centers for x-coordinates
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.plot(bin_centers, hist_counts, color='red', linewidth=2,
                label='Percentile-based sampling', alpha=0.8)
    except Exception as e:
        print(f"Warning: Could not create percentile sampling overlay: {e}")

    # Add vertical lines
    ax.axvline(stats['median'], color='green', linestyle='-', linewidth=2,
               label=f'Median ({stats["median"]:.0f})')
    ax.axvline(stats['p16'], color='purple', linestyle='-.', linewidth=2,
               label=f'16th percentile ({stats["p16"]:.0f})')
    ax.axvline(stats['p84'], color='purple', linestyle='-.', linewidth=2,
               label=f'84th percentile ({stats["p84"]:.0f})')

    ax.set_xlabel('Sonnet Token Count', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title(config['hist_title'], fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)


def create_monthly_bar_chart(ax, df, chart_type, config):
    """Create monthly bar charts for counts or token sums."""
    if chart_type == 'counts':
        monthly_data = df.groupby('month').size()
        color = config['count_color']
        ylabel = config['count_ylabel']
        title = config['count_title']
        format_func = lambda x: f'{x}'
    else:  # token_sums
        monthly_data = df.groupby('month')['num_sonnet_tokens'].sum()
        color = config['token_color']
        ylabel = config['token_ylabel']
        title = config['token_title']
        format_func = lambda x: f'{x/1000:.1f}K'

    avg_value = monthly_data.mean()
    months_str = monthly_data.index.astype(str)
    bars = ax.bar(months_str, monthly_data.values, color=color, alpha=0.7, edgecolor='black')
    avg_label = f'Average ({avg_value:.1f})' if chart_type == 'counts' else f'Average ({avg_value:.0f})'
    ax.axhline(avg_value, color='blue' if chart_type == 'counts' else 'orange',
               linestyle='--', linewidth=2, label=avg_label)

    # Add value labels inside bars
    for bar, value in zip(bars, monthly_data.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2, format_func(value),
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='x', rotation=45)

    if chart_type == 'token_sums':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

    return monthly_data, avg_value


def create_statistics_dict(monthly_counts, avg_monthly_count, monthly_tokens, avg_monthly_tokens, token_stats, data_type):
    """Create standardized statistics dictionary."""
    return {
        "monthly_averages": {
            f"avg_{data_type}_per_month": float(avg_monthly_count),
            "avg_sonnet_tokens_per_month": float(avg_monthly_tokens)
        },
        "token_distribution": {
            "median": float(token_stats['median']),
            "percentile_16th": float(token_stats['p16']),
            "percentile_84th": float(token_stats['p84']),
            "percentiles": {k: float(v) for k, v in token_stats['percentiles'].items()}
        }
    }


def save_analysis_results(stats, plot_filename, stats_filename, data_type):
    """Save plots and statistics to files."""
    output_dir = settings.output_path / "app_data_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save statistics to JSON
    stats_file = output_dir / stats_filename
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Adjust layout and save plot
    plt.tight_layout()

    plot_file = output_dir / plot_filename
    plt.savefig(str(plot_file), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{data_type.capitalize()} analysis plot saved to: {plot_file}")
    print(f"{data_type.capitalize()} statistics saved to: {stats_file}")


def get_analysis_config(data_type):
    """Get configuration for different analysis types."""
    configs = {
        'decisions': {
            'bins': 30,
            'hist_color': '#3498DB',
            'count_color': '#E74C3C',
            'token_color': '#9B59B6',
            'hist_title': 'Distribution of Decision Sonnet Token Counts',
            'count_ylabel': 'Number of Decisions',
            'count_title': 'Count of Decisions per Month',
            'token_ylabel': 'Total Sonnet Tokens',
            'token_title': 'Sum of Sonnet Tokens per Month',
            'plot_filename': 'decisions_analysis.png',
            'stats_filename': 'decision_statistics.json'
        },
        'tasks': {
            'bins': 250,
            'hist_color': '#2ECC71',
            'count_color': '#F39C12',
            'token_color': '#1ABC9C',
            'hist_title': 'Distribution of Task Sonnet Token Counts',
            'count_ylabel': 'Number of Tasks',
            'count_title': 'Count of Tasks per Month',
            'token_ylabel': 'Total Sonnet Tokens',
            'token_title': 'Sum of Sonnet Tokens per Month',
            'plot_filename': 'tasks_analysis.png',
            'stats_filename': 'task_statistics.json'
        },
        'discussions': {
            'bins': 30,
            'hist_color': '#E67E22',
            'count_color': '#8E44AD',
            'token_color': '#D35400',
            'hist_title': 'Distribution of Discussion Sonnet Token Counts',
            'count_ylabel': 'Number of Discussions',
            'count_title': 'Count of Discussions per Month',
            'token_ylabel': 'Total Sonnet Tokens',
            'token_title': 'Sum of Sonnet Tokens per Month',
            'plot_filename': 'discussions_analysis.png',
            'stats_filename': 'discussion_statistics.json'
        },
        'meetings': {
            'bins': 30,
            'hist_color': '#16A085',
            'count_color': '#2980B9',
            'token_color': '#27AE60',
            'hist_title': 'Distribution of Meeting Sonnet Token Counts',
            'count_ylabel': 'Number of Meetings',
            'count_title': 'Count of Meetings per Month',
            'token_ylabel': 'Total Sonnet Tokens',
            'token_title': 'Sum of Sonnet Tokens per Month',
            'plot_filename': 'meetings_analysis.png',
            'stats_filename': 'meeting_statistics.json'
        }
    }
    return configs[data_type]


def analyze_and_plot_data_generic(df, data_type):
    """Generic analysis function that uses all helper functions."""
    print(f"Analyzing and plotting {data_type} data...")

    # Ensure we have the necessary columns and data
    if len(df) == 0:
        print(f"No {data_type} data available for analysis")
        return

    if 'num_sonnet_tokens' not in df.columns:
        print(f"No num_sonnet_tokens column found in {data_type} data")
        return

    # Convert created_at to datetime if not already
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['month'] = df['created_at'].dt.to_period('M')

    # Get configuration
    config = get_analysis_config(data_type)

    # Calculate token statistics
    token_counts = df['num_sonnet_tokens'].dropna()
    token_stats = calculate_token_statistics(token_counts)
    if token_stats is None:
        return

    print(f"{data_type.capitalize()} tokens - Median: {token_stats['median']:.0f}")
    print(f"Percentiles - 16th: {token_stats['p16']:.0f}, 84th: {token_stats['p84']:.0f}")

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Panel 1: Histogram
    create_histogram_panel(ax1, token_counts, config, token_stats)

    # Panel 2: Monthly counts
    monthly_counts, avg_monthly_count = create_monthly_bar_chart(ax2, df, 'counts', config)
    print(f"Monthly {data_type} count - Average: {avg_monthly_count:.1f}")

    # Panel 3: Monthly token sums
    monthly_tokens, avg_monthly_tokens = create_monthly_bar_chart(ax3, df, 'token_sums', config)
    print(f"Monthly token sum - Average: {avg_monthly_tokens:.0f}")

    # Create statistics dictionary
    stats = create_statistics_dict(monthly_counts, avg_monthly_count, monthly_tokens,
                                  avg_monthly_tokens, token_stats, data_type)

    # Save results
    save_analysis_results(stats, config['plot_filename'], config['stats_filename'], data_type)


async def input_data_analysis():
    """
    Analyze input data.

    We want to:
    - Look at distributions from the input data
    - Fit models to the data
    """
    # Check if analysis has already been completed
    output_dir = settings.output_path / "app_data_analysis"
    if output_dir.exists():
        print("Input data analysis already completed (app_data_analysis directory exists)")
        print("Skipping input data analysis.")
        return

    print("Starting input data analysis...")

    # load all data
    (
        system_prompt_df,
        user_prompt_baseline_df,
        decisions_df,
        tasks_df,
        discussions_df,
        meetings_df
    ) = load_data()

    # filter data to date range
    decisions_df, tasks_df, discussions_df, meetings_df = filter_data_by_date_range(
        decisions_df, tasks_df, discussions_df, meetings_df
    )

    # transform OpenAI tokens to Sonnet tokens
    system_prompt_df, user_prompt_baseline_df, decisions_df, tasks_df, discussions_df, meetings_df = transform_tokens_to_sonnet(
        system_prompt_df, user_prompt_baseline_df, decisions_df, tasks_df, discussions_df, meetings_df
    )

    # Analyze and plot prompt token usage
    analyze_and_plot_prompts(system_prompt_df, user_prompt_baseline_df)

    # Analyze and plot decisions data
    analyze_and_plot_data_generic(decisions_df, 'decisions')

    # Analyze and plot tasks data
    analyze_and_plot_data_generic(tasks_df, 'tasks')

    # Analyze and plot discussions data
    analyze_and_plot_data_generic(discussions_df, 'discussions')

    # Analyze and plot meetings data
    analyze_and_plot_data_generic(meetings_df, 'meetings')
