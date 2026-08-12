import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from .settings import settings

RATED_STATED_GOALS_PATH = settings.data_path / "rated_individual_goals" / "rated_stated_goals.csv"
RATED_UNSTATED_GOALS_PATH = settings.data_path / "rated_individual_goals" / "rated_unstated_goals.csv"
METRICS_OUTPUT_DIR = settings.data_path / "metrics_analysis"
STATED_GOALS_SUMMARY_FILE = METRICS_OUTPUT_DIR / "stated_goals_summary.txt"
UNSTATED_GOALS_SUMMARY_FILE = METRICS_OUTPUT_DIR / "unstated_goals_summary.txt"
PROCESSED_GOAL_MINING_PATH = settings.data_path / "processed_data" / "processed_goal_mining_dump.csv"

RATER_ANALYSES = [
    ("All Three Raters", ['rater1', 'rater2', 'rater3']),
    ("Rater 1 vs Rater 2", ['rater1', 'rater2']),
    ("Rater 1 vs Rater 3", ['rater1', 'rater3']),
    ("Rater 2 vs Rater 3", ['rater2', 'rater3'])
]

# Activity type to count column mapping
ACTIVITY_COUNT_MAPPING = {
    'tasks': 'activity_tasks_total_count',
    'decisions': 'activity_decisions_total_count',
    'meetings': 'activity_meetings_total_count',
    'discussions': 'activity_discussions_total_count',
    'weekly_updates': 'activity_weekly_updates_total_count'
}

REQUIRED_RATING_COLUMNS = ['title', 'rater1_rating', 'rater2_rating', 'rater3_rating']


class PlotConfig:
    """Configuration for generic plot creation."""
    def __init__(self,
                 title_template: str,
                 ylabel: str,
                 filename_prefix: str,
                 data_key: str,
                 results_key: str = 'results_by_type',
                 value_format: str = '{:.1f}',
                 color: str = 'steelblue',
                 figsize: tuple = (10, 6),
                 has_na_cases: bool = False,
                 has_dual_axis: bool = False,
                 percentage_key: str = None,
                 has_error_bars: bool = False,
                 secondary_color: str = 'darkred',
                 has_grid: bool = False,
                 activity_type_key: str = 'activity_type'):
        self.title_template = title_template
        self.ylabel = ylabel
        self.filename_prefix = filename_prefix
        self.data_key = data_key
        self.results_key = results_key
        self.activity_type_key = activity_type_key
        self.value_format = value_format
        self.color = color
        self.figsize = figsize
        self.has_na_cases = has_na_cases
        self.has_dual_axis = has_dual_axis
        self.percentage_key = percentage_key
        self.has_error_bars = has_error_bars
        self.secondary_color = secondary_color
        self.has_grid = has_grid




def create_generic_plot(result: dict, config: PlotConfig) -> str:
    """Generic plotting function for all metrics."""
    # Extract data for plotting
    activity_types = [item[config.activity_type_key] for item in result[config.results_key]]
    values = []
    labels = []
    error_bars = []

    # Handle data extraction with N/A cases and error bars
    for item in result[config.results_key]:
        if config.has_na_cases and item.get('is_na_case', False):
            values.append(0)  # Use 0 for plotting N/A cases
            labels.append('N/A')
            error_bars.append(0)
        else:
            values.append(item[config.data_key])
            # Handle special formatting for error bars and percentages
            if config.has_error_bars and 'std_error' in item:
                # Use hardcoded format for error bars
                labels.append(f"{item[config.data_key]:.2f}\n(±{item['std_error']:.2f})")
                error_bars.append(item['std_error'])
            elif config.has_dual_axis and config.percentage_key:
                # Use hardcoded format for dual axis
                labels.append(f"{item[config.data_key]:.0f}\n({item[config.percentage_key]:.1f}%)")
                error_bars.append(0)
            else:
                # Use config format for simple cases
                labels.append(config.value_format.format(item[config.data_key]))
                error_bars.append(0)

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=config.figsize)

    # Create bars with optional error bars
    if config.has_error_bars:
        bars = ax1.bar(activity_types, values, color=config.color, alpha=0.7,
                      yerr=error_bars, capsize=5)
    else:
        bars = ax1.bar(activity_types, values, color=config.color, alpha=0.7)

    # Add dual axis for percentages if needed
    if config.has_dual_axis and config.percentage_key:
        ax2 = ax1.twinx()
        percentages = [item[config.percentage_key] for item in result[config.results_key]]
        ax2.plot(activity_types, percentages, color=config.secondary_color, marker='o',
                linewidth=2, markersize=6)
        ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold', color=config.secondary_color)
        ax2.tick_params(axis='y', labelcolor=config.secondary_color)
        ax2.set_ylim(0, 100)

    # Add value labels on bars
    for bar, label, error in zip(bars, labels, error_bars):
        if config.has_na_cases and label == 'N/A':
            ax1.text(bar.get_x() + bar.get_width()/2., 0.5,
                    label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
        else:
            height = bar.get_height()
            margin = (height + error + max(values) * 0.05) if config.has_error_bars else (height + max(values) * 0.01)
            ax1.text(bar.get_x() + bar.get_width()/2., margin,
                    label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Set labels and title
    ax1.set_xlabel('Activity Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel(config.ylabel, fontsize=12, fontweight='bold', color=config.color)
    ax1.tick_params(axis='y', labelcolor=config.color)
    ax1.set_title(config.title_template.format(result["analysis_label"]),
                  fontsize=14, fontweight='bold', pad=20)

    # Format plot
    max_val = max(values) if values else 1
    max_err = max(error_bars) if error_bars else 0
    if config.has_error_bars:
        ax1.set_ylim(0, max_val + max_err + 0.5)
    else:
        ax1.set_ylim(0, max_val * 1.2)

    ax1.tick_params(axis='x', rotation=45, labelrotation=45)

    if config.has_grid:
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Generate filename
    safe_name = result['analysis_label'].lower().replace(' ', '_').replace('vs', 'vs')
    filename = f"{config.filename_prefix}_{safe_name}.png"
    filepath = METRICS_OUTPUT_DIR / filename

    # Save plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return str(filepath)


def calculate_acceptance_rates(long_df: pd.DataFrame, df: pd.DataFrame, rater_list: list[str], analysis_label: str) -> dict:
    """Calculate acceptance rates for specific rater subset."""
    # Filter to specific raters
    filtered_long_df = long_df[long_df['rater'].str.replace('_rating', '').isin(rater_list)]

    # Calculate acceptance rates by type using pandas groupby
    stats = filtered_long_df.groupby('title').agg({
        'rating': ['count', lambda x: (x == 'acceptable').sum()]
    }).round(1)
    stats.columns = ['total_ratings', 'acceptable_ratings']
    stats['acceptance_rate'] = (stats['acceptable_ratings'] / stats['total_ratings'] * 100)
    stats['goals'] = df.groupby('title').size()
    stats = stats.sort_index()

    # Convert to list of dictionaries for compatibility with existing output code
    acceptance_by_type = [
        {'type': idx, 'goals': row['goals'], 'total_ratings': row['total_ratings'],
         'acceptable_ratings': row['acceptable_ratings'], 'acceptance_rate': row['acceptance_rate']}
        for idx, row in stats.iterrows()
    ]

    # Calculate overall statistics
    overall_acceptable = sum(r['acceptable_ratings'] for r in acceptance_by_type)
    overall_total = sum(r['total_ratings'] for r in acceptance_by_type)
    overall_rate = (overall_acceptable / overall_total * 100) if overall_total > 0 else 0

    return {
        'analysis_label': analysis_label,
        'rater_list': rater_list,
        'acceptance_by_type': acceptance_by_type,
        'overall_acceptable': overall_acceptable,
        'overall_total': overall_total,
        'overall_rate': overall_rate
    }


def create_acceptance_rate_plot(result: dict) -> str:
    """Create bar chart for single acceptance rate analysis."""
    config = PlotConfig(
        title_template="Stated Goals Acceptance Rates - {}",
        ylabel="Acceptance Rate (%)",
        filename_prefix="stated_goals_acceptance_rates",
        data_key="acceptance_rate",
        results_key="acceptance_by_type",
        value_format="{:.1f}%",
        activity_type_key="type"
    )
    return create_generic_plot(result, config)


def create_cumulative_score_plot(result: dict) -> str:
    """Create bar chart for single cumulative score analysis."""
    config = PlotConfig(
        title_template="Unstated Goals Cumulative Scores - {}",
        ylabel="Cumulative Score",
        filename_prefix="unstated_goals_cumulative_quality",
        data_key="cumulative_score",
        results_key="scores_by_type",
        value_format="{:.2f}"
    )
    return create_generic_plot(result, config)


def average_of_average_analysis(df: pd.DataFrame, rater_list: list[str], analysis_label: str) -> dict:
    """Calculate average of average ratings for each activity type."""
    rater_columns = [f"{rater}_rating" for rater in rater_list]

    # Calculate per-goal average ratings for specified raters
    df['per_goal_avg'] = df[rater_columns].mean(axis=1)

    # Group by activity type and calculate average of per-goal averages plus dispersion measures
    avg_stats = df.groupby('title')['per_goal_avg'].agg(['mean', 'std', 'count']).reset_index()
    avg_stats.columns = ['activity_type', 'avg_of_avg', 'std_dev', 'sample_size']

    # Calculate standard error
    avg_stats['std_error'] = avg_stats['std_dev'] / (avg_stats['sample_size'] ** 0.5)

    # Sort alphabetically
    avg_stats = avg_stats.sort_values('activity_type')

    # Convert to list of dictionaries
    results_by_type = [
        {
            'activity_type': row['activity_type'],
            'avg_of_avg': row['avg_of_avg'],
            'std_dev': row['std_dev'],
            'std_error': row['std_error'],
            'sample_size': int(row['sample_size'])
        }
        for _, row in avg_stats.iterrows()
    ]

    return {
        'analysis_label': analysis_label,
        'rater_list': rater_list,
        'results_by_type': results_by_type
    }




def high_quality_goal_analysis(df: pd.DataFrame, rater_list: list[str], analysis_label: str) -> dict:
    """Analyze high-quality goal counts (average rating > 4.0) by activity type."""
    rater_columns = [f"{rater}_rating" for rater in rater_list]

    # Calculate per-goal average ratings for specified raters
    df['per_goal_avg'] = df[rater_columns].mean(axis=1)

    # Identify high-quality goals (average > 4.0)
    df['is_high_quality'] = df['per_goal_avg'] > 4.0

    # Group by activity type and calculate counts
    quality_stats = df.groupby('title').agg({
        'is_high_quality': ['count', 'sum']
    }).reset_index()
    quality_stats.columns = ['activity_type', 'total_goals', 'high_quality_count']

    # Calculate percentage
    quality_stats['percentage'] = (quality_stats['high_quality_count'] / quality_stats['total_goals'] * 100)

    # Sort alphabetically
    quality_stats = quality_stats.sort_values('activity_type')

    # Convert to list of dictionaries
    results_by_type = [
        {
            'activity_type': row['activity_type'],
            'total_goals': int(row['total_goals']),
            'high_quality_count': int(row['high_quality_count']),
            'percentage': row['percentage']
        }
        for _, row in quality_stats.iterrows()
    ]

    return {
        'analysis_label': analysis_label,
        'rater_list': rater_list,
        'results_by_type': results_by_type
    }




def information_density_analysis(df: pd.DataFrame, rater_list: list[str], analysis_label: str) -> dict:
    """Analyze information density (tokens per high-quality goal) by activity type."""
    # Load token data
    token_df = pd.read_csv(PROCESSED_GOAL_MINING_PATH)

    # Group token data by activity type and sum tokens
    token_stats = token_df.groupby('title')['num_sonnet_4_tokens_user_prompt'].sum().reset_index()
    token_stats.columns = ['activity_type', 'total_tokens']

    # Get high-quality goal counts for this rater combination
    high_quality_result = high_quality_goal_analysis(df.copy(), rater_list, analysis_label)

    # Create lookup dict for high-quality counts
    hq_counts = {item['activity_type']: item['high_quality_count']
                for item in high_quality_result['results_by_type']}

    # Merge token data with high-quality counts
    results = []
    for _, row in token_stats.iterrows():
        activity_type = row['activity_type']
        total_tokens = row['total_tokens']
        hq_count = hq_counts.get(activity_type, 0)

        # Calculate information density, handle division by zero
        if hq_count > 0:
            density = total_tokens / hq_count
            is_na_case = False
        else:
            density = None  # Use None for N/A cases
            is_na_case = True

        results.append({
            'activity_type': activity_type,
            'total_tokens': int(total_tokens),
            'high_quality_count': hq_count,
            'density': density,
            'is_na_case': is_na_case
        })

    # Sort alphabetically
    results.sort(key=lambda x: x['activity_type'])

    return {
        'analysis_label': analysis_label,
        'rater_list': rater_list,
        'results_by_type': results
    }




def activity_density_analysis(df: pd.DataFrame, rater_list: list[str], analysis_label: str) -> dict:
    """Analyze activity density (tokens per activity count) by activity type."""
    # Load token data
    token_df = pd.read_csv(PROCESSED_GOAL_MINING_PATH)

    # Group token data by activity type and sum tokens
    token_stats = token_df.groupby('title')['num_sonnet_4_tokens_user_prompt'].sum().reset_index()
    token_stats.columns = ['activity_type', 'total_tokens']

    # Load activity count data (same for all rows, so take first row)
    first_row = token_df.iloc[0]

    # Calculate activity density for each type
    results = []
    for _, row in token_stats.iterrows():
        activity_type = row['activity_type']
        total_tokens = row['total_tokens']

        # Get activity count from mapping
        count_column = ACTIVITY_COUNT_MAPPING[activity_type]
        activity_count = first_row[count_column]

        # Calculate activity density
        activity_density = total_tokens / activity_count

        results.append({
            'activity_type': activity_type,
            'total_tokens': int(total_tokens),
            'activity_count': int(activity_count),
            'activity_density': activity_density
        })

    # Sort alphabetically
    results.sort(key=lambda x: x['activity_type'])

    return {
        'analysis_label': analysis_label,
        'rater_list': rater_list,
        'results_by_type': results
    }


def activities_per_hq_goal_analysis(df: pd.DataFrame, rater_list: list[str], analysis_label: str) -> dict:
    """Analyze activities per high-quality goal by activity type."""
    # Load activity count data
    token_df = pd.read_csv(PROCESSED_GOAL_MINING_PATH)
    first_row = token_df.iloc[0]

    # Get high-quality goal counts for this rater combination
    high_quality_result = high_quality_goal_analysis(df.copy(), rater_list, analysis_label)

    # Create lookup dict for high-quality counts
    hq_counts = {item['activity_type']: item['high_quality_count']
                for item in high_quality_result['results_by_type']}

    # Calculate activities per high-quality goal
    results = []
    for activity_type in ACTIVITY_COUNT_MAPPING.keys():
        # Get activity count
        count_column = ACTIVITY_COUNT_MAPPING[activity_type]
        activity_count = first_row[count_column]

        # Get high-quality goal count
        hq_count = hq_counts.get(activity_type, 0)

        # Calculate activities per high-quality goal, handle division by zero
        if hq_count > 0:
            activities_per_hq_goal = activity_count / hq_count
            is_na_case = False
        else:
            activities_per_hq_goal = None  # Use None for N/A cases
            is_na_case = True

        results.append({
            'activity_type': activity_type,
            'activity_count': int(activity_count),
            'high_quality_count': hq_count,
            'activities_per_hq_goal': activities_per_hq_goal,
            'is_na_case': is_na_case
        })

    # Sort alphabetically
    results.sort(key=lambda x: x['activity_type'])

    return {
        'analysis_label': analysis_label,
        'rater_list': rater_list,
        'results_by_type': results
    }






def cumulative_quality_score(df: pd.DataFrame, rater_list: list[str], analysis_label: str) -> dict:
    """Calculate cumulative quality score for specific rater subset."""
    rater_columns = [f"{rater}_rating" for rater in rater_list]

    # Single groupby with all calculations
    score_stats = df.groupby('title').agg({
        **{col: 'sum' for col in rater_columns},
        'title': 'size'
    }).rename(columns={'title': 'goal_count'})

    # Calculate totals and scores
    score_stats['total_ratings'] = score_stats[rater_columns].sum(axis=1)
    score_stats['cumulative_score'] = score_stats['total_ratings'] / score_stats['goal_count']
    score_stats = score_stats.sort_index()

    # Get summary stats directly from DataFrame
    cumulative_scores = score_stats['cumulative_score']

    return {
        'analysis_label': analysis_label,
        'rater_list': rater_list,
        'scores_by_type': [
            {
                'activity_type': idx,
                'goal_count': row['goal_count'],
                'total_ratings': row['total_ratings'],
                'cumulative_score': row['cumulative_score']
            }
            for idx, row in score_stats.iterrows()
        ],
        'avg_score': cumulative_scores.mean(),
        'max_score': cumulative_scores.max(),
        'min_score': cumulative_scores.min()
    }


def analyse_unstated_goals() -> None:
    """Analyze unstated goals metrics and calculate cumulative quality scores."""
    print("Analyzing unstated goals cumulative quality scores...")

    if not RATED_UNSTATED_GOALS_PATH.exists():
        print(f"Error: Rated unstated goals file not found: {RATED_UNSTATED_GOALS_PATH}")
        return

    # Read data
    df = pd.read_csv(RATED_UNSTATED_GOALS_PATH)
    print(f"Loaded {len(df)} unstated goals from {RATED_UNSTATED_GOALS_PATH}")
    print(f"Columns: {list(df.columns)}")

    # Validate required columns
    if not all(col in df.columns for col in REQUIRED_RATING_COLUMNS):
        print(f"Error: Missing required columns. Expected: {REQUIRED_RATING_COLUMNS}")
        return

    # Calculate cumulative quality scores for all combinations
    print(f"Calculating cumulative quality scores for {len(RATER_ANALYSES)} rater combinations...")
    cumulative_results = []
    for analysis_label, rater_list in RATER_ANALYSES:
        print(f"  - {analysis_label}")
        result = cumulative_quality_score(df, rater_list, analysis_label)
        cumulative_results.append(result)

    # Calculate average of averages for all combinations
    print(f"Calculating average of averages for {len(RATER_ANALYSES)} rater combinations...")
    avg_of_avg_results = []
    for analysis_label, rater_list in RATER_ANALYSES:
        print(f"  - {analysis_label}")
        result = average_of_average_analysis(df.copy(), rater_list, analysis_label)
        avg_of_avg_results.append(result)

    # Calculate high-quality goal counts for all combinations
    print(f"Calculating high-quality goal counts for {len(RATER_ANALYSES)} rater combinations...")
    high_quality_results = []
    for analysis_label, rater_list in RATER_ANALYSES:
        print(f"  - {analysis_label}")
        result = high_quality_goal_analysis(df.copy(), rater_list, analysis_label)
        high_quality_results.append(result)

    # Calculate information density for all combinations
    print(f"Calculating information density for {len(RATER_ANALYSES)} rater combinations...")
    info_density_results = []
    for analysis_label, rater_list in RATER_ANALYSES:
        print(f"  - {analysis_label}")
        result = information_density_analysis(df.copy(), rater_list, analysis_label)
        info_density_results.append(result)

    # Calculate activity density for all combinations
    print(f"Calculating activity density for {len(RATER_ANALYSES)} rater combinations...")
    activity_density_results = []
    for analysis_label, rater_list in RATER_ANALYSES:
        print(f"  - {analysis_label}")
        result = activity_density_analysis(df.copy(), rater_list, analysis_label)
        activity_density_results.append(result)

    # Calculate activities per high-quality goal for all combinations
    print(f"Calculating activities per high-quality goal for {len(RATER_ANALYSES)} rater combinations...")
    activities_per_hq_results = []
    for analysis_label, rater_list in RATER_ANALYSES:
        print(f"  - {analysis_label}")
        result = activities_per_hq_goal_analysis(df.copy(), rater_list, analysis_label)
        activities_per_hq_results.append(result)

    # Create output directory
    METRICS_OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate plots for each analysis
    plot_files = []

    # Cumulative quality score plots
    for result in cumulative_results:
        plot_file = create_cumulative_score_plot(result)
        plot_files.append(plot_file)

    # Average of average plots
    for result in avg_of_avg_results:
        config = PlotConfig(
            title_template='Unstated Goals Average of Averages - {}',
            ylabel='Average of Average Ratings',
            filename_prefix='unstated_goals_avg_of_avg',
            data_key='avg_of_avg',
            results_key='results_by_type',
            value_format='{:.2f}\n(±{std_error:.2f})',
            color='steelblue',
            has_error_bars=True
        )
        plot_file = create_generic_plot(result, config)
        plot_files.append(plot_file)

    # High-quality goal count plots
    for result in high_quality_results:
        config = PlotConfig(
            title_template='High-Quality Unstated Goals (>4.0) - {}',
            ylabel='High-Quality Goal Count',
            filename_prefix='unstated_goals_high_quality',
            data_key='high_quality_count',
            results_key='results_by_type',
            value_format='{:.0f}\n({percentage:.1f}%)',
            color='steelblue',
            has_dual_axis=True,
            percentage_key='percentage',
            secondary_color='darkred'
        )
        plot_file = create_generic_plot(result, config)
        plot_files.append(plot_file)

    # Information density plots
    for result in info_density_results:
        config = PlotConfig(
            title_template='Information Density - {}',
            ylabel='Information Density (Tokens/High-Quality Goal)',
            filename_prefix='unstated_goals_info_density',
            data_key='density',
            results_key='results_by_type',
            value_format='{:.1f}',
            color='steelblue',
            has_na_cases=True
        )
        plot_file = create_generic_plot(result, config)
        plot_files.append(plot_file)

    # Activity density plots
    for result in activity_density_results:
        config = PlotConfig(
            title_template='Activity Density (Tokens per Activity)\n{}',
            ylabel='Tokens per Activity',
            filename_prefix='unstated_goals_activity_density',
            data_key='activity_density',
            results_key='results_by_type',
            value_format='{:.1f}',
            color='steelblue',
            figsize=(12, 8),
            has_grid=True
        )
        plot_file = create_generic_plot(result, config)
        plot_files.append(plot_file)

    # Activities per high-quality goal plots
    for result in activities_per_hq_results:
        config = PlotConfig(
            title_template='Activities per High-Quality Goal\n{}',
            ylabel='Activities per High-Quality Goal',
            filename_prefix='unstated_goals_activities_per_hq_goal',
            data_key='activities_per_hq_goal',
            results_key='results_by_type',
            value_format='{:.1f}',
            color='forestgreen',
            figsize=(12, 8),
            has_na_cases=True,
            has_grid=True
        )
        plot_file = create_generic_plot(result, config)
        plot_files.append(plot_file)

    # Write results to file
    with open(UNSTATED_GOALS_SUMMARY_FILE, 'w') as f:
        f.write("UNSTATED GOALS METRICS ANALYSIS\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Unstated Goals: {len(df)}\n")
        f.write(f"Number of Analyses: {len(cumulative_results)}\n\n")

        # Activity Counts Section (single table, same for all rater combinations)
        f.write("ACTIVITY COUNTS BY TYPE\n")
        f.write("-" * 25 + "\n")
        f.write("Activity counts are consistent across all activity types in the dataset.\n\n")

        # Load activity count data once
        token_df = pd.read_csv(PROCESSED_GOAL_MINING_PATH)
        first_row = token_df.iloc[0]

        f.write(f"{'Activity Type':<20} {'Activity Count':<15}\n")
        f.write("-" * 40 + "\n")

        # Sort activity types alphabetically for consistency
        sorted_activity_types = sorted(ACTIVITY_COUNT_MAPPING.keys())
        for activity_type in sorted_activity_types:
            count_column = ACTIVITY_COUNT_MAPPING[activity_type]
            activity_count = first_row[count_column]
            f.write(f"{activity_type:<20} {int(activity_count):<15}\n")
        f.write("\n\n")

        # Cumulative Quality Score Analysis Section
        f.write("="*80 + "\n")
        f.write("CUMULATIVE QUALITY SCORE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write("Formula: Sum of specified rater ratings / Number of goals per type\n")
        f.write("This metric provides a weighted quality score accounting for both rating values and goal volume.\n\n")

        # Write each analysis
        for i, result in enumerate(cumulative_results, 1):
            f.write(f"ANALYSIS {i}: {result['analysis_label'].upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Raters: {', '.join(result['rater_list'])}\n")
            f.write(f"Sample Size: {len(df)} goals\n\n")

            f.write("CUMULATIVE QUALITY SCORE BY ACTIVITY TYPE\n")
            f.write("-" * 45 + "\n")
            f.write("Formula: Sum of specified rater ratings / Number of goals per type\n\n")
            f.write(f"{'Activity Type':<20} {'Goals':<8} {'Total Ratings':<15} {'Cumulative Score':<18}\n")
            f.write("-" * 70 + "\n")

            for score_result in result['scores_by_type']:
                f.write(f"{score_result['activity_type']:<20} {score_result['goal_count']:<8} "
                       f"{score_result['total_ratings']:<15} {score_result['cumulative_score']:<17.2f}\n")

            f.write("\n")

        # Average of Averages Analysis Section
        f.write("\n" + "="*80 + "\n")
        f.write("AVERAGE OF AVERAGES ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write("Formula: Average of (per-goal average ratings) per activity type\n")
        f.write("This metric calculates the mean rating for each goal, then averages those means by activity type.\n\n")

        # Write each avg of avg analysis
        for i, result in enumerate(avg_of_avg_results, 1):
            f.write(f"ANALYSIS {i}: {result['analysis_label'].upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Raters: {', '.join(result['rater_list'])}\n")
            f.write(f"Sample Size: {len(df)} goals\n\n")

            f.write("AVERAGE OF AVERAGES BY ACTIVITY TYPE\n")
            f.write("-" * 45 + "\n")
            f.write(f"{'Activity Type':<20} {'Goals':<8} {'Avg of Avg':<12} {'Std Dev':<10} {'Std Error':<12} {'67% CI':<20} {'95% CI':<20}\n")
            f.write("-" * 100 + "\n")

            for type_result in result['results_by_type']:
                avg_val = type_result['avg_of_avg']
                std_err = type_result['std_error']
                # 67% CI (approximately ±1 standard error)
                ci67_lower = avg_val - std_err
                ci67_upper = avg_val + std_err
                # 95% CI (±1.96 standard errors)
                ci95_lower = avg_val - 1.96 * std_err
                ci95_upper = avg_val + 1.96 * std_err
                f.write(f"{type_result['activity_type']:<20} {type_result['sample_size']:<8} "
                       f"{avg_val:<12.2f} {type_result['std_dev']:<10.3f} "
                       f"{std_err:<12.3f} [{ci67_lower:6.2f}, {ci67_upper:6.2f}] "
                       f"[{ci95_lower:6.2f}, {ci95_upper:6.2f}]\n")

            f.write("\n")

        # High-Quality Goal Counts Analysis Section
        f.write("\n" + "="*80 + "\n")
        f.write("HIGH-QUALITY GOAL COUNTS ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write("Definition: High-quality goals have average rating > 4.0\n")
        f.write("This analysis counts and calculates percentages of high-quality goals by activity type.\n\n")

        # Write each high-quality analysis
        for i, result in enumerate(high_quality_results, 1):
            f.write(f"ANALYSIS {i}: {result['analysis_label'].upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Raters: {', '.join(result['rater_list'])}\n")
            f.write(f"Sample Size: {len(df)} goals\n\n")

            f.write("HIGH-QUALITY GOAL COUNTS BY ACTIVITY TYPE\n")
            f.write("-" * 45 + "\n")
            f.write(f"{'Activity Type':<20} {'Total Goals':<12} {'High-Quality':<15} {'Percentage':<12}\n")
            f.write("-" * 65 + "\n")

            for type_result in result['results_by_type']:
                f.write(f"{type_result['activity_type']:<20} {type_result['total_goals']:<12} "
                       f"{type_result['high_quality_count']:<15} {type_result['percentage']:<11.1f}%\n")

            f.write("\n")

        # Information Density Analysis Section
        f.write("\n" + "="*80 + "\n")
        f.write("INFORMATION DENSITY ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write("Definition: Information density = Total input tokens / High-quality goal count\n")
        f.write("This metric measures token efficiency for producing high-quality goals.\n")
        f.write("N/A indicates zero high-quality goals (division by zero).\n\n")

        # Write each info density analysis
        for i, result in enumerate(info_density_results, 1):
            f.write(f"ANALYSIS {i}: {result['analysis_label'].upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Raters: {', '.join(result['rater_list'])}\n")
            f.write(f"Sample Size: {len(df)} goals\n\n")

            f.write("COMPREHENSIVE INFORMATION DENSITY BY ACTIVITY TYPE\n")
            f.write("-" * 85 + "\n")
            f.write("Additional Activity Metrics:\n")
            f.write("• Activity Count = Total activities processed per type\n")
            f.write("• Activity Density = Total tokens / Activity count\n")
            f.write("• Activities/HQ = Activity count / High-quality goals (N/A if zero HQ goals)\n\n")
            f.write(f"{'Activity Type':<15} {'Total':<8} {'Activity':<9} {'High':<5} {'Info':<8} {'Activity':<9} {'Act/':<6}\n")
            f.write(f"{'':<15} {'Tokens':<8} {'Count':<9} {'Qual':<5} {'Density':<8} {'Density':<9} {'HQ':<6}\n")
            f.write("-" * 85 + "\n")

            # Get corresponding activity and activities_per_hq results for this rater combination
            activity_result = activity_density_results[i-1]
            activities_hq_result = activities_per_hq_results[i-1]

            # Create lookups
            activity_lookup = {item['activity_type']: item for item in activity_result['results_by_type']}
            activities_hq_lookup = {item['activity_type']: item for item in activities_hq_result['results_by_type']}

            for type_result in result['results_by_type']:
                activity_type = type_result['activity_type']
                activity_data = activity_lookup[activity_type]
                activities_hq_data = activities_hq_lookup[activity_type]

                # Format values
                info_density_str = "N/A" if type_result['is_na_case'] else f"{type_result['density']:.0f}"
                activities_hq_str = "N/A" if activities_hq_data['is_na_case'] else f"{activities_hq_data['activities_per_hq_goal']:.1f}"

                f.write(f"{activity_type:<15} {type_result['total_tokens']:<8} "
                       f"{activity_data['activity_count']:<9} {type_result['high_quality_count']:<5} "
                       f"{info_density_str:<8} {activity_data['activity_density']:<9.0f} {activities_hq_str:<6}\n")

            f.write("\n")

    # Console output
    print(f"Unstated goals analysis complete. Results saved to: {UNSTATED_GOALS_SUMMARY_FILE}")
    print(f"Plots generated:")
    for plot_file in plot_files:
        print(f"  - {plot_file}")


def analyse_stated_goals() -> None:
    """Analyze stated goals metrics and calculate acceptance rates."""
    print("Analyzing stated goals acceptance rates...")

    if not RATED_STATED_GOALS_PATH.exists():
        print(f"Error: Rated stated goals file not found: {RATED_STATED_GOALS_PATH}")
        return

    # Read data
    df = pd.read_csv(RATED_STATED_GOALS_PATH)
    print(f"Loaded {len(df)} stated goals from {RATED_STATED_GOALS_PATH}")
    print(f"Columns: {list(df.columns)}")

    # Validate required columns
    if not all(col in df.columns for col in REQUIRED_RATING_COLUMNS):
        print(f"Error: Missing required columns. Expected: {REQUIRED_RATING_COLUMNS}")
        return

    # Transform to long format
    long_df = pd.melt(
        df,
        id_vars=['title'],
        value_vars=['rater1_rating', 'rater2_rating', 'rater3_rating'],
        var_name='rater',
        value_name='rating'
    )

    # Calculate acceptance rates for all combinations
    print(f"Calculating acceptance rates for {len(RATER_ANALYSES)} rater combinations...")
    results = []
    for analysis_label, rater_list in RATER_ANALYSES:
        print(f"  - {analysis_label}")
        result = calculate_acceptance_rates(long_df, df, rater_list, analysis_label)
        results.append(result)

    # Create output directory
    METRICS_OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate plots for each analysis
    plot_files = []
    for result in results:
        plot_file = create_acceptance_rate_plot(result)
        plot_files.append(plot_file)

    # Write results to file
    with open(STATED_GOALS_SUMMARY_FILE, 'w') as f:
        f.write("STATED GOALS ACCEPTANCE RATE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Stated Goals: {len(df)}\n")
        f.write(f"Number of Analyses: {len(results)}\n\n")

        # Write each analysis
        for i, result in enumerate(results, 1):
            f.write(f"ANALYSIS {i}: {result['analysis_label'].upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Raters: {', '.join(result['rater_list'])}\n")
            f.write(f"Total Ratings: {result['overall_total']}\n\n")

            f.write("ACCEPTANCE RATES BY ACTIVITY TYPE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Activity Type':<20} {'Goals':<8} {'Total Ratings':<15} {'Acceptable':<12} {'Acceptance Rate':<15}\n")
            f.write("-" * 80 + "\n")

            for type_result in result['acceptance_by_type']:
                f.write(f"{type_result['type']:<20} {type_result['goals']:<8} {type_result['total_ratings']:<15} "
                       f"{type_result['acceptable_ratings']:<12} {type_result['acceptance_rate']:<14.1f}%\n")

            f.write(f"\nOVERALL SUMMARY FOR {result['analysis_label'].upper()}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall Acceptance Rate: {result['overall_rate']:.1f}%\n")
            f.write(f"Total Acceptable Ratings: {result['overall_acceptable']}\n")
            f.write(f"Total Ratings: {result['overall_total']}\n\n")

    # Console output
    print(f"\nResults saved to: {STATED_GOALS_SUMMARY_FILE}")
    print(f"Plots generated:")
    for plot_file in plot_files:
        print(f"  - {plot_file}")


def metrics_analysis() -> None:
    """Main metrics analysis function."""
    print("Checking metrics analysis...")

    if METRICS_OUTPUT_DIR.exists():
        print(f"Metrics analysis already exists at: {METRICS_OUTPUT_DIR}")
        print("Skipping metrics analysis.")
        return

    print("Starting metrics analysis...")

    analyse_stated_goals()
    analyse_unstated_goals()

    print("Metrics analysis complete.")
