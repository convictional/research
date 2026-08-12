from collections import defaultdict
from datetime import datetime, timedelta
from matplotlib.cm import inferno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pytz
from scipy.optimize import curve_fit

from ..helpers.io import load_checkpoint
from ..settings import settings


def analyze_fact_distribution():
    """Analyze how facts are distributed across source documents for each named entity."""

    # Load the final entities from checkpoint
    entities = load_checkpoint("entity_store_final_entities")
    if not entities:
        raise ValueError("No checkpoint found")

    # Define cutoff date for recent documents (6 months ago) with UTC timezone
    cutoff_date = datetime.now(pytz.UTC) - timedelta(days=180)

    # Create analysis structures
    entity_stats = []
    recent_entity_stats = []

    for entity in entities:
        # Analyze all facts
        all_stats = analyze_entity_facts(entity.facts, entity.name, entity.entity_type)
        entity_stats.append(all_stats)

        # Filter for recent facts, ensuring datetime comparison uses UTC
        recent_facts = [fact for fact in entity.facts if fact.created_at.replace(tzinfo=pytz.UTC) >= cutoff_date]
        if recent_facts:  # Only include if entity has recent facts
            recent_stats = analyze_entity_facts(recent_facts, entity.name, entity.entity_type)
            recent_entity_stats.append(recent_stats)

    # Save results
    output_dir = settings.output_path / "fact_distribution_analysis"
    output_dir.mkdir(exist_ok=True)

    # Save summaries
    save_summary_data(entity_stats, output_dir / "entity_fact_distribution_summary.csv")
    save_summary_data(recent_entity_stats, output_dir / "recent_entity_fact_distribution_summary.csv")

    # Save detailed distributions
    save_distribution_data(entity_stats, output_dir / "entity_fact_distribution_detailed.csv")
    save_distribution_data(recent_entity_stats, output_dir / "recent_entity_fact_distribution_detailed.csv")

    # Calculate and save aggregate statistics
    all_agg_stats = calculate_aggregate_stats(entity_stats, "all_time")
    recent_agg_stats = calculate_aggregate_stats(recent_entity_stats, "recent")

    # Combine stats and save
    combined_stats = {**all_agg_stats, **recent_agg_stats}
    pd.Series(combined_stats).to_csv(output_dir / "aggregate_statistics.csv")

    # Create visualizations with proper file names
    save_fact_distribution_plot(entity_stats, output_dir / "fact_distribution_visualization.png")
    if recent_entity_stats:  # Only create recent visualization if we have data
        save_fact_distribution_plot(recent_entity_stats, output_dir / "recent_fact_distribution_visualization.png")

    return entity_stats, recent_entity_stats, combined_stats


def analyze_entity_facts(facts, entity_name, entity_type):
    """Analyze facts distribution for a given set of facts."""
    source_docs = set(fact.source_id for fact in facts)
    total_docs = len(source_docs)
    total_facts = len(facts)

    # Count facts per document
    doc_fact_counts = defaultdict(int)
    for fact in facts:
        doc_fact_counts[fact.source_id] += 1

    # Sort documents by fact count (descending)
    sorted_docs = sorted(doc_fact_counts.items(), key=lambda x: x[1], reverse=True)

    # Calculate cumulative fact distribution
    cumulative_facts = 0
    distribution_data = []

    for doc_num, (_, fact_count) in enumerate(sorted_docs, 1):
        cumulative_facts += fact_count
        pct_docs = doc_num / total_docs * 100
        pct_facts = cumulative_facts / total_facts * 100

        distribution_data.append({"doc_count": doc_num, "pct_docs": pct_docs, "pct_facts": pct_facts})

    # Find key percentile points
    percentile_points = []
    for target_pct in [50, 75, 90, 95]:
        for point in distribution_data:
            if point["pct_facts"] >= target_pct:
                percentile_points.append(
                    {
                        "percentile": target_pct,
                        "docs_needed": point["doc_count"],
                        "pct_docs_needed": point["pct_docs"],
                    }
                )
                break

    return {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "total_facts": total_facts,
        "total_docs": total_docs,
        "facts_per_doc": total_facts / total_docs if total_docs > 0 else 0,
        "distribution_data": distribution_data,
        "percentile_points": percentile_points,
    }


def save_summary_data(stats, output_path):
    """Save entity-level summary statistics."""
    summary_rows = []
    for stat in stats:
        row = {
            "entity_name": stat["entity_name"],
            "entity_type": stat["entity_type"],
            "total_facts": stat["total_facts"],
            "total_docs": stat["total_docs"],
            "facts_per_doc": stat["facts_per_doc"],
        }
        # Add percentile data
        for p in stat["percentile_points"]:
            row[f'docs_for_{p["percentile"]}pct_facts'] = p["docs_needed"]
            row[f'pct_docs_for_{p["percentile"]}pct_facts'] = p["pct_docs_needed"]
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(output_path, index=False)


def save_distribution_data(stats, output_path):
    """Save detailed distribution data."""
    distribution_rows = []
    for stat in stats:
        for point in stat["distribution_data"]:
            distribution_rows.append(
                {
                    "entity_name": stat["entity_name"],
                    "entity_type": stat["entity_type"],
                    "doc_count": point["doc_count"],
                    "pct_docs": point["pct_docs"],
                    "pct_facts": point["pct_facts"],
                }
            )

    pd.DataFrame(distribution_rows).to_csv(output_path, index=False)


def calculate_aggregate_stats(stats, prefix):
    """Calculate aggregate statistics for a set of entity stats."""
    if not stats:
        return {}

    agg_stats = {
        f"{prefix}_total_entities": len(stats),
        f"{prefix}_avg_facts_per_entity": np.mean([s["total_facts"] for s in stats]),
        f"{prefix}_avg_docs_per_entity": np.mean([s["total_docs"] for s in stats]),
        f"{prefix}_avg_facts_per_doc": np.mean([s["facts_per_doc"] for s in stats]),
    }

    # Calculate percentile statistics
    for target_pct in [50, 75, 90, 95]:
        docs_needed = [
            next(p["docs_needed"] for p in s["percentile_points"] if p["percentile"] == target_pct) for s in stats
        ]
        agg_stats[f"{prefix}_avg_docs_for_{target_pct}pct_facts"] = np.mean(docs_needed)
        agg_stats[f"{prefix}_median_docs_for_{target_pct}pct_facts"] = np.median(docs_needed)

    return agg_stats


def plot_fact_distribution(df: pd.DataFrame, output_path: Path):
    """Create a scatter plot with trend lines showing fact distribution across documents."""
    plt.figure(figsize=(15, 10))

    colors = [inferno(x) for x in np.linspace(0.2, 0.8, 4)]
    percentiles = [50, 75, 90, 95]

    n_points = len(df)  # Will be 500 from our filtering
    for idx, percentile in enumerate(percentiles):
        col_name = f"docs_for_{percentile}pct_facts"

        # Get data points - use actual positions (1-based)
        x = np.array(range(1, n_points + 1))
        y = df[col_name].values

        # Create scatter plot
        plt.scatter(x, y, c=[colors[idx]], s=3, alpha=0.5, label=f"{percentile}% coverage")

        # Take logs first
        log_y = np.log(y)

        # Fit log(y) = log(a) - b*log(x)
        # Initial guess based on observed pattern: a increases with percentile, b ≈ 1.1
        initial_a = percentile * 30  # rough scaling factor

        def fit_func(x, log_a, b):
            return log_a - b * np.log(x)

        popt, _ = curve_fit(
            fit_func,
            x,
            log_y,
            p0=[np.log(initial_a), 1.1],
            bounds=([0, 0.9], [np.inf, 1.3]),  # constrain b between 0.9 and 1.3
        )
        log_a, b = popt
        a = np.exp(log_a)

        # Plot the fitted curve
        x_smooth = np.linspace(1, n_points, 100)
        y_smooth = np.exp(fit_func(x_smooth, log_a, b))
        plt.plot(
            x_smooth,
            y_smooth,
            color=colors[idx],
            alpha=0.8,
            linewidth=2,
            label=f"{percentile}%: log(y) = {a:.0f}·x^(-{b:.2f})",
        )

    # Only log scale for y-axis
    plt.yscale("log")

    # Customize grid
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Customize background
    plt.gca().set_facecolor("#f5f5f5")
    plt.gcf().patch.set_facecolor("white")

    # Labels and title
    plt.xlabel("Top 500 Entities, sorted by total associated documents")
    plt.ylabel("Document Count (Log Scale)")
    plt.title("Distribution of Facts across Documents\nTop 500 Entities based on total related documents")

    # Update legend to show the fit equations
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, title="Coverage Thresholds & Power Law Fits")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def save_fact_distribution_plot(stats, output_path: Path):
    """Create and save fact distribution visualization."""
    # Convert stats to DataFrame
    summary_rows = []
    for stat in stats:
        row = {
            "entity_name": stat["entity_name"],
            "total_facts": stat["total_facts"],
            "total_docs": stat["total_docs"],
        }
        # Add percentile data
        for p in stat["percentile_points"]:
            row[f'docs_for_{p["percentile"]}pct_facts'] = p["docs_needed"]
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    # Sort by total_docs and take top 500
    df = df.sort_values("total_docs", ascending=False).head(500).reset_index(drop=True)

    # Create visualization - use output_path directly since it now includes the filename
    plot_fact_distribution(df, output_path)
