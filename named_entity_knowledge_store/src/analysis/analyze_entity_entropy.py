import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from typing import Dict

from ..helpers.io import load_checkpoint
from ..settings import settings, logger


def analyze_entity_entropy():
    """Analyze entity entropy distribution and generate visualizations."""
    # Load entities from checkpoint
    entities = load_checkpoint("entity_store_final_entities")
    if not entities:
        raise ValueError("No checkpoint found")

    # Create output directory
    output_dir = settings.output_path / "entity_entropy_analysis"
    output_dir.mkdir(exist_ok=True)

    # Prepare data structures
    entity_data = []

    for idx, entity in enumerate(entities):
        # Count facts per document
        facts_by_doc = defaultdict(int)
        for fact in entity.facts:
            facts_by_doc[str(fact.source_id)] += 1

        if not facts_by_doc:  # Skip entities with no facts
            continue

        entropy = calculate_entity_entropy(facts_by_doc)
        coverage_docs = calculate_coverage_threshold(facts_by_doc)

        entity_data.append(
            {
                "name": entity.name,
                "type": entity.entity_type,
                "total_facts": len(entity.facts),
                "unique_docs": len(facts_by_doc),
                "entropy": entropy,
                "coverage_docs": coverage_docs,
                "entity_idx": idx,
            }
        )

    df = pd.DataFrame(entity_data)

    # Generate all visualizations
    plot_entropy_distribution(df, output_dir / "entropy_distribution.png")
    plot_size_entropy_relationship(df, output_dir / "size_entropy_relationship.png")
    plot_coverage_distribution(df, output_dir / "coverage_distribution.png")
    plot_entropy_by_category(df, output_dir / "entropy_by_category.png")
    plot_temporal_entropy_top_k(df, entities, output_dir / "temporal_entropy_large.png")
    plot_temporal_entropy_small(df, entities, output_dir / "temporal_entropy_small.png")
    plot_early_vs_final_entropy(df, entities, output_dir / "early_vs_final_entropy.png")

    # Save raw data
    df.to_csv(output_dir / "entity_entropy_metrics.csv", index=False)

    # Log summary statistics
    log_summary_statistics(df)


def calculate_entity_entropy(facts_by_doc: Dict[str, int]) -> float:
    """Calculate Shannon entropy for an entity's fact distribution across documents."""
    total_facts = sum(facts_by_doc.values())
    if total_facts == 0:
        return 0.0

    # Calculate probability distribution
    probs = [count / total_facts for count in facts_by_doc.values()]
    # Calculate entropy using log base 2 (bits)
    return -sum(p * np.log2(p) for p in probs if p > 0)


def calculate_coverage_threshold(facts_by_doc: Dict[str, int], threshold: float = 0.95) -> int:
    """Calculate number of documents needed to reach coverage threshold."""
    total_facts = sum(facts_by_doc.values())
    if total_facts == 0:
        return 0

    # Sort documents by fact count in descending order
    sorted_docs = sorted(facts_by_doc.values(), reverse=True)
    cumulative_facts = 0
    for i, count in enumerate(sorted_docs, 1):
        cumulative_facts += count
        if cumulative_facts / total_facts >= threshold:
            return i
    return len(sorted_docs)


def calculate_temporal_entropy(entity, dates) -> Dict[str, float]:
    """Calculate entropy of an entity at different points in time."""
    temporal_entropy = {}

    # Sort facts by date
    facts_by_date = defaultdict(lambda: defaultdict(int))
    for fact in entity.facts:
        fact_date = fact.created_at.date()
        facts_by_date[fact_date][str(fact.source_id)] += 1

    # Calculate cumulative entropy for each date
    cumulative_facts = defaultdict(int)
    for date in sorted(dates):
        # Accumulate all facts up to this date
        for d in facts_by_date.keys():
            if d <= date:
                for doc, count in facts_by_date[d].items():
                    cumulative_facts[doc] += count

        # Calculate entropy if we have any facts
        if cumulative_facts:
            temporal_entropy[date] = calculate_entity_entropy(cumulative_facts)

    return temporal_entropy


def plot_entropy_distribution(df: pd.DataFrame, output_path: Path):
    """Create Figure 1: Distribution of entity entropies."""
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

    plt.figure(figsize=(12, 8))

    # Create histogram showing counts
    ax = plt.gca()
    sns.histplot(
        data=df,
        x="entropy",
        stat="count",  # Changed from density to count
        alpha=0.6,
        color=plt.cm.viridis(0.3),
        ax=ax,
    )

    # Add KDE plot on secondary y-axis
    ax2 = ax.twinx()
    sns.kdeplot(data=df, x="entropy", color=plt.cm.viridis(0.8), linewidth=2, ax=ax2, label="KDE (solid orange line)")
    ax2.legend()

    # Labels and formatting
    ax.set_xlabel("Entity Information Content (bits)")
    ax.set_ylabel("Count")
    ax2.set_ylabel("Density")

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add summary statistics
    stats_text = f"Mean: {df['entropy'].mean():.2f} bits\nMedian: {df['entropy'].median():.2f} bits"
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_size_entropy_relationship(df: pd.DataFrame, output_path: Path):
    """Create Figure 2: Relationship between entity size and entropy."""
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

    plt.figure(figsize=(12, 8))

    # Create scatter plot with logarithmic scales using viridis colors
    plt.scatter(df["total_facts"], df["entropy"], alpha=0.5, s=20, color=plt.cm.viridis(0.3))
    plt.xscale("log")

    # Add trend line with viridis color
    z = np.polyfit(np.log10(df["total_facts"]), df["entropy"], 1)
    p = np.poly1d(z)
    x_trend = np.logspace(0, np.log10(df["total_facts"].max()), 100)
    plt.plot(x_trend, p(np.log10(x_trend)), "--", color=plt.cm.viridis(0.8), alpha=0.8)

    plt.xlabel("Total Facts (log scale)")
    plt.ylabel("Entropy (bits)")

    # Add correlation coefficient
    corr = df["total_facts"].corr(df["entropy"])
    plt.text(
        0.05,
        0.95,
        f"Correlation: {corr:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_coverage_distribution(df: pd.DataFrame, output_path: Path):
    """Create Figure 3: Rank-ordered plot of document coverage."""
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

    plt.figure(figsize=(12, 8))

    # Sort entities by coverage docs needed
    sorted_coverage = np.sort(df["coverage_docs"].values)
    ranks = np.arange(1, len(sorted_coverage) + 1) / len(sorted_coverage)

    plt.plot(ranks, sorted_coverage, "-", color=plt.cm.viridis(0.6))
    plt.yscale("log")

    plt.axvline(x=0.90, color="red", linestyle="--", alpha=0.8, label="90th percentile")

    plt.xlabel("Entity Rank (normalized)")
    plt.ylabel("Documents Needed for 95% Coverage (log scale)")

    # Add summary statistics
    stats_text = (
        f"Median docs: {df['coverage_docs'].median():.0f}\n"
        f"90th percentile: {df['coverage_docs'].quantile(0.9):.0f}\n"
        f"Max docs: {df['coverage_docs'].max():.0f}"
    )
    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_entropy_by_category(df: pd.DataFrame, output_path: Path, min_observations: int = 5):
    """Create vertically stacked histograms of entropy by entity category."""
    # Filter categories with too few observations and calculate max entropy
    category_counts = df.groupby("type").size()
    valid_categories = category_counts[category_counts >= min_observations].index
    df_filtered = df[df["type"].isin(valid_categories)]

    # Calculate max entropy for each remaining category and sort
    max_entropies = (
        df_filtered.groupby("type")["entropy"].max().sort_values(ascending=True)
    )  # Sort ascending for bottom-to-top
    category_order = max_entropies.index.tolist()

    n_categories = len(category_order)
    fig, axs = plt.subplots(n_categories, 1, figsize=(12, n_categories * 2))

    if n_categories == 1:  # Handle case with single category
        axs = [axs]

    fig.subplots_adjust(hspace=0.05)

    # Find global max count for normalizing histograms
    max_density = 0
    for category in category_order:  # Use ordered categories
        category_data = df[df["type"] == category]["entropy"]
        if len(category_data) > 0:
            hist, _ = np.histogram(category_data, bins=20, density=True)
            max_density = max(max_density, hist.max())

    # Find global max entropy for consistent x-axis
    max_entropy = df_filtered["entropy"].max()

    # Create histograms for each category
    for ax, category in zip(axs, category_order):
        category_data = df[df["type"] == category]["entropy"]
        count = len(category_data)

        if count > 0:
            # Create histogram with log scale
            n, bins, patches = ax.hist(category_data, bins=20, density=True, color=plt.cm.viridis(0.3), alpha=0.6)
            ax.set_yscale("log")

            # Add scatter points for individual observations
            y_height = max(n) * 1.1
            ax.scatter(
                category_data, [y_height] * len(category_data), color=plt.cm.viridis(0.6), alpha=0.3, s=20, zorder=3
            )

        # Add category label and stats below x-axis - adjust position for bottom plot
        stats_text = f"{category} (n={count:,}, max={category_data.max():.2f})"
        y_pos = -0.2 if ax == axs[-1] else -0.1  # Lower position for bottom plot
        ax.annotate(
            stats_text,
            xy=(0.5, y_pos),  # Adjusted y-position
            xytext=(0, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="center",
            va="center",
        )

        # Clean up the plot
        ax.set_ylim(0.01, max_density * 2)  # Adjusted for log scale
        ax.set_yticks([0.01, 0.1, 1])  # Show a reference point
        ax.set_yticklabels(["0.01", "0.1", "1"])  # Log density scale
        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(True)

        # Add y-axis label only to top plot
        if ax == axs[0]:
            ax.set_ylabel("Log Density")

        # Set consistent x-axis limits
        ax.set_xlim(0, max_entropy + 1)

        # Show x-axis ticks only for bottom plot
        if ax != axs[-1]:
            ax.set_xticklabels([])

    # Remove the global x-axis label since we now have category labels
    # Add only the title
    fig.suptitle("Entity Entropy Distribution by Category\n(ordered by maximum entropy)", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close()


def plot_temporal_entropy_top_k(df: pd.DataFrame, entities, output_path: Path):
    """Create visualization of entropy evolution over time for top entities."""
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

    plt.figure(figsize=(15, 8))

    # Get top 5 entities by current entropy
    top_indices = df.nlargest(5, "entropy")["entity_idx"].tolist()
    print(top_indices)

    # Define custom names for the top 5 entities
    custom_names = ["Convictional", "Company Main Product", "Founder", "Head of Engineering", "UX Principles"]

    # Create color palette from viridis
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_indices)))

    # Calculate and plot temporal entropy for each top entity
    for idx, (entity_idx, color) in enumerate(zip(top_indices, colors)):
        entity = entities[entity_idx]  # Match by index
        # Get all dates for this entity
        entity_dates = sorted(set(fact.created_at.date() for fact in entity.facts))
        entity_earliest_date = min(entity_dates)

        temporal_entropy = calculate_temporal_entropy(entity, entity_dates)

        # Convert dates to days since this entity's first mention
        days_since_start = [(date - entity_earliest_date).days for date in temporal_entropy.keys()]
        entropy_values = list(temporal_entropy.values())

        # Use custom name instead of entity index
        plt.plot(
            days_since_start,
            entropy_values,
            "-o",
            label=f"{custom_names[idx]} (H={df[df['entity_idx'] == entity_idx]['entropy'].iloc[0]:.2f})",
            color=color,
            markersize=4,
        )

    plt.xlabel("Days Since First Mention")
    plt.ylabel("Entropy (bits)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_temporal_entropy_small(df: pd.DataFrame, entities, output_path: Path, n_samples: int = 25):
    """Create visualization of entropy evolution over time for randomly selected high-entropy entities."""
    plt.figure(figsize=(15, 8))

    # Ensure reproducible sampling
    high_entropy_entities = df[df["entropy"] >= df["entropy"].quantile(0.9)]
    sampled_indices = high_entropy_entities.sample(n=n_samples, random_state=7)["entity_idx"].tolist()

    # Create color palette from viridis
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sampled_indices)))

    # Calculate and plot temporal entropy for each sampled entity
    for rank, (ent_idx, color) in enumerate(zip(sampled_indices, colors)):
        entity = entities[ent_idx]
        entity_dates = sorted(set(fact.created_at.date() for fact in entity.facts))
        if not entity_dates:  # Skip if no dates available
            continue
        entity_earliest_date = min(entity_dates)

        temporal_entropy = calculate_temporal_entropy(entity, entity_dates)

        # Convert dates to days since this entity's first mention
        days_since_start = [(date - entity_earliest_date).days for date in temporal_entropy.keys()]
        entropy_values = list(temporal_entropy.values())

        plt.plot(
            days_since_start,
            entropy_values,
            "-o",
            label=f"Entity {rank + 1} (H={df[df['entity_idx'] == ent_idx]['entropy'].iloc[0]:.2f})",
            color=color,
            markersize=4,
        )

    plt.xlabel("Days Since First Mention")
    plt.ylabel("Entropy (bits)")
    plt.title("Evolution of Entity Entropy Over Time (Random Selection from Top 10%)")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_early_vs_final_entropy(df: pd.DataFrame, entities, output_path: Path):
    """Compare early entropy growth (first 30 days) vs final entropy at 270 days."""
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

    plt.figure(figsize=(12, 8))

    # Find the earliest date in the dataset
    earliest_date = min(min(fact.created_at.date() for fact in entity.facts) for entity in entities if entity.facts)

    # Only consider entities created within first 90 days of data
    cutoff_date = earliest_date + pd.Timedelta(days=90)

    early_growth = []
    final_entropies = []
    total_entities = 0
    stable_entities = 0

    for _, row in df.iterrows():
        entity = next((e for e in entities if e.name == row["name"]), None)
        if not entity or len(entity.facts) < 2:
            continue

        # Get creation date and check if it's within our window
        creation_date = min(fact.created_at.date() for fact in entity.facts)
        if creation_date > cutoff_date:
            continue

        # Check if we have enough history (at least 180 days)
        if (max(fact.created_at.date() for fact in entity.facts) - creation_date).days < 90:
            continue

        total_entities += 1

        # Get dates and calculate entropies
        entity_dates = sorted(set(fact.created_at.date() for fact in entity.facts))
        temporal_entropy = calculate_temporal_entropy(entity, entity_dates)
        if not temporal_entropy:
            continue

        # Convert dates to days since entity's first mention
        days = [(date - creation_date).days for date in temporal_entropy.keys()]
        values = list(temporal_entropy.values())

        # Get entropy at 10 days and 90 days
        early_entropy = next((v for d, v in zip(days, values) if d >= 10), values[-1])
        final_entropy = next((v for d, v in zip(days, values) if d >= 90), values[-1])

        if early_entropy > 0:
            if abs(final_entropy - early_entropy) < 1e-6:
                stable_entities += 1
            early_growth.append(early_entropy)
            final_entropies.append(final_entropy)

    if not early_growth:  # If no entities meet our criteria
        plt.text(
            0.5,
            0.5,
            "No entities found with sufficient history\n(need ≥90 days and creation within first 90 days)",
            transform=plt.gca().transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.8),
        )
        plt.xlabel("10-Day Entropy (bits)")
        plt.ylabel("90-Day Entropy (bits)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    # Convert to numpy arrays for easier filtering
    early_growth = np.array(early_growth)
    final_entropies = np.array(final_entropies)

    # Create scatter plot
    plt.scatter(early_growth, final_entropies, alpha=0.5, color=plt.cm.viridis(0.3))

    # Add diagonal line for reference
    max_val = max(max(early_growth), max(final_entropies))
    plt.plot([0, max_val], [0, max_val], "--", color="gray", alpha=0.5, label="Diagonal reference line")

    # Add trend line for all points
    z = np.polyfit(early_growth, final_entropies, 1)
    p = np.poly1d(z)
    plt.plot(
        sorted(early_growth),
        p(sorted(early_growth)),
        "-",
        color=plt.cm.viridis(0.8),
        alpha=0.8,
        label="All entities trend",
    )

    # Add trend line for entities with change
    mask = abs(final_entropies - early_growth) > 1e-6
    if np.any(mask):
        z_change = np.polyfit(early_growth[mask], final_entropies[mask], 1)
        p_change = np.poly1d(z_change)
        plt.plot(
            sorted(early_growth),
            p_change(sorted(early_growth)),
            "--",
            color=plt.cm.viridis(0.5),
            alpha=0.8,
            label="Growing entities trend",
        )

    # Calculate correlations
    corr_all = np.corrcoef(early_growth, final_entropies)[0, 1]
    corr_changing = np.corrcoef(early_growth[mask], final_entropies[mask])[0, 1] if np.any(mask) else 0

    # Add statistics text
    stable_pct = (stable_entities / total_entities) * 100 if total_entities > 0 else 0
    stats_text = (
        f"All entities correlation: {corr_all:.2f}\n"
        f"Growing entities correlation: {corr_changing:.2f}\n"
        f"Stable after 10 days: {stable_pct:.1f}% ({stable_entities}/{total_entities})\n"
        f"(Among entities with ≥90 days history)"
    )

    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    plt.xlabel("10-Day Entropy (bits)")
    plt.ylabel("90-Day Entropy (bits)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def log_summary_statistics(df: pd.DataFrame):
    """Log summary statistics about entity entropy analysis."""
    logger.info("\nEntity Entropy Analysis Summary:")
    logger.info(f"Total entities analyzed: {len(df)}")
    logger.info("\nEntropy Statistics:")
    logger.info(f"Mean entropy: {df['entropy'].mean():.2f} bits")
    logger.info(f"Median entropy: {df['entropy'].median():.2f} bits")
    logger.info(f"Std dev entropy: {df['entropy'].std():.2f} bits")

    logger.info("\nCoverage Statistics:")
    logger.info(f"Median docs for 95% coverage: {df['coverage_docs'].median():.1f}")
    logger.info(f"90th percentile docs needed: {df['coverage_docs'].quantile(0.9):.1f}")

    logger.info("\nBy Category:")
    category_stats = df.groupby("type").agg({"entropy": ["mean", "median", "std", "count"]}).round(2)
    logger.info("\n" + str(category_stats))
