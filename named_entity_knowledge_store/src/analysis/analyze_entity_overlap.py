import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
import itertools
import seaborn as sns

from ..helpers.io import load_checkpoint
from ..settings import settings, logger


def analyze_entity_document_overlap():
    """Analyze document overlap between entities and generate visualizations."""
    # Load entities from checkpoint once
    entities = load_checkpoint("entity_store_final_entities")
    if not entities:
        raise ValueError("No checkpoint found")

    # Create output directory
    output_dir = settings.output_path / "entity_overlap_analysis"
    output_dir.mkdir(exist_ok=True)

    # Get document sets for each entity
    entity_docs = {entity.name: {fact.source_id for fact in entity.facts} for entity in entities}

    # Calculate overlap data
    overlap_data = calculate_overlap_matrix(entity_docs)

    # Save raw overlap data
    save_overlap_data(overlap_data, output_dir / "entity_document_overlap.csv")

    # Generate visualizations with passed entities
    generate_visualizations(overlap_data, output_dir, entities)


def calculate_overlap_matrix(entity_docs: Dict[str, Set]) -> List[dict]:
    """Calculate document overlap between all entity pairs."""
    overlap_data = []
    isolated_entities = set(entity_docs.keys())  # Track entities with no overlaps

    # Get all entity pairs
    entity_pairs = itertools.combinations(entity_docs.keys(), 2)

    for entity1, entity2 in entity_pairs:
        docs1 = entity_docs[entity1]
        docs2 = entity_docs[entity2]

        # Calculate overlap
        shared_docs = len(docs1.intersection(docs2))

        # Create entry regardless of overlap
        overlap_data.append(
            {
                "entity1": entity1,
                "entity2": entity2,
                "shared_documents": shared_docs,
                "entity1_total_docs": len(docs1),
                "entity2_total_docs": len(docs2),
                "overlap_ratio": shared_docs / min(len(docs1), len(docs2)) if shared_docs > 0 else 0,
            }
        )

        # If there's any overlap, these entities aren't isolated
        if shared_docs > 0:
            isolated_entities.discard(entity1)
            isolated_entities.discard(entity2)

    # Log statistics about isolated entities
    total_entities = len(entity_docs)
    isolated_count = len(isolated_entities)

    if isolated_count > 0:
        logger.info(
            f"\nFound {isolated_count} entities ({isolated_count / total_entities:.1%}) with no document overlap:"
        )
        for entity in sorted(isolated_entities):
            logger.info(f"- {entity} ({len(entity_docs[entity])} documents)")

    return overlap_data


def save_overlap_data(overlap_data: List[dict], output_path: Path):
    """Save overlap data to CSV."""
    df = pd.DataFrame(overlap_data)
    df.to_csv(output_path, index=False)


def generate_visualizations(overlap_data: List[dict], output_dir: Path, entities: List):
    """Generate visualizations of the overlap data."""
    df = pd.DataFrame(overlap_data)

    # Get unique entity document counts and build fact counts per document
    entity_doc_counts = {}
    doc_fact_counts = defaultdict(int)

    # First get entity document counts
    for _, row in df.iterrows():
        if row["entity1"] not in entity_doc_counts:
            entity_doc_counts[row["entity1"]] = row["entity1_total_docs"]
        if row["entity2"] not in entity_doc_counts:
            entity_doc_counts[row["entity2"]] = row["entity2_total_docs"]

    # Count facts per unique document across all entities
    for entity in entities:
        for fact in entity.facts:
            doc_fact_counts[fact.source_id] += 1

    # Generate visualizations
    plot_docs_per_entity_histogram(entity_doc_counts, output_dir / "docs_per_entity_histogram.png")
    plot_facts_per_doc_histogram(dict(doc_fact_counts), output_dir / "facts_per_doc_histogram.png")
    plot_entity_overlap_scatter(df, entity_doc_counts, output_dir / "entity_overlap_scatter.png")
    plot_entity_overlap_heatmap(df, entity_doc_counts, output_dir / "top_100_entity_overlap_scatter.png", top_n=100)


def plot_docs_per_entity_histogram(entity_doc_counts: Dict[str, int], output_path: Path):
    """Create a histogram of the number of documents per entity."""
    plt.figure(figsize=(12, 8))

    doc_counts = np.array(list(entity_doc_counts.values()))
    median_docs = np.median(doc_counts)
    mean_docs = np.mean(doc_counts)

    plt.hist(
        doc_counts,
        bins=50,
        alpha=0.7,
        color="#7B2B8B",
        edgecolor="black",
    )

    plt.yscale("log")  # Only y-axis in log scale
    plt.grid(True, which="major", ls="-", alpha=0.2)
    plt.xlabel("Number of Documents per Entity")
    plt.ylabel("Number of Entities (log scale)")
    plt.title("Distribution of Documents per Entity")

    # Add statistics text box
    stats_text = f"Median Docs/Entity: {median_docs:.1f}\nMean Docs/Entity: {mean_docs:.1f}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="#333333", alpha=0.8, edgecolor="none", pad=1.0),
        color="white",
    )

    plt.gca().set_facecolor("#f5f5f5")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_facts_per_doc_histogram(doc_fact_counts: Dict[str, int], output_path: Path):
    """Create a histogram of the number of facts per document."""
    plt.figure(figsize=(12, 8))

    # Get the distribution of fact counts
    fact_counts = list(doc_fact_counts.values())

    plt.hist(
        fact_counts,
        bins=50,
        alpha=0.7,
        color="#7B2B8B",
        edgecolor="black",
    )

    # Add statistics
    median_facts = np.median(fact_counts)
    mean_facts = np.mean(fact_counts)

    stats_text = f"Median Facts/Doc: {median_facts:.1f}\nMean Facts/Doc: {mean_facts:.1f}"
    plt.text(
        0.98,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="#333333", alpha=0.8, edgecolor="none", pad=1.0),
        color="white",
    )

    plt.grid(True, which="major", ls="-", alpha=0.2)
    plt.xlabel("Number of Facts per Document")
    plt.ylabel("Number of Documents")
    plt.title("Distribution of Facts per Document")

    plt.gca().set_facecolor("#f5f5f5")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_entity_overlap_heatmap(
    df: pd.DataFrame, entity_doc_counts: Dict[str, int], output_path: Path, top_n: int | None = None
):
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18})
    sorted_entities = sorted(entity_doc_counts.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_entities = sorted_entities[:top_n]
    entity_to_idx = {e: i for i, (e, _) in enumerate(sorted_entities)}
    size = len(entity_to_idx)

    # Build overlap matrix
    matrix = np.zeros((size, size), dtype=int)

    # Iterate over DataFrame rows correctly
    for _, row in df.iterrows():
        e1, e2 = row["entity1"], row["entity2"]
        shared_docs = row["shared_documents"]
        if e1 in entity_to_idx and e2 in entity_to_idx:
            i, j = entity_to_idx[e1], entity_to_idx[e2]
            matrix[i, j] = shared_docs
            matrix[j, i] = shared_docs

    data = np.log10(matrix + 1)
    mask = matrix == 0

    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(
        data,
        mask=mask,  # Mask zero values
        cmap="viridis",
        square=True,
        cbar_kws={"label": "log₁₀(1 + Shared Docs)"},
        linewidths=0.0,
        center=None,
        vmin=0,
        alpha=0.7,
    )

    # Modify colorbar ticks to show epsilon
    cbar = ax.collections[0].colorbar
    ticks = cbar.get_ticks()
    tick_labels = ["0 + ε" if i == 0 else f"{tick:.1f}" for i, tick in enumerate(ticks)]
    cbar.set_ticklabels(tick_labels)

    # Reverse y-axis after plotting
    ax.invert_yaxis()

    plt.title(f"{'Top ' + str(top_n) + ' ' if top_n else ''}Entity Overlap Heatmap")
    plt.xlabel("Entity Index")
    plt.ylabel("Entity Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_entity_overlap_scatter(
    df: pd.DataFrame, entity_doc_counts: Dict[str, int], output_path: Path, top_n: int | None = None
):
    """Create a scatter plot of entity document overlap."""
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18})
    # Sort entities by number of documents
    sorted_entities = sorted(entity_doc_counts.items(), key=lambda x: x[1], reverse=True)

    # Filter for top N if specified
    if top_n:
        sorted_entities = sorted_entities[:top_n]

    entity_to_idx = {entity: idx for idx, (entity, _) in enumerate(sorted_entities)}
    entity_set = set(entity_to_idx.keys())

    # Create scatter plot data
    scatter_data = []
    for _, row in df.iterrows():
        # Only include points where both entities are in our filtered set
        if row["entity1"] in entity_set and row["entity2"] in entity_set:
            i = entity_to_idx[row["entity1"]]
            j = entity_to_idx[row["entity2"]]
            if row["shared_documents"] > 0:  # Only plot non-zero overlaps
                scatter_data.append((i, j, row["shared_documents"]))
                scatter_data.append((j, i, row["shared_documents"]))  # Mirror the point

    # Convert to numpy arrays for plotting
    if scatter_data:
        x, y, colors = zip(*scatter_data)
    else:
        x, y, colors = [], [], []

    # Create the plot
    plt.figure(figsize=(12, 12))

    # Use a higher contrast colormap and lower alpha for better visibility
    scatter = plt.scatter(
        x,
        y,
        c=np.log10(colors),
        cmap="viridis",
        s=3 if top_n else 1,  # Larger points for zoomed view
        alpha=0.7,
    )

    cbar = plt.colorbar(scatter, label="log₁₀(Shared Documents)")
    # Modify colorbar ticks to show epsilon
    ticks = cbar.get_ticks()
    tick_labels = ["0 + ε" if i == 0 else f"{tick:.1f}" for i, tick in enumerate(ticks)]
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=10)

    plt.gca().set_facecolor("#f5f5f5")
    plt.grid(True, alpha=0.2, color="gray")

    # Set axis limits and labels
    plt.xlim(-1, len(entity_to_idx))
    plt.ylim(-1, len(entity_to_idx))
    plt.xlabel("Entity Index (sorted by total documents)")
    plt.ylabel("Entity Index (sorted by total documents)")
    title_prefix = f"Top {top_n} " if top_n else ""
    plt.title(f"{title_prefix}Entity Document Overlap")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
