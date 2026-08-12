from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from pathlib import Path
import pytz
from typing import Dict, Any

from ..helpers.io import load_checkpoint
from ..settings import settings, logger
from ..build_knowledge_store.get_context import query_bq

# Organization ID from settings
ORGANIZATION_ID = settings.organization_id

# Query to retrieve source content types from BigQuery
BQ_CONTENT_TYPE_QUERY = """
SELECT
    id,
    content_type
FROM
    `${GCP_PROJECT}.cloudsql_decide_public.content`
WHERE
    organization_id = '{}'
"""


def analyze_fact_docs_distribution():
    """
    Analyze how facts and entities are distributed within documents.

    This analysis examines:
    1. How many facts a typical document holds
    2. How fact distribution varies by document type
    3. How many entities are typically contained in a document
    4. How entity distribution varies by document type
    """
    # Load the final entities from checkpoint
    entities = load_checkpoint("entity_store_final_entities")
    if not entities:
        raise ValueError("No checkpoint found")

    # Define cutoff date for recent documents (6 months ago) with UTC timezone
    cutoff_date = datetime.now(pytz.UTC) - timedelta(days=180)

    # Build document-centric data structures
    doc_facts = defaultdict(list)  # source_id -> list of facts
    doc_entities = defaultdict(set)  # source_id -> set of entity names

    # Track all facts and their source documents
    for entity in entities:
        for fact in entity.facts:
            doc_facts[fact.source_id].append(fact)
            doc_entities[fact.source_id].add(entity.name)

    # Get document types from BigQuery
    logger.info(f"Found {len(doc_facts)} unique source documents")

    # Query BigQuery for all source types
    try:
        logger.info("Fetching document types from BigQuery...")
        source_types_df = query_bq(BQ_CONTENT_TYPE_QUERY.format(ORGANIZATION_ID))
        logger.info(f"Retrieved {len(source_types_df)} document types from BigQuery")

        # Convert to dictionary for faster lookups
        # source_id_to_content_type_map: Maps document source IDs (str) to their content types (str)
        # For example: {"doc_123": "Meeting", "doc_456": "GitHub"}
        source_id_to_content_type_map = {}
        for _, row in source_types_df.iterrows():
            try:
                # The source_id might be stored as string in BigQuery
                source_id = row["id"]
                source_id_to_content_type_map[source_id] = row["content_type"]
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue
    except Exception as e:
        logger.error(f"Error querying BigQuery: {e}")
        source_id_to_content_type_map = {}

    # Create a mapping of document IDs to types
    # doc_types: Maps document IDs to their content types, using either BigQuery data or regex extraction
    doc_types = {}
    missing_types = 0

    # For each document, try to get its type from the BigQuery results
    for source_id in doc_facts.keys():
        str_source_id = str(source_id)

        if str_source_id in source_id_to_content_type_map:
            doc_types[source_id] = source_id_to_content_type_map[str_source_id]
        else:
            # Fallback to regex-based extraction
            doc_types[source_id] = extract_doc_type(str_source_id)
            missing_types += 1

    if missing_types > 0:
        logger.info(f"Used fallback type extraction for {missing_types} documents not found in BigQuery")

    # Create separate structures for recent documents
    recent_doc_facts = {}
    recent_doc_entities = {}

    for doc_id, facts in doc_facts.items():
        # Check if any facts in this document are recent
        recent_facts = [f for f in facts if f.created_at.replace(tzinfo=pytz.UTC) >= cutoff_date]
        if recent_facts:
            recent_doc_facts[doc_id] = recent_facts
            recent_doc_entities[doc_id] = doc_entities[doc_id]

    # Analyze document distribution patterns
    doc_stats = analyze_doc_distributions(doc_facts, doc_entities, doc_types)
    recent_doc_stats = analyze_doc_distributions(recent_doc_facts, recent_doc_entities, doc_types)

    # Analyze by document type
    doc_type_stats = analyze_doc_types(doc_facts, doc_entities, doc_types)
    recent_type_stats = analyze_doc_types(recent_doc_facts, recent_doc_entities, doc_types)

    # Save results
    output_dir = settings.output_path / "fact_docs_distribution_analysis"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Saving analysis results to: {output_dir}")

    # Save document statistics
    save_doc_stats(doc_stats, output_dir / "document_statistics.csv")
    save_doc_stats(recent_doc_stats, output_dir / "recent_document_statistics.csv")

    # Save document type statistics
    save_doc_type_stats(doc_type_stats, output_dir / "document_type_statistics.csv")
    save_doc_type_stats(recent_type_stats, output_dir / "recent_document_type_statistics.csv")

    # Generate visualizations
    create_fact_distribution_plots(doc_stats, doc_type_stats, output_dir)
    create_entity_distribution_plots(doc_stats, doc_type_stats, output_dir)

    return doc_stats, doc_type_stats


def extract_doc_type(source_id: str) -> str:
    """
    Extract document type from source_id string.
    This is a heuristic function and may need adjustment based on actual source_id formats.
    """
    # Common document type patterns - modify based on actual data
    patterns = {
        r"github|issue|pull|pr": "GitHub",
        r"google|docs|sheets|slides": "Google Workspace",
        r"meeting|transcript|zoom|teams": "Meeting",
        r"slack|message": "Slack",
        r"email": "Email",
        r"decide|app": "Decide App",
    }

    source_id_lower = source_id.lower()
    for pattern, doc_type in patterns.items():
        if re.search(pattern, source_id_lower):
            return doc_type

    return "Other"


def calculate_percentile_stats(values: list) -> Dict:
    """
    Calculate common statistical measures for an array of values.

    Args:
        values: List of numeric values to analyze

    Returns:
        Dictionary with statistical measures (min, percentiles, mean, max, std)
    """
    values_array = np.array(values)

    return {
        "min": np.min(values_array),
        "25th": np.percentile(values_array, 25),
        "median": np.median(values_array),
        "mean": np.mean(values_array),
        "75th": np.percentile(values_array, 75),
        "90th": np.percentile(values_array, 90),
        "95th": np.percentile(values_array, 95),
        "max": np.max(values_array),
        "std": np.std(values_array),
    }


def analyze_doc_distributions(doc_facts: Dict, doc_entities: Dict, doc_types: Dict) -> Dict:
    """
    Analyze the distribution of facts and entities across documents.

    Args:
        doc_facts: Dictionary mapping document IDs to lists of facts
        doc_entities: Dictionary mapping document IDs to sets of entity names
        doc_types: Dictionary mapping document IDs to document types

    Returns:
        Dictionary containing statistics about documents
    """
    if not doc_facts:
        return {}

    # Count facts and entities per document
    # facts_per_doc: Maps document IDs to the count of facts in each document
    facts_per_doc = {doc_id: len(facts) for doc_id, facts in doc_facts.items()}

    # entities_per_doc: Maps document IDs to the count of unique entities in each document
    entities_per_doc = {doc_id: len(entities) for doc_id, entities in doc_entities.items()}

    # Calculate fact-to-entity ratios per document (where entities exist)
    # fact_entity_ratios: Maps document IDs to the ratio of facts to entities in each document
    fact_entity_ratios = {
        doc_id: len(facts) / len(doc_entities[doc_id])
        for doc_id, facts in doc_facts.items()
        if len(doc_entities[doc_id]) > 0
    }

    # Calculate overall statistics
    total_docs = len(doc_facts)
    total_facts = sum(len(facts) for facts in doc_facts.values())
    unique_entities = set()
    for entities in doc_entities.values():
        unique_entities.update(entities)

    # Get percentile distributions using the helper function
    facts_percentiles = calculate_percentile_stats(list(facts_per_doc.values()))
    entities_percentiles = calculate_percentile_stats(list(entities_per_doc.values()))
    ratio_percentiles = calculate_percentile_stats(list(fact_entity_ratios.values()))

    # Return compiled statistics
    return {
        "total_documents": total_docs,
        "total_facts": total_facts,
        "total_unique_entities": len(unique_entities),
        "facts_per_doc": facts_per_doc,
        "entities_per_doc": entities_per_doc,
        "fact_entity_ratios": fact_entity_ratios,
        "facts_percentiles": facts_percentiles,
        "entities_percentiles": entities_percentiles,
        "ratio_percentiles": ratio_percentiles,
        "avg_facts_per_doc": total_facts / total_docs if total_docs > 0 else 0,
        "avg_entities_per_doc": len(unique_entities) / total_docs if total_docs > 0 else 0,
        "doc_types_count": {
            doc_type: list(doc_types.values()).count(doc_type) for doc_type in set(doc_types.values())
        },
    }


def analyze_doc_types(doc_facts: Dict, doc_entities: Dict, doc_types: Dict) -> Dict:
    """
    Analyze distribution of facts and entities by document type.

    Args:
        doc_facts: Dictionary mapping document IDs to lists of facts
        doc_entities: Dictionary mapping document IDs to sets of entity names
        doc_types: Dictionary mapping document IDs to document types

    Returns:
        Dictionary containing statistics grouped by document type
    """
    if not doc_facts or not doc_types:
        return {}

    # Group documents by type
    docs_by_type = defaultdict(list)
    for doc_id in doc_facts.keys():
        doc_type = doc_types.get(doc_id, "Unknown")
        docs_by_type[doc_type].append(doc_id)

    # Calculate statistics per document type
    type_stats = {}

    for doc_type, doc_ids in docs_by_type.items():
        # Skip if no documents of this type
        if not doc_ids:
            continue

        # Count total facts and entities
        facts_count = sum(len(doc_facts[doc_id]) for doc_id in doc_ids)
        unique_entities = set()
        for doc_id in doc_ids:
            unique_entities.update(doc_entities[doc_id])

        # Calculate facts and entities per document for this type
        facts_per_doc = [len(doc_facts[doc_id]) for doc_id in doc_ids]
        entities_per_doc = [len(doc_entities[doc_id]) for doc_id in doc_ids]

        # Calculate fact-to-entity ratios per document (where entities exist)
        fact_entity_ratios = [
            len(doc_facts[doc_id]) / len(doc_entities[doc_id]) for doc_id in doc_ids if len(doc_entities[doc_id]) > 0
        ]

        type_stats[doc_type] = {
            "doc_count": len(doc_ids),
            "total_facts": facts_count,
            "unique_entities": len(unique_entities),
            "facts_per_doc_avg": np.mean(facts_per_doc),
            "facts_per_doc_median": np.median(facts_per_doc),
            "facts_per_doc_std": np.std(facts_per_doc),
            "facts_per_doc_max": np.max(facts_per_doc),
            "entities_per_doc_avg": np.mean(entities_per_doc),
            "entities_per_doc_median": np.median(entities_per_doc),
            "entities_per_doc_std": np.std(entities_per_doc),
            "entities_per_doc_max": np.max(entities_per_doc),
            "fact_entity_ratio_avg": np.mean(fact_entity_ratios) if fact_entity_ratios else 0,
            "fact_entity_ratio_median": np.median(fact_entity_ratios) if fact_entity_ratios else 0,
            "percent_of_total_docs": len(doc_ids) / len(doc_facts) * 100,
            "percent_of_total_facts": facts_count / sum(len(facts) for facts in doc_facts.values()) * 100,
        }

    return type_stats


def save_doc_stats(doc_stats: Dict, output_path: Path):
    """Save document statistics to CSV file."""
    if not doc_stats:
        logger.warning(f"No document statistics to save to {output_path}")
        return

    # Basic statistics for summary file
    summary_data = {
        "metric": [
            "Total Documents",
            "Total Facts",
            "Total Unique Entities",
            "Average Facts per Document",
            "Median Facts per Document",
            "Max Facts per Document",
            "Average Entities per Document",
            "Median Entities per Document",
            "Max Entities per Document",
            "Average Fact-Entity Ratio",
            "Median Fact-Entity Ratio",
        ],
        "value": [
            doc_stats["total_documents"],
            doc_stats["total_facts"],
            doc_stats["total_unique_entities"],
            doc_stats["facts_percentiles"]["mean"],
            doc_stats["facts_percentiles"]["median"],
            doc_stats["facts_percentiles"]["max"],
            doc_stats["entities_percentiles"]["mean"],
            doc_stats["entities_percentiles"]["median"],
            doc_stats["entities_percentiles"]["max"],
            doc_stats["ratio_percentiles"]["mean"],
            doc_stats["ratio_percentiles"]["median"],
        ],
    }

    # Add document type counts
    for doc_type, count in doc_stats["doc_types_count"].items():
        summary_data["metric"].append(f"Documents of type: {doc_type}")
        summary_data["value"].append(count)

    pd.DataFrame(summary_data).to_csv(output_path, index=False)

    # Also save detailed document data
    detailed_path = output_path.with_stem(output_path.stem + "_detailed")

    detailed_rows = []
    for doc_id in doc_stats["facts_per_doc"].keys():
        detailed_rows.append(
            {
                "document_id": str(doc_id),
                "facts_count": doc_stats["facts_per_doc"].get(doc_id, 0),
                "entities_count": doc_stats["entities_per_doc"].get(doc_id, 0),
                "fact_entity_ratio": doc_stats["fact_entity_ratios"].get(doc_id, 0),
            }
        )

    pd.DataFrame(detailed_rows).to_csv(detailed_path, index=False)


def save_doc_type_stats(doc_type_stats: Dict, output_path: Path):
    """Save document type statistics to CSV file."""
    if not doc_type_stats:
        logger.warning(f"No document type statistics to save to {output_path}")
        return

    # Convert nested dictionary to dataframe
    rows = []
    for doc_type, stats in doc_type_stats.items():
        row = {"document_type": doc_type}
        row.update(stats)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def create_fact_distribution_plots(doc_stats: Dict, doc_type_stats: Dict, output_dir: Path):
    """Create visualizations for fact distribution across documents."""
    if not doc_stats or not doc_type_stats:
        logger.warning("Insufficient data for creating fact distribution plots")
        return

    # 1. Histogram of facts per document
    plt.figure(figsize=(12, 8))
    # facts_per_doc values converted to array for plotting
    facts_array = np.array(list(doc_stats["facts_per_doc"].values()))

    # Plot histogram with log scale for x-axis
    plt.hist(
        facts_array,
        bins=50,
        alpha=0.7,
        color="#7B2B8B",
        edgecolor="black",
    )

    plt.grid(True, which="major", ls="-", alpha=0.2)
    plt.xlabel("Number of Facts per Document")
    plt.ylabel("Number of Documents")
    plt.title("Distribution of Facts per Document")

    # Add statistics text box
    stats_text = (
        f"Mean: {doc_stats['facts_percentiles']['mean']:.1f}\n"
        f"Median: {doc_stats['facts_percentiles']['median']:.1f}\n"
        f"Max: {doc_stats['facts_percentiles']['max']:.0f}"
    )
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
    plt.savefig(output_dir / "facts_per_document_histogram.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # 2. Bar chart comparing facts per document by document type
    plt.figure(figsize=(14, 10))

    # Sort types by average facts per document (descending)
    sorted_types = sorted(doc_type_stats.items(), key=lambda x: x[1]["facts_per_doc_avg"], reverse=True)

    doc_types = [t[0] for t in sorted_types]
    avg_facts = [t[1]["facts_per_doc_avg"] for t in sorted_types]
    median_facts = [t[1]["facts_per_doc_median"] for t in sorted_types]

    x = np.arange(len(doc_types))
    width = 0.35

    plt.bar(x - width / 2, avg_facts, width, label="Mean", color="#1f77b4", alpha=0.7)
    plt.bar(x + width / 2, median_facts, width, label="Median", color="#ff7f0e", alpha=0.7)

    plt.xlabel("Document Type")
    plt.ylabel("Facts per Document")
    plt.title("Facts per Document by Document Type")
    plt.xticks(x, doc_types, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    # Add document counts
    for i, doc_type in enumerate(doc_types):
        count = doc_type_stats[doc_type]["doc_count"]
        plt.text(i, 0.5, f"n={count}", ha="center", va="bottom", color="black")

    plt.tight_layout()
    plt.savefig(output_dir / "facts_by_document_type.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # 3. Scatter plot showing relationship between document fact count and entity count
    plt.figure(figsize=(12, 10))

    facts = np.array(list(doc_stats["facts_per_doc"].values()))
    entities = np.array(list(doc_stats["entities_per_doc"].values()))

    # Create scatter plot
    plt.scatter(facts, entities, alpha=0.5, s=20, color="#7B2B8B")

    # Add trend line
    z = np.polyfit(facts, entities, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(facts), p(np.sort(facts)), "r--", alpha=0.8)

    plt.xlabel("Facts per Document")
    plt.ylabel("Entities per Document")
    plt.title("Relationship Between Facts and Entities per Document")
    plt.grid(True, alpha=0.2)

    # Add correlation coefficient
    corr = np.corrcoef(facts, entities)[0, 1]
    plt.text(
        0.05,
        0.95,
        f"Correlation: {corr:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "facts_entities_relationship.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_entity_distribution_plots(doc_stats: Dict, doc_type_stats: Dict, output_dir: Path):
    """Create visualizations for entity distribution across documents."""
    if not doc_stats or not doc_type_stats:
        logger.warning("Insufficient data for creating entity distribution plots")
        return

    # 1. Histogram of entities per document
    plt.figure(figsize=(12, 8))
    # entities_per_doc values converted to array for plotting
    entities_array = np.array(list(doc_stats["entities_per_doc"].values()))

    plt.hist(
        entities_array,
        bins=50,
        alpha=0.7,
        color="#2C7BB6",
        edgecolor="black",
    )

    plt.grid(True, which="major", ls="-", alpha=0.2)
    plt.xlabel("Number of Entities per Document")
    plt.ylabel("Number of Documents")
    plt.title("Distribution of Entities per Document")

    # Add statistics text box
    stats_text = (
        f"Mean: {doc_stats['entities_percentiles']['mean']:.1f}\n"
        f"Median: {doc_stats['entities_percentiles']['median']:.1f}\n"
        f"Max: {doc_stats['entities_percentiles']['max']:.0f}"
    )
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
    plt.savefig(output_dir / "entities_per_document_histogram.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # 2. Bar chart comparing entities per document by document type
    plt.figure(figsize=(14, 10))

    # Sort types by average entities per document (descending)
    sorted_types = sorted(doc_type_stats.items(), key=lambda x: x[1]["entities_per_doc_avg"], reverse=True)

    doc_types = [t[0] for t in sorted_types]
    avg_entities = [t[1]["entities_per_doc_avg"] for t in sorted_types]
    median_entities = [t[1]["entities_per_doc_median"] for t in sorted_types]

    x = np.arange(len(doc_types))
    width = 0.35

    plt.bar(x - width / 2, avg_entities, width, label="Mean", color="#1f77b4", alpha=0.7)
    plt.bar(x + width / 2, median_entities, width, label="Median", color="#ff7f0e", alpha=0.7)

    plt.xlabel("Document Type")
    plt.ylabel("Entities per Document")
    plt.title("Entities per Document by Document Type")
    plt.xticks(x, doc_types, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    # Add document counts
    for i, doc_type in enumerate(doc_types):
        count = doc_type_stats[doc_type]["doc_count"]
        plt.text(i, 0.5, f"n={count}", ha="center", va="bottom", color="black")

    plt.tight_layout()
    plt.savefig(output_dir / "entities_by_document_type.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # 3. Histogram of fact-to-entity ratios
    plt.figure(figsize=(12, 8))
    # fact_entity_ratios values converted to array for plotting
    ratios_array = np.array(list(doc_stats["fact_entity_ratios"].values()))

    plt.hist(
        ratios_array,
        bins=50,
        alpha=0.7,
        color="#D95F02",
        edgecolor="black",
    )

    plt.grid(True, which="major", ls="-", alpha=0.2)
    plt.xlabel("Fact-to-Entity Ratio (Facts per Entity)")
    plt.ylabel("Number of Documents")
    plt.title("Distribution of Fact-to-Entity Ratios in Documents")

    # Add statistics text box
    stats_text = (
        f"Mean: {doc_stats['ratio_percentiles']['mean']:.2f}\n"
        f"Median: {doc_stats['ratio_percentiles']['median']:.2f}\n"
        f"Max: {doc_stats['ratio_percentiles']['max']:.2f}"
    )
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
    plt.savefig(output_dir / "fact_entity_ratio_histogram.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # 4. Bar chart comparing fact-entity ratios by document type
    plt.figure(figsize=(14, 10))

    # Sort types by average fact-entity ratio (descending)
    sorted_types = sorted(doc_type_stats.items(), key=lambda x: x[1]["fact_entity_ratio_avg"], reverse=True)

    doc_types = [t[0] for t in sorted_types]
    avg_ratios = [t[1]["fact_entity_ratio_avg"] for t in sorted_types]
    median_ratios = [t[1]["fact_entity_ratio_median"] for t in sorted_types]

    x = np.arange(len(doc_types))
    width = 0.35

    plt.bar(x - width / 2, avg_ratios, width, label="Mean", color="#1f77b4", alpha=0.7)
    plt.bar(x + width / 2, median_ratios, width, label="Median", color="#ff7f0e", alpha=0.7)

    plt.xlabel("Document Type")
    plt.ylabel("Fact-to-Entity Ratio")
    plt.title("Fact-to-Entity Ratio by Document Type")
    plt.xticks(x, doc_types, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    # Add document counts
    for i, doc_type in enumerate(doc_types):
        count = doc_type_stats[doc_type]["doc_count"]
        plt.text(i, 0.05, f"n={count}", ha="center", va="bottom", color="black")

    plt.tight_layout()
    plt.savefig(output_dir / "fact_entity_ratio_by_document_type.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
