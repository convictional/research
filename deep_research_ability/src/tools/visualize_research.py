#!/usr/bin/env python
"""
Tool to create tree visualizations from existing research CSV files.

Usage:
    python -m src.tools.visualize_research path/to/research_data.csv
"""

import sys
import logging
from pathlib import Path

from deep_research_ability.src.deep_research.tree_visualizer import create_tree_visualization_from_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("visualize_research")

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def main():
    """Main entry point."""
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.visualize_research path/to/research_data.csv")
        return 1

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return 1

    # Output directory (same as CSV file)
    output_dir = csv_path.parent

    try:
        # Create visualizations
        json_path, viz_path = create_tree_visualization_from_csv(csv_path, output_dir)

        # Display results
        print("\nVisualization created successfully!")
        print(f"Tree visualization: {viz_path}")
        print(f"JSON structure: {json_path}")
        return 0

    except Exception as e:
        logger.error(f"Error visualizing research data: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
