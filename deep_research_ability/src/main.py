import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List

from common.instruct_llm import set_async_instructor_client
from common.prompt_template_engine import initialize_and_register_prompt_templates
from src.deep_research.framework import ResearchProgress, research_with_report
from src.settings import settings, OPENAI_O3_MINI
from src.deep_research.get_context import InternalContentSearch, ContextSource


class TerminalProgress:
    """Simple terminal progress display."""

    def __init__(self):
        self.progress_lines = 4
        print("\n" * self.progress_lines)  # Initialize space for progress
        initialize_and_register_prompt_templates(Path(__file__).parent / "prompts")

    def update(self, progress: ResearchProgress):
        """Update progress display in terminal."""
        # Move cursor up to progress area
        sys.stdout.write(f"\x1b[{self.progress_lines}A")
        sys.stdout.write("\x1b[0J")  # Clear from cursor to end of screen

        # Print progress bars
        depth_percent = (
            ((progress.total_depth - progress.current_depth) / progress.total_depth) * 100
            if progress.total_depth
            else 0
        )
        breadth_percent = (
            ((progress.total_breadth - progress.current_breadth) / progress.total_breadth) * 100
            if progress.total_breadth
            else 0
        )
        query_percent = (progress.completed_queries / progress.total_queries) * 100 if progress.total_queries else 0

        print(f"Depth:   [{'=' * int(depth_percent / 5)}{' ' * (20 - int(depth_percent / 5))}] {depth_percent:.0f}%")
        print(
            f"Breadth: [{'=' * int(breadth_percent / 5)}{' ' * (20 - int(breadth_percent / 5))}] {breadth_percent:.0f}%"
        )
        print(f"Queries: [{'=' * int(query_percent / 5)}{' ' * (20 - int(query_percent / 5))}] {query_percent:.0f}%")
        if progress.current_query:
            print(f"Current: {progress.current_query[:60]}...")
        else:
            print()


async def get_terminal_input(prompt: str) -> str:
    """Get user input from terminal."""
    print(f"\n{prompt}")
    return input("> ").strip()


async def get_source_types() -> List[ContextSource]:
    """Get user input for content source types."""
    print("\nSelect content sources to search (comma-separated):")
    print("1. Internal (default)")
    print("2. GitHub")
    print("3. Google Drive")
    print("4. All sources")

    selection = await get_terminal_input("Enter your selection (1-4, default: 1):")

    # Default to internal content
    if not selection:
        return [ContextSource.INTERNAL]

    # Process selection
    sources = []
    if selection == "4" or "all" in selection.lower():
        return [ContextSource.INTERNAL, ContextSource.GITHUB, ContextSource.GOOGLE_DRIVE]

    options = selection.split(",")
    for option in options:
        option = option.strip()
        if option == "1" or "internal" in option.lower():
            sources.append(ContextSource.INTERNAL)
        elif option == "2" or "github" in option.lower():
            sources.append(ContextSource.GITHUB)
        elif option == "3" or "google" in option.lower():
            sources.append(ContextSource.GOOGLE_DRIVE)

    # Ensure at least internal is included if no valid options selected
    if not sources:
        sources.append(ContextSource.INTERNAL)
        sources.append(ContextSource.GITHUB)
        sources.append(ContextSource.GOOGLE_DRIVE)

    return sources


async def visualize_from_csv(csv_filename: str):
    """Visualize an existing research CSV."""
    from src.deep_research.tree_visualizer import create_tree_visualization_from_csv, analyze_research_tree  # type: ignore

    csv_path = settings.output_path / csv_filename
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print(f"Please check that the file exists in {settings.output_path}")
        return

    print(f"Visualizing research data from {csv_path}...")
    json_path, viz_path, research_json_path = create_tree_visualization_from_csv(csv_path, settings.output_path)

    # Run analysis
    analysis = analyze_research_tree(csv_path)

    # Display results
    print("\nVisualization created successfully!")
    print(f"Research D3 collapsible tree HTML: {viz_path}")
    print(f"Tree JSON structure: {json_path}")
    print(f"Complete research JSON: {research_json_path}")

    print("\nResearch Tree Analysis:")
    print(f"- Total iterations: {analysis['total_iterations']}")
    print(f"- Total queries: {analysis['total_queries']}")
    print(f"- Maximum depth reached: {analysis['max_depth']}")
    print(f"- Queries by depth: {analysis['queries_by_depth']}")
    print(f"- Average URLs per query: {analysis['avg_urls_per_query']:.2f}")
    print(f"- Average learnings per query: {analysis['avg_learnings_per_query']:.2f}")
    print(
        f"- Tree structure follows expected narrowing pattern: "
        f"{'Yes' if analysis['follows_expected_structure'] else 'No'}"
    )


async def visualize_from_json(json_filename: str):
    """Visualize an existing *research.json file (the raw research data)."""
    from src.deep_research.tree_visualizer import ResearchTree  # type: ignore

    json_path = settings.output_path / json_filename
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        print(f"Please check that the file exists in {settings.output_path}")
        return

    print(f"Loading research data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        research_data = json.load(f)

    tree = ResearchTree(research_data)
    # Save the 'flat' tree JSON for reference
    tree_json_path = tree.save_json(settings.output_path)
    research_json_path = tree.export_research_json(settings.output_path)
    # Generate the collapsible D3 HTML
    viz_path = tree.visualize(settings.output_path)

    print("\nVisualization created successfully!")
    print(f"D3 collapsible tree HTML: {viz_path}")
    print(f"Tree JSON structure: {tree_json_path}")
    print(f"Complete research JSON (re-saved): {research_json_path}")


async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Deep Research Experiment")
    parser.add_argument("--topic", type=str, help="Research topic")
    parser.add_argument("--breadth", type=int, default=4, help="Research breadth (default: 4)")
    parser.add_argument("--depth", type=int, default=2, help="Research depth (default: 2)")
    parser.add_argument(
        "--sources",
        type=str,
        default="internal",
        help="Content sources to search (comma-separated: internal,github,google_drive)",
    )
    parser.add_argument(
        "--visualize_csv", type=str, help="Visualize an existing CSV file (from src/output/) and exit."
    )
    parser.add_argument(
        "--visualize_json", type=str, help="Visualize an existing *research.json file (from src/output/) and exit."
    )

    # Parse arguments
    args = parser.parse_args()

    # If user only wants to visualize CSV or JSON, do that and exit
    if args.visualize_csv and args.visualize_json:
        parser.error("Cannot specify both --visualize_csv and --visualize_json at the same time.")

    if args.visualize_csv:
        # Just visualize from CSV and return
        await visualize_from_csv(args.visualize_csv)
        return

    if args.visualize_json:
        # Just visualize from research.json and return
        await visualize_from_json(args.visualize_json)
        return

    # Regular research mode requires a topic
    if not args.topic:
        parser.error("--topic is required if not using --visualize_csv or --visualize_json")

    # Configure logging for detailed output
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set specific modules to DEBUG level if you want more detail
    logger = logging.getLogger("deep_research_ability")
    logger.setLevel(logging.DEBUG)
    logging.getLogger("src.deep_research.get_context").setLevel(logging.DEBUG)
    logging.getLogger("src.deep_research.framework").setLevel(logging.DEBUG)

    # Initialize LLM client
    set_async_instructor_client(
        OPENAI_O3_MINI,
        settings.openai_api_key,
        settings.openai_organization,
    )

    # Use command-line arguments
    query = args.topic
    breadth = args.breadth
    depth = args.depth

    # Parse sources
    sources = []
    for source_arg in args.sources.split(","):
        s = source_arg.strip().lower()
        if s == "internal" or s == "1":
            sources.append(ContextSource.INTERNAL)
        elif s == "github" or s == "2":
            sources.append(ContextSource.GITHUB)
        elif s == "google_drive" or s == "3":
            sources.append(ContextSource.GOOGLE_DRIVE)

    # Ensure at least internal is included
    if not sources:
        sources = [ContextSource.INTERNAL]

    # Display selected parameters
    source_names = [src.name for src in sources]
    print(f"\nResearch topic: {query}")
    print(f"Breadth: {breadth}")
    print(f"Depth: {depth}")
    print(f"Selected sources: {', '.join(source_names)}")

    # Initialize progress display
    progress_display = TerminalProgress()

    # Create search provider using our implementation
    search_provider = InternalContentSearch()

    try:
        print("\nStarting research...\n")

        result = await research_with_report(
            query=query,
            breadth=breadth,
            depth=depth,
            search_provider=search_provider,
            on_progress=progress_display.update,
        )

        print("\n\nResearch completed!")
        print(f"\nReport saved as: {result['title']}.md")
        print(f"Detailed research data exported to CSV: {result['csv_path']}")
        print(f"Detailed research data exported to JSON: {result['research_json_path']}")
        print(f"Research flowchart HTML: {result['tree_viz_path']}")
        print(f"Tree JSON structure: {result['tree_json_path']}")
        print("\nKey Takeaways:")
        for takeaway in result["key_takeaways"]:
            print(f"- {takeaway}")

        print("\nSuggested Further Research:")
        for topic in result["further_research"]:
            print(f"- {topic}")

    except Exception as e:
        print(f"\nError during research: {e}")
        raise
    finally:
        # Ensure we clean up the search provider's connections
        await search_provider.cleanup()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
