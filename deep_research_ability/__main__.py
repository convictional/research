import asyncio
import sys

import src.main as src_main


async def main():
    await src_main.main()


if __name__ == "__main__":
    """
    This is the main entrypoint for the deep_research_ability experiment.
    Run this using:
      `make run_experiment ARGS="deep_research_ability [OPTIONS]"`

    Available options for research:
      --topic "Your research topic here"  (required for research)
      --breadth 1-10                      (default: 4)
      --depth 1-5                         (default: 2)
      --sources internal,github,google_drive  (default: internal)

    Visualization of existing research:
      --visualize_csv "filename.csv"      (visualize an existing CSV file from src/output/)

    Examples:
      `make run_experiment ARGS="deep_research_ability --topic 'AI impact on business decisions' --depth 3"`
      `make run_experiment ARGS="deep_research_ability --visualize_csv Research_Report_summary_research_data.csv"`
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(main())
