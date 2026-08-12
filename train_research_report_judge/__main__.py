import asyncio

import src.main as src_main


async def main():
    await src_main.main()


if __name__ == "__main__":
    """
    Train an LLM judge for research report quality scoring.
    Run this using:
      `make run_experiment ARGS="train_research_report_judge <command>"`

    Available commands:
      load_data              Load CSV and create train/dev/test splits
      discover_rubric        Analyze scoring patterns to build a quality rubric
      evaluate_scorer        Score reports and compute alignment metrics
      analyze_disagreements  Analyze cases where scorer disagrees with expert
      full_pipeline          Run all phases end-to-end with iteration
      export_service         Export final scorer config for reuse
      score                  Score a single report

    Examples:
      `make run_experiment ARGS="train_research_report_judge load_data"`
      `make run_experiment ARGS="train_research_report_judge evaluate_scorer --split dev"`
      `make run_experiment ARGS="train_research_report_judge full_pipeline"`
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(main())
