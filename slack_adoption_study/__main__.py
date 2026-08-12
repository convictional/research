import asyncio
import sys
import logging

import src.main as src_main


async def main():
    """
    Main entrypoint for the Slack adoption study experiment.

    This experiment studies the causal impact of Slack adoption on public company
    performance using staggered difference-in-differences and other causal inference methods.
    """
    try:
        await src_main.main()
    except KeyboardInterrupt:
        logging.info("Experiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """
    Run this using: make run_experiment ARGS="slack_adoption_study"
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the experiment
    asyncio.run(main())
