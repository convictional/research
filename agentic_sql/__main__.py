import asyncio

from src.main import main

if __name__ == "__main__":
    """
    This is the main entrypoint for the agentic_sql experiment.
    Run this using: `make run_experiment ARGS="agentic_sql [OPTIONS]"`

    Options:
        --demo: Run with example questions
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(main())
