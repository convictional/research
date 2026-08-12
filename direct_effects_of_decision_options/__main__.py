import asyncio

import src.main as src_main


async def main():
    await src_main.main()


if __name__ == "__main__":
    """
    This is the main entrypoint for the direct_effects_of_decision_options experiment.
    Run this using `make run_experiment ARGS="direct_effects_of_decision_options"`
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(main())
