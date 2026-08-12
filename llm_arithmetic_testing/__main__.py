import asyncio

import src.main as src_main


async def main():
    await src_main.main()


if __name__ == "__main__":
    """
    This is the main entrypoint for the llm_arithmetic_testing experiment.
    Run this using `make run_experiment ARGS="llm_arithmetic_testing"`
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(main())
