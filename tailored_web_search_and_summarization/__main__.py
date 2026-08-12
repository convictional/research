import asyncio

import src.main as src_main


async def main():
    await src_main.main()


if __name__ == "__main__":
    """
    This is the main entrypoint for the tailored_web_search_and_summarization experiment.
    Run this using `make run_experiment ARGS="tailored_web_search_and_summarization"`
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(main())
