import asyncio

from decisions_to_goals.src import main as src_main


async def main():
    await src_main.main()


if __name__ == "__main__":
    """
    Run with: make run_experiment ARGS="decisions_to_goals mine_all"
    or:       make run_experiment ARGS="decisions_to_goals mine_goals --condition limited"
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
