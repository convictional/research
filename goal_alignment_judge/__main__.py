import asyncio

import src.main as src_main


async def main():
    await src_main.main()


if __name__ == "__main__":
    """
    This is the main entrypoint for the goal_alignment_judge experiment.
    Run this using `make run_experiment ARGS="goal_alignment_judge dspy_pipeline --method gepa"` or
    `make run_experiment ARGS="goal_alignment_judge ablation_train_size --repetitions 3"`
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(main())
