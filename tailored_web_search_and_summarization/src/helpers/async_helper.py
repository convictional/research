import asyncio
from typing import Any, Awaitable
from tqdm import tqdm


async def limited_task(task: Awaitable[Any], semaphore: asyncio.Semaphore, delay_between_tasks: float):
    async with semaphore:
        await asyncio.sleep(delay_between_tasks)
        return await task


async def wrap_task_progress_bar(task: Any, pbar: tqdm) -> Any:
    result = await task
    pbar.update(1)
    return result


async def execute_tasks_with_manual_pbar(tasks: Any):
    pbar = tqdm(total=len(tasks), desc="Executing tasks...")
    wrapped_tasks = [wrap_task_progress_bar(task, pbar) for task in tasks]
    results = await asyncio.gather(*wrapped_tasks)
    pbar.close()

    return results
