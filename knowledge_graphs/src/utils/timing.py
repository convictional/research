import time
from typing import Tuple, Any


def time_function_calls(*args) -> Tuple[Any, float]:
    """
    Wrap potentially multiple function calls in a timer and return the result and the time taken to execute the functions.
    Note, will only return the result of the last function call when multiple functions are passed.
    """
    start_time = time.perf_counter()
    for func in args:
        result = func()
    end_time = time.perf_counter()

    return result, end_time - start_time


async def atime_coroutine_call(coroutine) -> Tuple[Any, float]:
    """
    Wrap an async coroutine in a timer and return the result and the time taken to execute the coroutine.
    """
    start_time = time.perf_counter()
    result = await coroutine
    end_time = time.perf_counter()

    return result, end_time - start_time
