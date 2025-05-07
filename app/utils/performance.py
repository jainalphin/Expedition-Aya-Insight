import time
import functools
import logging
import logging.config
import sys
from typing import Callable, Any, TypeVar, cast

# TypeVar for better type hints
F = TypeVar('F', bound=Callable[..., Any])


def timeit(func: F) -> F:
    """
    Decorator that logs the execution time of the decorated function.

    Args:
        func: The function to be timed

    Returns:
        The wrapped function with timing functionality
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        logger = logging.getLogger(func.__module__)
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")

        return result

    return cast(F, wrapper)
