import asyncio
import inspect
import time
from functools import wraps

limits = {}
rps = 1.5


def rate_limiter(func):
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            last_called = limits.get(func, 0)
            elapsed = now - last_called
            wait_time = max(0, (1 / rps) - elapsed)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            limits[func] = time.time()
            return await func(*args, **kwargs)
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            last_called = limits.get(func, 0)
            elapsed = now - last_called
            wait_time = max(0, (1 / rps) - elapsed)
            if wait_time > 0:
                time.sleep(wait_time)
            limits[func] = time.time()
            return func(*args, **kwargs)

    return wrapper
