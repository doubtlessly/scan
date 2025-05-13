# modules/utils.py
import asyncio
import random
from functools import wraps
from ccxt.base.errors import ExchangeError, RequestTimeout

def retry_async(
    attempts: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    exceptions: tuple = (ExchangeError, RequestTimeout),
):
    """
    Async retry decorator with exponential backoff + jitter.

    - attempts: total tries (including the first)
    - initial_delay: delay before first retry
    - backoff_factor: multiplier for delay after each failure
    - max_delay: cap on delay
    - jitter: +/- random seconds added to each sleep
    - exceptions: exception types to catch/retry
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == attempts:
                        raise
                    # compute sleep with jitter
                    sleep = min(delay, max_delay) + random.uniform(-jitter, jitter)
                    await asyncio.sleep(max(0, sleep))
                    delay *= backoff_factor
            # fallback (shouldn't reach)
            return await fn(*args, **kwargs)
        return wrapper
    return decorator
