"""
Retry mechanism with exponential backoff for API calls.

Simple and focused retry decorator for handling transient failures
when calling the Google Gemini API.
"""

import time
import functools
from typing import Callable, Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from google.api_core.exceptions import TooManyRequests, InternalServerError, ServiceUnavailable
from core.log import LOG


class RateLimitError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, original_error: Exception):
        self.original_error = original_error
        self.message = message
        super().__init__(message)


def _should_retry(exception: Exception) -> bool:
    """
    Determine if an exception warrants a retry.

    Retries on rate limit (429), server errors (5xx), and transient failures.
    """
    if isinstance(exception, TooManyRequests):
        return True
    if isinstance(exception, (InternalServerError, ServiceUnavailable)):
        return True
    if hasattr(exception, "status_code") and exception.status_code // 100 >= 4:
        return True
    if "429" in str(exception) or "rate limit" in str(exception).lower():
        return True
    return False


def retry_with_backoff(func: Callable) -> Callable:
    """
    Decorator that adds retry logic with exponential backoff to a function.

    Retries up to 3 times with exponential backoff (1-2 seconds initial, up to 60 seconds max).
    Raises RateLimitError if all attempts fail.

    :param func: The function to wrap with retry logic.
    :return: A wrapped function with retry capabilities.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        return func(*args, **kwargs)

    original_wrapper = wrapper

    @functools.wraps(func)
    def error_handling_wrapper(*args, **kwargs) -> Any:
        attempt = 0
        last_error: Optional[Exception] = None

        try:
            while attempt < 3:
                attempt += 1
                try:
                    result = original_wrapper(*args, **kwargs)
                    if attempt > 1:
                        LOG.info(f"✓ Succeeded on attempt {attempt}")
                    return result
                except RetryError as e:
                    last_error = e.last_attempt.exception()
                    if _should_retry(last_error):
                        LOG.warning(
                            f"Attempt {attempt}/3 failed with {type(last_error).__name__}: {str(last_error)[:100]}. "
                            f"Retrying with exponential backoff..."
                        )
                    else:
                        raise last_error
                except Exception as e:
                    last_error = e
                    if _should_retry(e):
                        LOG.warning(
                            f"Attempt {attempt}/3 failed with {type(e).__name__}: {str(e)[:100]}. "
                            f"Retrying with exponential backoff..."
                        )
                        if attempt < 3:
                            backoff_time = min(2 ** (attempt - 1), 60)
                            time.sleep(backoff_time)
                    else:
                        raise

        except Exception as final_error:
            last_error = final_error
            LOG.error(
                f"✗ All 3 retry attempts failed. Final error: {type(final_error).__name__}: {str(final_error)}"
            )
            raise RateLimitError(
                message=f"Failed after 3 retry attempts: {type(final_error).__name__}",
                original_error=final_error,
            )

        if last_error:
            raise RateLimitError(
                message=f"Failed after 3 retry attempts", original_error=last_error
            )

    return error_handling_wrapper
