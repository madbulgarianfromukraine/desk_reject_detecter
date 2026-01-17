import threading

from core.log import LOG
# CONSTANTS NEEDED FOR EXPONENTIAL BACKOFF
__WAITING_DEFAULT_SECONDS : int = 12
__WAITING_TIME_SECONDS: int = __WAITING_DEFAULT_SECONDS
__MAX_WAITING_SECONDS: int = 96  # Prevent waiting forever
__WAITING_TIME_LOCK: threading.Lock = threading.Lock()

def get_waiting_time() -> int:
    return __WAITING_TIME_SECONDS

def reset_waiting_time() -> None:
    with __WAITING_TIME_LOCK:
        global __WAITING_TIME_SECONDS
        __WAITING_TIME_SECONDS = __WAITING_DEFAULT_SECONDS

def double_waiting_time():
    LOG.debug("429 happened, backing off exponentially.")
    with __WAITING_TIME_LOCK:
        global __WAITING_TIME_SECONDS
        __WAITING_TIME_SECONDS = min(2 * __WAITING_TIME_SECONDS, __MAX_WAITING_SECONDS)
