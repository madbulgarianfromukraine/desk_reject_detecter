import os
from multiprocessing import cpu_count


def num_workers() -> int:
    """
    Returns the number of workers for tasks executed in parallel.

    :return: The number of workers to use
    """
    cpus = __get_core_count()
    return cpus * 2  # Our tasks are very much IO bound


def __get_core_count() -> int:
    """
    Returns the numer of available CPU cores in the system.

    :return: The number of available CPU cores
    """
    try:
        # NOTE: only available on some Unix platforms
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return cpu_count()
