import logging
import sys
from typing import Union

# The logger to be used throughout the program
LOG = logging.getLogger("dev.desk_rejection_system")


def configure_logging(level: Union[str, int]):
    """
    Configure the logger to be used by throughout the program. See the LOG variable in this module.

    :param level: The log level (name or numeric level) to use. Name must be one of CRITICAL / FATAL (50),
                  ERROR (40), WARN / WARNING (30), INFO (20), DEBUG (10), or NOTSET (0).
    """
    if isinstance(level, str) and hasattr(logging, level):
        log_level = getattr(logging, level)
    else:
        log_level = level

    if not (isinstance(log_level, int) and logging.NOTSET <= log_level <= logging.CRITICAL):
        print(f"Invalid log level '{level}', defaulting to WARNING (f{logging.WARNING})", file=sys.stderr)
        log_level = logging.WARNING

    log_formatter = logging.Formatter("%(relativeCreated)6d %(process)d %(threadName)s %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    stdout_handler.setLevel(log_level)

    LOG.setLevel(log_level)
    LOG.addHandler(stdout_handler)
    LOG.debug("Initialized logging framework with level %s", log_level)
