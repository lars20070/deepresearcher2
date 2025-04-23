import logfire
from loguru import logger

from . import (
    config,
    env,  # noqa: F401
)

# Configure Logfire
logfire.configure(
    token=config.logfire_token,
    send_to_logfire=True,
)

# Configure Loguru
logger.remove(0)  # Remove default console logger
logger.add(
    __name__.split(".")[0] + ".log",
    rotation="500 MB",
    level="DEBUG",
)

# Logfire as sink for Loguru
# i.e. emit a Logfire log for every Loguru log
# logger.configure(handlers=[logfire.loguru_handler()])
