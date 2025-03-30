import os

import logfire
from dotenv import load_dotenv
from loguru import logger

# Configure Loguru
logger.remove(0)  # Remove default console logger
logger.add(
    __name__.split(".")[0] + ".log",
    rotation="500 MB",
    level="DEBUG",
)

# Configure Logfire
load_dotenv()
logfire.configure(
    token=os.environ.get("LOGFIRE_TOKEN"),
    send_to_logfire=True,
)
