#!/usr/bin/env python3

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Configure Loguru
logger.remove(0)  # Remove default console logger
logger.add(
    __name__.split(".")[0] + ".log",
    rotation="500 MB",
    level="DEBUG",
)
