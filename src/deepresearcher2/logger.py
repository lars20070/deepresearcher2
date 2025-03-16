from loguru import logger

logger.remove(0)  # Remove default console logger
logger.add(__name__.split(".")[0] + ".log", rotation="500 MB")
