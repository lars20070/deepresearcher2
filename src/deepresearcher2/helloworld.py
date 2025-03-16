#!/usr/bin/env python3

from deepresearcher2 import logger


class HelloWorld:
    """
    HelloWorld class
    """

    def __init__(self) -> None:
        """
        Initialize HelloWorld
        """
        logger.info("Initializing HelloWorld.")
        pass


def main() -> None:
    """
    Main function for the script.

    Returns:
        None
    """

    logger.info("Starting main function.")
    hw = HelloWorld()
    logger.info(f"HelloWorld object: {hw}")


if __name__ == "__main__":
    main()
