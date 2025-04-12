#!/usr/bin/env python3

import logfire

from deepresearcher2 import logger


def test_loguru() -> None:
    """
    Test the Loguru logging functionality
    """
    logger.info("Testing Loguru logging functionality")

    # Check the Loguru output at ./deepresearcher2.log


def test_logfire() -> None:
    """
    Test the Logfire logging functionality
    """
    logfire.info("Testing Logfire logging functionality")

    # Check the logfire output at https://logfire-eu.pydantic.dev/lars20070/deepresearcher2
