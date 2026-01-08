#!/usr/bin/env python3

from dotenv import load_dotenv

from deepresearcher2.agents import create_model
from deepresearcher2.config import Config, Provider
from deepresearcher2.logger import logger

load_dotenv()


def test_create_model() -> None:
    """
    Test create_model() functionality for all providers.
    """
    logger.info("Testing create_model() functionality for all providers.")

    for provider in Provider:
        logger.debug(f"Provider: {provider.value}")

        config = Config()
        config.provider = provider

        model = create_model(config)
        logger.debug(f"Created model: {model}")
        assert model is not None
