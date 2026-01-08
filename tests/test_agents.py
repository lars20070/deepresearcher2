#!/usr/bin/env python3

from dotenv import load_dotenv

from deepresearcher2.agents import create_model
from deepresearcher2.config import Config, Provider, config
from deepresearcher2.logger import logger

load_dotenv()


def test_create_model() -> None:
    """
    Test create_model() functionality for all providers.
    """
    logger.info("Testing create_model() functionality for all providers.")

    for provider in Provider:
        logger.debug(f"Provider: {provider}")

        config = Config()
        config.provider = provider

        model = create_model(config)
        logger.debug(f"Created model:\n{model}")
        assert model is not None


def test_config() -> None:
    """
    Test the Config class
    """
    logger.info("Testing the Config() class")

    # See values in config_for_testing() fixture
    assert config is not None
    assert config.max_research_loops == 3
    assert config.search_engine == "serper"

    # logger.debug(f"Config:\n{config.model_dump_json(indent=2)}")
