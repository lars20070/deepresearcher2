#!/usr/bin/env python3

from dotenv import load_dotenv

from deepresearcher2.config import SearchEngine, config
from deepresearcher2.logger import logger

load_dotenv()


def test_config() -> None:
    """
    Test the Config class
    """
    logger.info("Testing the Config() class")

    # See values in config_for_testing() fixture
    assert config is not None
    assert config.max_research_loops == 3
    assert config.search_engine == SearchEngine.serper

    # logger.debug(f"Config:\n{config.model_dump_json(indent=2)}")
