#!/usr/bin/env python3
import json
import os

from dotenv import load_dotenv

from deepresearcher2.config import config
from deepresearcher2.logger import logger

load_dotenv()


def test_config() -> None:
    """
    Test the Config class
    """
    logger.info("Testing the Config() class")

    # Load environment variables the conventional way
    max_research_loops = int(os.environ.get("MAX_RESEARCH_LOOPS", "3"))
    search_engine = os.environ.get("SEARCH_ENGINE", "duckduckgo")

    assert config is not None
    assert config.max_research_loops == max_research_loops
    assert config.search_engine == search_engine
    logger.debug(f"Config:\n{json.dumps(config.model_dump(), indent=2)}")
