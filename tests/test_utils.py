#!/usr/bin/env python3

import json
import os

from pydantic import HttpUrl

from deepresearcher2 import logger
from deepresearcher2.utils import duckduckgo, duckduckgo_search


def test_duckduckgo_search(topic: str, load_env: None) -> None:
    # Number of results
    n = 3

    logger.info("Testing searching with DuckDuckGo.")
    result = duckduckgo_search(topic, max_results=n, fetch_full_page=False)
    logger.debug(f"Entire search result: {result}")

    # Check whether the result contains a 'results' key
    assert "results" in result
    logger.debug(f"Number of search results: {len(result['results'])}")
    if len(result["results"]) > 0:
        for i, item in enumerate(result["results"]):
            logger.debug(f"Result {i + 1}:\n{json.dumps(item, indent=2)}")

    # Check if the number of results is correct
    assert len(result["results"]) == n


def test_duckduckgoo(load_env: None) -> None:
    """
    Test the duckduckgo() search function
    """

    n = 3  # Number of results
    topic = os.environ.get("TOPIC", "petrichor")
    results = duckduckgo(
        topic,
        max_results=n,
    )

    assert len(results) <= n
    for r in results:
        logger.debug(f"search result title: {r.title}")
        logger.debug(f"search result url: {r.url}")
        assert r.title is not None
        assert r.url is not None
        assert r.content is not None
        assert isinstance(r.url, HttpUrl)
