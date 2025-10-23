#!/usr/bin/env python3

import pytest
from dotenv import load_dotenv
from pydantic import HttpUrl, ValidationError

from deepresearcher2.config import config
from deepresearcher2.logger import logger
from deepresearcher2.utils import (
    brave_search,
    duckduckgo_search,
    fetch_full_page_content,
    perplexity_search,
    searxng_search,
    serper_search,
    tavily_search,
)

load_dotenv()


def test_fetch_full_page_content() -> None:
    """
    Test the fetch_full_page_content() function
    """

    url = "https://en.wikipedia.org/wiki/Daniel_Noboa"

    try:
        validated_url = HttpUrl(url)
        content = fetch_full_page_content(validated_url)
    except ValidationError as err:
        raise ValueError(f"Invalid URL: {url}") from err

    assert "Daniel Noboa" in content

    # TODO: Workaround for 403 Access Denied
    # url2 = "https://www.politico.com/news/magazine/2025/01/30/curtis-yarvins-ideas-00201552"
    # content2 = fetch_full_page_content(url2)
    # assert "Curtis Yarvin's Ideas" in content2


@pytest.mark.skip(reason="DuckDuckGo aggressively rate limited.")
def test_duckduckgo_search() -> None:
    """
    Test the duckduckgo_search() search function
    """
    logger.info("Testing duckduckgo_search().")

    # Full content length
    n = 3  # Number of results
    topic = config.topic
    results = duckduckgo_search(
        topic,
        max_results=n,
    )

    assert len(results) <= n
    for r in results:
        assert r.title is not None
        assert r.url is not None
        assert r.content is not None
        assert r.summary == ""  # DuckDuckGo does not provide a summary.
        assert isinstance(r.url, str)
        logger.debug(f"search result title: {r.title}")
        logger.debug(f"search result url: {r.url}")
        logger.debug(f"search result content length: {len(r.content)}")

    # Restricted content length
    m = 100  # Max content length
    results2 = duckduckgo_search(
        topic,
        max_results=n,
        max_content_length=m,
    )

    assert len(results2) <= n
    for r in results2:
        assert r.content is not None
        assert len(r.content) <= m


@pytest.mark.paid
def test_tavily_search() -> None:
    """
    Test the tavily_search() search function
    """

    # Full content length
    n = 3  # Number of results
    topic = config.topic
    results = tavily_search(
        topic,
        max_results=n,
    )

    assert len(results) <= n
    for r in results:
        assert r.title is not None
        assert r.url is not None
        assert r.summary is not None
        assert r.content is not None
        assert isinstance(r.url, str)
        logger.debug(f"search result title: {r.title}")
        logger.debug(f"search result url: {r.url}")
        logger.debug(f"search result summary: {r.summary}")
        logger.debug(f"search result content length: {len(r.content)}")
        # logger.debug(f"search result content: {r.content}")

    # results_json = json.dumps([r.model_dump() for r in results], indent=2)
    # logger.debug(f"Tavily search results:\n{results_json}")

    # Restricted content length
    m = 100  # Max content length
    results2 = tavily_search(
        topic,
        max_results=n,
        max_content_length=m,
    )

    assert len(results2) <= n
    for r in results2:
        assert r.content is not None
        assert len(r.content) <= m


@pytest.mark.paid
def test_perplexity_search() -> None:
    """
    Test the perplexity_search() search function
    """

    topic = config.topic
    results = perplexity_search(topic)
    result = results[0]  # Perplexity search returns only a single result

    assert result.title is not None
    assert result.url is not None
    assert result.summary is not None and result.summary == ""  # Perplexity does not provide a summary.
    assert result.content is not None
    logger.debug(f"search result title: {result.title}")
    logger.debug(f"search result url: {result.url}")
    logger.debug(f"search result content length: {len(result.content)}")
    # logger.debug(f"search result content: {result.content}")


# @pytest.mark.paid
# Brave API is generous. 2,000 free requests per month. Hence, we always run the test.
def test_brave_search() -> None:
    topic = config.topic
    results = brave_search(topic, max_results=3)
    result = results[0]

    assert len(results) == 3
    assert result.title is not None
    assert result.url is not None
    assert result.summary is not None
    assert result.content is not None
    logger.debug(f"search result title: {result.title}")
    logger.debug(f"search result url: {result.url}")
    logger.debug(f"search result summary: {result.summary}")
    logger.debug(f"search result content length: {len(result.content)}")
    # logger.debug(f"search result content: {result.content}")


# @pytest.mark.paid
# Serper API is generous. 2,500 free requests per month. Hence, we always run the test.
def test_serper_search() -> None:
    topic = config.topic
    results = serper_search(topic, max_results=3)
    result = results[0]

    assert len(results) == 3
    assert result.title is not None
    assert result.url is not None
    assert result.summary is not None
    assert result.content is not None
    logger.debug(f"search result title: {result.title}")
    logger.debug(f"search result url: {result.url}")
    logger.debug(f"search result summary: {result.summary}")
    logger.debug(f"search result content length: {len(result.content)}")
    # logger.debug(f"search result content: {result.content}")


@pytest.mark.searxng
def test_searxng_search() -> None:
    topic = config.topic
    results = searxng_search(topic, max_results=3)
    result = results[0]

    assert len(results) == 3
    assert result.title is not None
    assert result.url is not None
    assert result.summary is not None
    assert result.content is not None
    logger.debug(f"search result title: {result.title}")
    logger.debug(f"search result url: {result.url}")
    logger.debug(f"search result summary: {result.summary}")
    logger.debug(f"search result content length: {len(result.content)}")
    # logger.debug(f"search result content: {result.content}")
