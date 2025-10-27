#!/usr/bin/env python3

from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from pydantic import HttpUrl, ValidationError
from pytest_mock import MockerFixture

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


def test_duckduckgo_search(mocker: MockerFixture, mock_fetch_full_page_content: MagicMock) -> None:
    """
    Test the duckduckgo_search() search function

    Both external dependency `DDGS` and internal `fetch_full_page_content` method are mocked.
    """
    # Mock both the DDGS context manager and its instance
    mock_ddgs_instance = mocker.MagicMock()  # Class instance ddgs used within the 'with' block
    mock_ddgs_instance.text.return_value = [
        {
            "title": "Mocked Title 1",
            "href": "http://example.com/1",
            "body": "Short body for test 1",
        },
        {
            "title": "Mocked Title 2",
            "href": "http://example.com/2",
            "body": "Short body for test 2",
        },
        {
            "title": "Mocked Title 3",
            "href": "http://example.com/3",
            "body": "Short body for test 3",
        },
    ]

    mock_ddgs_context_manager = mocker.MagicMock()  # Context manager returned by DDGS()
    mock_ddgs_context_manager.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_context_manager.__exit__.return_value = None

    mocker.patch("deepresearcher2.utils.DDGS", return_value=mock_ddgs_context_manager)

    logger.info("Testing duckduckgo_search().")

    # Full content length
    n = 3  # Number of results
    topic = config.topic
    results = duckduckgo_search(
        topic,
        max_results=n,
    )

    assert len(results) == n
    for r in results:
        assert r.title is not None
        assert r.url is not None
        assert r.content == "Mocked long page content for testing purposes."  # Mocked content from fetch_full_page_content
        assert r.summary == ""  # DuckDuckGo does not provide a summary.
        assert isinstance(r.url, str)
        logger.debug(f"search result title: {r.title}")
        logger.debug(f"search result url: {r.url}")
        logger.debug(f"search result content length: {len(r.content)}")

    # Restricted content length
    m = 10  # Rather short max content length
    results2 = duckduckgo_search(
        topic,
        max_results=n,
        max_content_length=m,
    )

    assert len(results2) == n
    for r in results2:
        assert r.content is not None
        assert len(r.content) <= m
        assert r.content == "Short body"  # Mocked content from ddgs


def test_tavily_search(mocker: MockerFixture, mock_fetch_full_page_content: MagicMock) -> None:
    """
    Test the tavily_search() search function

    Both external dependency `TavilyClient` and internal `fetch_full_page_content` method are mocked.
    """
    # Mock TavilyClient
    mock_tavily_client_instance = mocker.MagicMock()
    mock_tavily_client_instance.search.return_value = {
        "results": [
            {
                "title": "Mocked Title 1",
                "url": "http://example.com/1",
                "content": "Mocked summary 1",
                "raw_content": "Mocked raw content 1",
            },
            {
                "title": "Mocked Title 2",
                "url": "http://example.com/2",
                "content": "Mocked summary 2",
                "raw_content": "Mocked raw content 2",
            },
            {
                "title": "Mocked Title 3",
                "url": "http://example.com/3",
                "content": "Mocked summary 3",
                "raw_content": "Mocked raw content 3",
            },
        ]
    }
    mocker.patch("deepresearcher2.utils.TavilyClient", return_value=mock_tavily_client_instance)

    logger.info("Testing tavily_search().")

    # Full content length
    n = 3  # Number of results
    topic = config.topic
    results = tavily_search(
        topic,
        max_results=n,
    )

    assert len(results) == n
    for r in results:
        assert r.title is not None
        assert r.url is not None
        assert r.summary is not None
        assert r.content == "Mocked long page content for testing purposes."  # Mocked content from fetch_full_page_content
        assert isinstance(r.url, str)
        logger.debug(f"search result title: {r.title}")
        logger.debug(f"search result url: {r.url}")
        logger.debug(f"search result summary: {r.summary}")
        logger.debug(f"search result content length: {len(r.content)}")
        # logger.debug(f"search result content: {r.content}")

    # results_json = json.dumps([r.model_dump() for r in results], indent=2)
    # logger.debug(f"Tavily search results:\n{results_json}")

    # Restricted content length
    m = 10  # Max content length
    results2 = tavily_search(
        topic,
        max_results=n,
        max_content_length=m,
    )

    assert len(results2) == n
    for r in results2:
        assert r.content is not None
        assert len(r.content) <= m
        assert r.content.startswith("Mocked raw")


def test_perplexity_search(mocker: MockerFixture) -> None:
    """
    Test the perplexity_search() search function

    The external dependency `requests.post` is mocked.
    """
    # Mock requests.post
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Mocked Perplexity content."}}],
        "citations": ["http://example.com/perplexity"],
    }
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.post", return_value=mock_response)

    logger.info("Testing perplexity_search().")

    topic = config.topic
    results = perplexity_search(topic)
    result = results[0]  # Perplexity search returns only a single result

    assert result.title == topic
    assert result.url == "http://example.com/perplexity"
    assert result.summary is not None and result.summary == ""  # Perplexity does not provide a summary.
    assert result.content == "Mocked Perplexity content."
    logger.debug(f"search result title: {result.title}")
    logger.debug(f"search result url: {result.url}")
    logger.debug(f"search result content length: {len(result.content)}")
    # logger.debug(f"search result content: {result.content}")


# Brave API is generous. 2,000 free requests per month.
@pytest.mark.vcr()
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


# Serper API is generous. 2,500 free requests per month.
@pytest.mark.vcr()
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


@pytest.mark.vcr()
# @pytest.mark.searxng
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
