#!/usr/bin/env python3

import gzip
import urllib.error
import urllib.request
import zlib

import brotli
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential

from deepresearcher2 import logger


def retry_with_backoff(func: callable) -> callable:
    """
    Retry decorator with exponential backoff.

    For example, the first retry will wait 20 seconds, the second 40 seconds, the third 80 seconds, and so on. Stopping after 5 attempts.
    """
    retry_min = 20
    retry_max = 1000
    retry_attempts = 5

    return retry(wait=wait_exponential(min=retry_min, max=retry_max), stop=stop_after_attempt(retry_attempts))(func)


@retry_with_backoff
def fetch_page_content(url: str) -> str:
    """Fetch the content of a webpage given its URL."""

    try:
        response = urllib.request.urlopen(url, timeout=10)
        html = response.read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()
    except urllib.error.HTTPError as e:
        if e.code in (403, 401):
            logger.error(f"Authentication error for {url}: {e.code}")
            return f"[Error: Access denied to {url} (code {e.code})]"
        else:
            logger.error(f"HTTP error for {url}: {e.code}")
            raise  # Will be retried by decorator
    except urllib.error.URLError as e:
        logger.error(f"Network error for {url}: {str(e)}")
        raise  # Will be retried by decorator


def duckduckgo_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> dict[str, list[dict[str, str]]]:
    """Search the web using DuckDuckGo.

    Args:
        query (str): The search query to execute
        max_results (int): Maximum number of results to return

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Same as content since DDG doesn't provide full page content
    """
    logger.info(f"Searching the web using DuckDuckGo for: {query}")
    results = []

    try:
        with DDGS() as ddgs:
            try:
                search_results = list(ddgs.text(query, max_results=max_results))
                if not search_results:
                    logger.warning(f"DuckDuckGo returned no results for: {query}")
                    return {"results": []}

            except (ConnectionError, TimeoutError) as e:
                logger.error(f"Network error during search: {str(e)}")
                raise  # Will be retried by decorator

            for r in search_results:
                url = r.get("href")
                title = r.get("title")
                content = r.get("body")

                if not all([url, title, content]):
                    logger.warning(f"Warning: Incomplete result from DuckDuckGo: {r}")
                    continue

                raw_content = content
                if fetch_full_page:
                    try:
                        raw_content = fetch_page_content(url)

                    except Exception as e:
                        logger.error(f"Error: Failed to fetch full page content for {url}: {str(e)}")

                # Add result to list
                result = {"title": title, "url": url, "content": content, "raw_content": raw_content}
                results.append(result)

            return {"results": results}

    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {str(e)}")
        logger.error(f"Full error details: {type(e).__name__}")
        raise  # Will be retried by decorator


class WebSearchResult2(BaseModel):
    title: str = Field(..., description="short descriptive title of the web search result")
    url: HttpUrl = Field(..., description="URL of the web search result")
    content: str = Field(..., description="main content of the web search result")


def fetch_full_page_content(url: HttpUrl, timeout: int = 10) -> str:
    """
    Fetch the full content of a webpage.

    Args:
        url (HttpUrl): The URL of a webpage
        timeout (int): Timeout in seconds for the request. Defaults to 10 seconds.

    Returns:
        str: The full content of the webpage

    Raises:
        urllib.error.HTTPError: If the server can be reached but returns an error
        urllib.error.URLError: If the server cannot be reached

    Example:
        >>> content = fetch_full_page_content("https://example.com")
        >>> print(content)
    """

    try:
        # Mimic a browser by setting appropriate headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        request = urllib.request.Request(str(url), headers=headers)
        response = urllib.request.urlopen(request, timeout=timeout)
        raw = response.read()
        encoding = response.headers.get("Content-Encoding")

        # Decompress the content if necessary
        if encoding == "gzip":
            html = gzip.decompress(raw)
        elif encoding == "deflate":
            html = zlib.decompress(raw)
        elif encoding == "br":
            html = brotli.decompress(raw)
        else:
            html = raw

        # Decode the HTML content
        text = BeautifulSoup(html, "html.parser").get_text()
        return text

    except urllib.error.HTTPError as e:
        if e.code in (403, 401):
            logger.error(f"Authentication error for {url}: {e.code}")
            return f"[Error: Access denied to {url} (code {e.code})]"
        else:
            logger.error(f"HTTP error for {url}: {e.code}")
            raise

    except urllib.error.URLError as e:
        logger.error(f"Network error for {url}: {str(e)}")
        raise


def duckduckgo(query: str, max_results: int = 2, max_content_length: int | None = None) -> list[WebSearchResult2]:
    """
    Perform a web search using DuckDuckGo and return a list of results.

    Args:
        query (str): The search query to execute.
        max_results (int, optional): Maximum number of results to return. Defaults to 2.
        max_content_length (int | None, optional): Maximum character length of the content. If none, the full content is returned. Defaults to None.

    Returns:
        list[WebSearchResult2]: list of search results

    Example:
        >>> results = duckduckgo("petrichor", max_results=10)
        >>> for result in results:
        ...     print(result.title, result.url)
    """
    logger.info(f"DuckDuckGo web search for: {query}")

    # Run the search
    with DDGS() as ddgs:
        try:
            ddgs_results = list(ddgs.text(query, max_results=max_results))
            if not ddgs_results:
                logger.warning(f"DuckDuckGo returned no results for: {query}")
                return []
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Network error during DuckDuckGo search: {str(e)}")
            raise

    # Convert to pydantic objects
    results = []
    for r in ddgs_results:
        result = WebSearchResult2(
            title=r.get("title"),
            url=r.get("href"),
            content=r.get("body"),
        )
        results.append(result)

    return results
