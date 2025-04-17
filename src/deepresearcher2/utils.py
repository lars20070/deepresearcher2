#!/usr/bin/env python3

from duckduckgo_search import DDGS
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
    import urllib.error
    import urllib.request

    from bs4 import BeautifulSoup

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
