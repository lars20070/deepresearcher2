#!/usr/bin/env python3
import gzip
import os
import re
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable

import brotli
import pypandoc
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import HttpUrl
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .logger import logger
from .models import WebSearchResult


def retry_with_backoff(retry_min: int = 20, retry_max: int = 1000, retry_attempts: int = 5) -> Callable:
    """
    Decorator factory for retrying a function with exponential backoff.

    For example, the first retry will wait 20 seconds, the second 40 seconds, the third 80 seconds, and so on.
    But never exceeding 1000 seconds. Stopping after 5 attempts.

    Args:
        retry_min (int): First retry wait time in seconds. Defaults to 20 seconds.
        retry_max (int): Maximum retry wait time in seconds.
            The wait time no longer rises exponentially beyond this maximum wait time. Defaults to 1000 seconds.
        retry_attempts (int): Maximum number of retry attempts. Defaults to 5 attempts.

    Returns:
        Callable: A tenacity decorator instance.

    Example:
        >>> @retry_with_backoff(retry_min=20, retry_max=2000, retry_attempts=50)
    """

    return retry(wait=wait_exponential(exp_base=2, multiplier=retry_min, min=retry_min, max=retry_max), stop=stop_after_attempt(retry_attempts))


def html2text(html: bytes) -> str:
    """
    Convert HTML to (clean) plain text.

    Args:
        html (bytes): The HTML content to convert

    Returns:
        str: The converted plain text content
    """

    # Parse HTML
    soup = BeautifulSoup(html, "lxml")

    # Remove scripts and style elements
    for junk in soup(["nav", "footer", "header", "aside", "form", "script", "style"]):
        junk.decompose()

    # Get text from the entire soup object after removing junk
    text = soup.get_text()

    # Some cleanup
    lines = (line.strip() for line in text.splitlines())  # Break into lines and remove leading and trailing space on each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # Break multi-headlines into a line each
    clean_text = "\n".join(chunk for chunk in chunks if chunk)  # Drop blank lines
    clean_text = clean_text.replace("\xa0", " ")  # Replace non-breaking spaces

    return clean_text


# @retry_with_backoff()
def fetch_full_page_content(url: HttpUrl, timeout: int = 10) -> str:
    """
    Fetch the full content of a webpage.
    Note that the method never raises an exception but returns an empty string if the page cannot be fetched.

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
    logger.info(f"Fetching content from URL: {str(url)}")

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
        if encoding:
            encoding = encoding.lower()
            if "gzip" in encoding:
                html = gzip.decompress(raw)
            elif "deflate" in encoding:
                html = zlib.decompress(raw)
            elif "br" in encoding:
                html = brotli.decompress(raw)
            else:
                html = raw
        else:
            html = raw

        # Clean up html
        text = html2text(html)

        return text

    except urllib.error.HTTPError as e:
        if e.code in (403, 401):
            logger.error(f"Authentication error for {url}: {e.code}")
            return ""
        else:
            logger.error(f"HTTP error for {url}: {e.code}")
            return ""

    except urllib.error.URLError as e:
        logger.error(f"Network error for {url}: {str(e)}")
        return ""


@retry_with_backoff(retry_min=20, retry_max=2000, retry_attempts=50)
def duckduckgo_search(query: str, max_results: int = 2, max_content_length: int | None = None) -> list[WebSearchResult]:
    """
    Perform a web search using DuckDuckGo and return a list of results.

    Args:
        query (str): The search query to execute.
        max_results (int, optional): Maximum number of results to return. Defaults to 2.
        max_content_length (int | None, optional): Maximum character length of the content. If none, the full content is returned. Defaults to None.

    Returns:
        list[WebSearchResult]: list of search results

    Example:
        >>> results = duckduckgo_search("petrichor", max_results=10)
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
        title = r.get("title")
        url = r.get("href")
        content = r.get("body")

        # Should we fetch full page content?
        should_fetch = content is None or max_content_length is None or (max_content_length is not None and len(content) < max_content_length)

        # Only fetch if necessary
        if should_fetch:
            full_content = fetch_full_page_content(url)
            if content is None or len(full_content) > len(content):
                content = full_content

        # Apply length constraint if needed
        if max_content_length is not None and content is not None:
            content = content[:max_content_length]

        result = WebSearchResult(title=title, url=str(url), content=content)
        results.append(result)

    return results


@retry_with_backoff()
def tavily_search(query: str, max_results: int = 2, max_content_length: int | None = None) -> list[WebSearchResult]:
    """
    Perform a web search using Tavily and return a list of results.

    Args:
        query (str): The search query to execute.
        max_results (int, optional): Maximum number of results to return. Defaults to 2.
        max_content_length (int | None, optional): Maximum character length of the content. If none, the full content is returned. Defaults to None.

    Returns:
        list[WebSearchResult]: list of search results

    Example:
        >>> results = tavily_search("petrichor", max_results=10)
        >>> for result in results:
        ...     print(result.title, result.url)
    """
    logger.info(f"Tavily web search for: {query}")

    tavily_client = TavilyClient()
    try:
        tavily_results = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=True,
        )
        if not tavily_results:
            logger.warning(f"Tavily returned no results for: {query}")
            return []
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Network error during Tavily search: {str(e)}")
        raise

    # logger.debug(f"Complete Tavily results:\n{json.dumps(tavily_results['results'], indent=2)}")

    # Convert to pydantic objects
    results = []
    for r in tavily_results["results"]:
        title = r["title"]
        url = r["url"]
        summary = r["content"]
        content = r["raw_content"]

        # Should we fetch full page content?
        should_fetch = content is None or max_content_length is None or (max_content_length is not None and len(content) < max_content_length)

        # Only fetch if necessary
        if should_fetch:
            full_content = fetch_full_page_content(url)
            if content is None or len(full_content) > len(content):
                content = full_content

        # Apply length constraint if needed
        if max_content_length is not None and content is not None:
            content = content[:max_content_length]

        result = WebSearchResult(title=title, url=str(url), summary=summary, content=content)
        results.append(result)

    return results


@retry_with_backoff()
def perplexity_search(query: str) -> list[WebSearchResult]:
    """
    Perform a web search using Perplexity and return a list of results.
    Note that the list of results contains only a single item.
    Note that the 'summary' field is empty.

    Args:
        query (str): The search query to execute.

    Returns:
        list[WebSearchResult]: list of search results

    Example:
        >>> results = perplexity_search("petrichor")
        >>> for result in results:
        ...     print(result.title, result.url)
    """
    logger.info(f"Perplexity web search for: {query}")

    perplexity_url = "https://api.perplexity.ai/chat/completions"
    headers = {"accept": "application/json", "content-type": "application/json", "Authorization": f"Bearer {config.perplexity_api_key}"}
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "Search the web and provide factual information with sources."},
            {"role": "user", "content": query},
        ],
    }

    try:
        response = requests.post(perplexity_url, headers=headers, json=payload)
        response.raise_for_status()
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Network error during Perplexity search: {str(e)}")
        raise

    perplexity_results = response.json()
    # logger.debug(f"Complete Perplexity results:\n{json.dumps(perplexity_results, indent=2)}")

    title = query
    url = perplexity_results["citations"][0]  # TODO: A list of URLs is returned, but we cannot press them into WebSearchResult.
    content = perplexity_results["choices"][0]["message"]["content"]

    result = WebSearchResult(title=title, url=url, content=content)
    return [result]


@retry_with_backoff()
def brave_search(query: str, max_results: int = 2, max_content_length: int | None = None) -> list[WebSearchResult]:
    """
    Perform a web search using Brave and return a list of results.

    Args:
        query (str): The search query to execute.
        max_results (int, optional): Maximum number of results to return. Defaults to 2.
        max_content_length (int | None, optional): Maximum character length of the content. If none, the full content is returned. Defaults to None.

    Returns:
        list[WebSearchResult]: list of search results

    Example:
        >>> results = brave_search("petrichor", max_results=10)
        >>> for result in results:
        ...     print(result.title, result.url)
    """
    logger.info(f"Brave web search for: {query}")

    brave_url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "X-Subscription-Token": config.brave_api_key,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "count": max_results,
        "search_lang": "en",
        "country": "US",
    }

    try:
        response = requests.get(brave_url, headers=headers, params=params)
        response.raise_for_status()
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Network error during Brave search: {str(e)}")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during Brave search: {str(e)}")
        raise

    brave_results = response.json().get("web", {}).get("results", [])
    if not brave_results:
        logger.warning(f"Brave returned no results for: {query}")
        return []

    results = []
    for r in brave_results:
        title = r.get("title")
        url = r.get("url")
        summary = r.get("description")
        content = fetch_full_page_content(url)

        if max_content_length is not None and content is not None:
            content = content[:max_content_length]

        result = WebSearchResult(title=title, url=str(url), summary=summary, content=content)
        results.append(result)

    return results


# @retry_with_backoff()
def serper_search(query: str, max_results: int = 2, max_content_length: int | None = None) -> list[WebSearchResult]:
    """
    Perform a web search using Serper and return a list of results.

    Args:
        query (str): The search query to execute.
        max_results (int, optional): Maximum number of results to return. Defaults to 2.
        max_content_length (int | None, optional): Maximum character length of the content. If none, the full content is returned. Defaults to None.

    Returns:
        list[WebSearchResult]: list of search results

    Example:
        >>> results = serper_search("petrichor", max_results=10)
        >>> for result in results:
        ...     print(result.title, result.url)
    """
    logger.info(f"Serper web search for: {query}")

    serper_url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": config.serper_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": max_results,
    }

    try:
        response = requests.post(serper_url, headers=headers, json=payload)
        response.raise_for_status()
        # serper_results = response.json().get("organic", [])
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Network error during Serper search: {str(e)}")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during Serper search: {str(e)}")
        raise

    serper_results = response.json().get("organic", [])
    if not serper_results:
        logger.warning(f"Serper returned no results for: {query}")
        return []

    results = []
    for r in serper_results:
        title = r.get("title")
        url = r.get("link")
        summary = r.get("snippet")
        content = fetch_full_page_content(url)

        if max_content_length is not None and content is not None:
            content = content[:max_content_length]

        result = WebSearchResult(title=title, url=str(url), summary=summary, content=content)
        results.append(result)

    return results


def export_report(report: str, topic: str = "Report", output_dir: str = "reports/") -> None:
    """
    Export the report to markdown (and pdf).
    If 'pandoc' is installed on the system, the report will be exported to both markdown and pdf formats.

    Args:
        report (str): The report content in markdown format.
        topic (str): The topic of the report. Defaults to "Report".
        output_dir (str): The directory where the report will be saved. Defaults to "reports/".
    """
    if os.path.exists(output_dir):
        file_name = re.sub(r"[^a-zA-Z0-9]", "_", topic).lower()
        path_md = os.path.join(output_dir, f"{file_name}.md")
        logger.debug(f"Exporting report to {path_md}")
        with open(path_md, "w", encoding="utf-8") as f:
            f.write(report)

        # Convert markdown to PDF using Pandoc
        try:
            logger.info(f"Writing the final report as PDF to '{output_dir}'")
            logger.debug(f"Pandoc version {pypandoc.get_pandoc_version()} is installed at path: '{pypandoc.get_pandoc_path()}'")
            path_pdf = os.path.join(output_dir, f"{file_name}.pdf")
            pypandoc.convert_file(
                path_md,
                "pdf",
                outputfile=path_pdf,
                extra_args=[
                    "--pdf-engine=xelatex",
                    "-V",
                    "colorlinks=true",
                    "-V",
                    "linkcolor=blue",  # Internal links
                    "-V",
                    "urlcolor=blue",  # External links
                    "-V",
                    "citecolor=blue",
                    "--from",
                    "markdown+autolink_bare_uris",  # Ensures bare URLs are also hyperlinked
                ],
            )
        except Exception:
            logger.error("Pandoc is not installed. Skipping conversion to PDF.")
    else:
        logger.error(f"Output directory {output_dir} does not exist. Skipping writing the final report.")


def remove_reasoning_tags(text: str) -> str:
    """
    Remove any text between reasoning tags <think>...</think>.

    Args:
        text (str): The text containing reasoning tags.

    Returns:
        str: The text without reasoning tags.
    """
    pattern = r"(?:<|&lt;)think(?:>|&gt;).*?(?:<|&lt;)/think(?:>|&gt;)"
    return re.sub(pattern, "", text, flags=re.DOTALL)
