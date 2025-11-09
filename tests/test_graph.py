#!/usr/bin/env python3
import glob
import json
import os
from collections.abc import Generator
from typing import cast

import pytest
from pydantic_graph import End

from deepresearcher2.config import config
from deepresearcher2.graph import DeepState, FinalizeSummary, GraphRunContext, ReflectOnSearch, SummarizeSearchResults, WebSearch
from deepresearcher2.logger import logger
from deepresearcher2.models import WebSearchResult, WebSearchSummary


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_websearch_without_reflection(topic: str) -> None:
    """
    Test WebSearch node without reflection
    """
    logger.info("Testing WebSearch() node without reflection")

    # Prepare the initial state
    state = DeepState(topic=topic, count=1)
    ctx = GraphRunContext(state=state, deps=None)
    assert ctx.state.topic == topic
    assert ctx.state.count == 1
    assert ctx.state.reflection is None
    assert ctx.state.search_query is None
    assert ctx.state.search_results is None
    assert ctx.state.search_summaries is None

    # Run the WebSearch node
    node = WebSearch()
    result = await node.run(ctx)

    assert isinstance(result, SummarizeSearchResults)
    search_results = ctx.state.search_results
    assert search_results is not None
    search_results = cast(list[WebSearchResult], search_results)
    if config.search_engine == "perplexity":
        assert len(search_results) == 1  # Perplexity return only one result
    else:
        assert len(search_results) == config.max_web_search_results
    for r in search_results:
        assert r.title is not None
        assert r.url is not None
        assert r.summary is not None or r.content is not None

    results_json = json.dumps([r.model_dump() for r in search_results], indent=2)
    logger.debug(f"Search results:\n{results_json}")

    # # Serialize the state to JSON
    # with open("tests/data/state_1.json", "w") as f:
    #     f.write(ctx.state.model_dump_json(indent=2))


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_websearch_with_reflection(topic: str) -> None:
    """
    Test WebSearch node with reflection
    """
    logger.info("Testing WebSearch() node with reflection")

    # Prepare the initial state
    with open("tests/data/state_3.json") as f:
        state_json = f.read()

    state = DeepState.model_validate_json(state_json)
    ctx = GraphRunContext(state=state, deps=None)

    assert ctx.state.topic == "petrichor"
    assert ctx.state.count == 2  # We are now in the second loop. We have the reflections from the first loop.
    assert ctx.state.search_query is not None
    assert ctx.state.search_results is not None
    assert len(ctx.state.search_results) == 3
    assert ctx.state.search_summaries is not None
    assert len(ctx.state.search_summaries) == 1
    assert ctx.state.reflection is not None

    # Run the WebSearch node
    node = WebSearch()
    result = await node.run(ctx)

    assert isinstance(result, SummarizeSearchResults)
    search_results = ctx.state.search_results
    assert search_results is not None
    if config.search_engine == "perplexity":
        assert len(search_results) == 1  # Perplexity return only one result
    else:
        assert len(search_results) == config.max_web_search_results
    for r in search_results:
        assert r.title is not None
        assert r.url is not None
        assert r.summary is not None or r.content is not None

    results_json = json.dumps([r.model_dump() for r in search_results], indent=2)
    logger.debug(f"Search results:\n{results_json}")


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_summarizesearchresults() -> None:
    """
    Test SummarizeSearchResults() node
    """
    logger.info("Testing SummarizeSearchResults() node")

    # Prepare the initial state
    with open("tests/data/state_1.json") as f:
        state_json = f.read()

    state = DeepState.model_validate_json(state_json)
    ctx = GraphRunContext(state=state, deps=None)

    assert ctx.state.topic == "petrichor"
    assert ctx.state.count == 1
    assert ctx.state.search_query is not None
    assert ctx.state.search_results is not None
    assert len(ctx.state.search_results) == 3
    assert ctx.state.search_summaries is None
    assert ctx.state.reflection is None

    # Run the SummarizeSearchResults node
    node = SummarizeSearchResults()
    result = await node.run(ctx)

    assert isinstance(result, ReflectOnSearch)
    search_summaries = ctx.state.search_summaries
    assert search_summaries is not None
    search_summaries = cast(list[WebSearchSummary], search_summaries)
    assert len(search_summaries) == 1
    for s in search_summaries:
        assert s.summary is not None
        assert s.aspect is not None
        # TODO: Since there is a summary, there should be at least one reference. But sometimes the model struggles the genereate valid JSON.
        assert s.references is not None
        for r in s.references:
            assert r.title is not None
            assert r.url is not None

    summaries_json = json.dumps([s.model_dump() for s in search_summaries], indent=2)
    logger.debug(f"Search summaries:\n{summaries_json}")

    # # Serialize the state to JSON
    # with open("tests/data/state_2.json", "w") as f:
    #     f.write(ctx.state.model_dump_json(indent=2))


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_reflectonsearch() -> None:
    """
    Test ReflectOnSearch() node
    """
    logger.info("Testing ReflectOnSearch() node")

    # Prepare the initial state
    with open("tests/data/state_2.json") as f:
        state_json = f.read()

    state = DeepState.model_validate_json(state_json)
    ctx = GraphRunContext(state=state, deps=None)

    assert ctx.state.topic == "petrichor"
    assert ctx.state.count == 1
    assert ctx.state.search_query is not None
    assert ctx.state.search_results is not None
    assert len(ctx.state.search_results) == 3
    assert ctx.state.search_summaries is not None
    assert len(ctx.state.search_summaries) == 1
    assert ctx.state.reflection is None

    # Run the ReflectOnSearch node
    node = ReflectOnSearch()
    result = await node.run(ctx)

    assert isinstance(result, (WebSearch | FinalizeSummary))
    reflection = ctx.state.reflection
    assert reflection is not None
    assert reflection.knowledge_gaps is not None
    assert reflection.covered_topics is not None

    reflection_json = reflection.model_dump_json(indent=2)
    logger.debug(f"Reflection:\n{reflection_json}")

    # # Serialize the state to JSON
    # with open("tests/data/state_3.json", "w") as f:
    #     f.write(ctx.state.model_dump_json(indent=2))


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_finalizesummary(config_for_testing: Generator[None, None, None], cleanup_reports_folder: None) -> None:
    """
    Test FinalizeSummary() node

    Note that we are only testing the creation of markdown reports.
    PDF reports might or might not be created depending on whether 'pandoc' is installed on the system.
    """
    logger.info("Testing FinalizeSummary() node")

    # Prepare the initial state
    with open("tests/data/state_3.json") as f:
        state_json = f.read()

    state = DeepState.model_validate_json(state_json)
    ctx = GraphRunContext(state=state, deps=None)

    assert ctx.state.topic == "petrichor"
    assert ctx.state.count == 2
    assert ctx.state.search_query is not None
    assert ctx.state.search_results is not None
    assert len(ctx.state.search_results) == 3
    assert ctx.state.search_summaries is not None
    assert len(ctx.state.search_summaries) == 1
    assert ctx.state.reflection is not None

    # Run the FinalizeSummary node
    node = FinalizeSummary()
    result = await node.run(ctx)

    assert isinstance(result, End)
    md_files = glob.glob(os.path.join(config.reports_folder, "*.md"))
    assert len(md_files) == 1  # A single report should be created.
