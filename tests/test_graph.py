#!/usr/bin/env python3

import json

import pytest

from deepresearcher2.config import config
from deepresearcher2.graph import DeepState, GraphRunContext, SummarizeSearchResults, WebSearch
from deepresearcher2.logger import logger


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_websearch(topic: str) -> None:
    """
    Test that WebSearch node without reflection
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
    if config.search_engine == "perplexity":
        assert len(search_results) == 1  # Perplexity return only one result
    else:
        assert len(search_results) == config.max_web_search_results
    for r in search_results:
        assert r.title is not None
        assert r.url is not None
        assert r.summary is not None or r.content is not None

    logger.debug(f"Search results:\n{json.dumps([r.model_dump() for r in search_results], indent=2)}")
