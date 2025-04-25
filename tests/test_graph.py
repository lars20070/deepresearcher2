#!/usr/bin/env python3

import pytest

from deepresearcher2.graph import DeepState, GraphRunContext, SummarizeSearchResults, WebSearch


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_websearch(topic: str) -> None:
    """
    Test that WebSearch node
    """

    state = DeepState(topic=topic, count=1)
    ctx = GraphRunContext(state=state, deps=None)

    node = WebSearch()
    result = await node.run(ctx)

    assert isinstance(result, SummarizeSearchResults)
