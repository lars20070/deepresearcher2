#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from deepresearcher2.agents import query_agent
from deepresearcher2.logger import logger
from deepresearcher2.utils import duckduckgo_search


# Data classes
@dataclass
class DeepState:
    topic: str = "petrichor"
    search_query: str = ""
    search_results: list[str] = field(default_factory=list)
    count: int = 0
    summary: str = ""


# Nodes
@dataclass
class WebSearch(BaseNode[DeepState]):
    """
    Web Search node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> SummarizeSearchResults:
        logger.debug(f"Running Web Search with count number {ctx.state.count}.")

        topic = ctx.state.topic

        # Generate the query
        async with query_agent.run_mcp_servers():
            prompt = f"Please generate a web search query for the following topic: {topic}"
            result = await query_agent.run(prompt)
            query = result.output
            logger.debug(f"Web search query: {query}")

        # Run the search
        search_results = duckduckgo_search(query=query.query, max_results=10)
        for r in search_results:
            logger.debug(f"Search result title: {r.title}")
            logger.debug(f"Search result url: {r.url}")
            logger.debug(f"Search result content length: {len(r.content)}")

        return SummarizeSearchResults()


@dataclass
class SummarizeSearchResults(BaseNode[DeepState]):
    """
    Summarize Search Results node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> ReflectOnSearch:
        logger.debug(f"Running Summarize Search Results with count number {ctx.state.count}.")
        return ReflectOnSearch()


@dataclass
class ReflectOnSearch(BaseNode[DeepState]):
    """
    Reflect on Search node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> WebSearch | FinalizeSummary:
        logger.debug(f"Running Reflect on Search with count number {ctx.state.count}.")
        if ctx.state.count < int(os.environ.get("MAX_RESEARCH_LOOPS", "10")):
            ctx.state.count += 1
            return WebSearch()
        else:
            return FinalizeSummary()


@dataclass
class FinalizeSummary(BaseNode[DeepState]):
    """
    Finalize Summary node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> End:
        logger.debug("Running Finalize Summary.")
        return End("End of deep research workflow.")


async def deepresearch() -> None:
    """
    Graph use
    """
    logger.info("Starting deep research workflow.")

    load_dotenv()

    # Define the agent graph
    graph = Graph(nodes=[WebSearch, SummarizeSearchResults, ReflectOnSearch, FinalizeSummary])

    # Run the agent graph
    state = DeepState(
        topic=os.environ.get("TOPIC", "petrichor"),
        count=1,
    )
    result = await graph.run(WebSearch(), state=state)
    logger.debug(f"Result: {result.output}")

    # Mermaid code
    mermaid_code = graph.mermaid_code(start_node=WebSearch())
    logger.debug(f"Mermaid graph:\n{mermaid_code}")


def main() -> None:
    """
    Main function containing the deep research workflow.
    """

    logger.info("Starting deep research.")
    asyncio.run(deepresearch())


if __name__ == "__main__":
    main()
