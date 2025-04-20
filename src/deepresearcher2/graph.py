#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import format_as_xml
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from deepresearcher2.agents import query_agent, reflection_agent, summary_agent
from deepresearcher2.logger import logger
from deepresearcher2.models import DeepState, WebSearchSummary
from deepresearcher2.utils import duckduckgo_search


# Nodes
@dataclass
class WebSearch(BaseNode[DeepState]):
    """
    Web Search node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> SummarizeSearchResults:
        logger.debug(f"Running Web Search with count number {ctx.state.count}.")

        load_dotenv()
        topic = ctx.state.topic

        # Generate the query
        async with query_agent.run_mcp_servers():
            prompt = f"Please generate a web search query for the following topic: {topic}"
            result = await query_agent.run(prompt)
            query = result.output
            logger.debug(f"Web search query: {query}")

        # Run the search
        ctx.state.search_results = duckduckgo_search(
            query=query.query,
            max_results=int(os.environ.get("MAX_WEB_SEARCH_RESULTS", "2")),
            max_content_length=12000,  # maximum length of 12k characters
        )
        # for r in ctx.state.search_results:
        #     logger.debug(f"Search result title: {r.title}")
        #     logger.debug(f"Search result url: {r.url}")
        #     logger.debug(f"Search result content length: {len(r.content)}")
        #     logger.debug(f"Search result content:\n{r.content}")

        return SummarizeSearchResults()


@dataclass
class SummarizeSearchResults(BaseNode[DeepState]):
    """
    Summarize Search Results node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> ReflectOnSearch:
        logger.debug(f"Running Summarize Search Results with count number {ctx.state.count}.")

        @summary_agent.system_prompt
        def add_web_search_results() -> str:
            """
            Add web search results to the system prompt.
            """
            xml = format_as_xml(ctx.state.search_results, root_tag="search_results")
            return f"List of web search results:\n{xml}"

        # Generate the summary
        async with query_agent.run_mcp_servers():
            summary = await summary_agent.run(user_prompt=f"Please summarize the provided web search results for the topic {ctx.state.topic}.")
            logger.debug(f"Web search summary:\n{summary.output.summary}")

            # Append the summary to the list of all search summaries
            ctx.state.search_summaries = ctx.state.search_summaries or []
            ctx.state.search_summaries.append(WebSearchSummary(summary=summary.output.summary, aspect=summary.output.aspect))

        return ReflectOnSearch()


@dataclass
class ReflectOnSearch(BaseNode[DeepState]):
    """
    Reflect on Search node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> WebSearch | FinalizeSummary:
        logger.debug(f"Running Reflect on Search with count number {ctx.state.count}.")

        # xml = format_as_xml(ctx.state.search_summaries, root_tag="search_summaries")
        # logger.debug(f"Search summaries:\n{xml}")

        @reflection_agent.system_prompt
        def add_search_summaries() -> str:
            """
            Add search summaries to the system prompt.
            """
            xml = format_as_xml(ctx.state.search_summaries, root_tag="search_summaries")
            return f"List of search summaries:\n{xml}"

        # Reflect on the summaries so far
        async with query_agent.run_mcp_servers():
            result = await reflection_agent.run(user_prompt=f"Please reflect on the provided web search summaries for the topic {ctx.state.topic}.")
            logger.debug(f"Reflection result:\n{result.output}")

        # Flow control
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
        return End("End of deep research workflow.\n\n")


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
    # mermaid_code = graph.mermaid_code(start_node=WebSearch())
    # logger.debug(f"Mermaid graph:\n{mermaid_code}")


def main() -> None:
    """
    Main function containing the deep research workflow.
    """

    logger.info("Starting deep research.")
    asyncio.run(deepresearch())


if __name__ == "__main__":
    main()
