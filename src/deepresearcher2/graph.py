#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from deepresearcher2 import logger
from deepresearcher2.models import WebSearchQuery
from deepresearcher2.prompts import query_instructions
from deepresearcher2.utils import duckduckgo_search


async def deepresearch() -> None:
    """
    Deep research workflow.
    """
    load_dotenv()

    topic = os.environ.get("TOPIC", "petrichor")

    # LLM setup
    model_name = "llama3.3"
    # model_name = "firefunction-v2"
    # model_name = "mistral-nemo"
    ollama_model = OpenAIModel(
        model_name=model_name,
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    )

    # MCP server setup
    mcp_server_duckduckgo = MCPServerStdio("uvx", args=["duckduckgo-mcp-server"])

    # Agent setup
    # Note that we provide internet access to the query writing agent. This might be a bit circular.
    # TODO: Check whether this improves the queries or is just a waste of time.
    query_agent = Agent(
        model=ollama_model,
        # model="openai:gpt-4o",
        mcp_servers=[mcp_server_duckduckgo],
        output_type=WebSearchQuery,
        system_prompt=query_instructions,
        retries=5,
        instrument=True,
    )

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


# Data classes
@dataclass
class DeepState:
    topic: str = "petrichor"
    search_query: str = ""
    search_results: list[str] = field(default_factory=list)
    count: int = 0
    summary: str = ""


# Agents
ollama_model = OpenAIModel(
    # model_name="llama3.3",
    model_name="gemma3:4b",  # fast, for debugging purposes
    # model_name="gemma3:27b",
    # model_name="qwen2.5:7b",
    # model_name="mistral:7b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
)

agent = Agent(
    model=ollama_model,
    # model="openai:gpt-4o",
    output_type=str,
    system_prompt=query_instructions,
    instrument=True,
)


# Nodes
@dataclass
class WebSearch(BaseNode[DeepState]):
    """
    Web Search node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> SummarizeSearchResults:
        logger.debug(f"Running Web Search with count number {ctx.state.count}.")

        prompt = f"Research the topic {ctx.state.topic}."
        result = await agent.run(user_prompt=prompt)
        logger.debug(f"Web Search result:\n{result.output}")

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


async def deepresearch_2() -> None:
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
