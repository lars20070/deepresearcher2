#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from deepresearcher2 import logger


@dataclass
class DeepState:
    topic: str | None = None
    loop_count: int = 0


async def deepresearch() -> None:
    """
    Deep research workflow.
    """
    load_dotenv()

    # LLM setup
    model = "llama3.3"
    # model = "firefunction-v2"
    # model = "mistral-nemo"
    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1",
        ),
    )

    # MCP setup
    mcp_server_python = MCPServerStdio(
        "deno",
        args=[
            "run",
            "-N",
            "-R=node_modules",
            "-W=node_modules",
            "--node-modules-dir=auto",
            "jsr:@pydantic/mcp-run-python",
            "stdio",
        ],
    )

    mcp_server_duckduckgo = MCPServerStdio(
        "uvx",
        args=[
            "duckduckgo-mcp-server",
        ],
    )

    agent = Agent(
        model=ollama_model,
        # model="openai:gpt-4o",
        mcp_servers=[
            mcp_server_python,
            mcp_server_duckduckgo,
        ],
        result_type=str,
        instrument=True,
    )
    logger.debug(f"Agent: {agent}")

    async with agent.run_mcp_servers():
        # prompt = "What is the capital of France?"
        prompt = "What time is it in Zurich?"
        result = await agent.run(prompt)
        logger.debug(f"Result: {result.data}")


@dataclass
class WebSearch(BaseNode[int]):
    """
    Web Search node.
    """

    count: int = 0

    async def run(self, ctx: GraphRunContext) -> BaseNode:
        logger.debug(f"Running Web Search with count number {self.count}.")
        return SummarizeSearchResults(self.count)


@dataclass
class SummarizeSearchResults(BaseNode[int]):
    """
    Summarize Search Results node.
    """

    count: int = 0

    async def run(self, ctx: GraphRunContext) -> BaseNode:
        logger.debug(f"Running Summarize Search Results with count number {self.count}.")
        return ReflectOnSearch(self.count)


@dataclass
class ReflectOnSearch(BaseNode[int]):
    """
    Reflect on Search node.
    """

    count: int = 0

    async def run(self, ctx: GraphRunContext) -> BaseNode:
        logger.debug(f"Running Reflect on Search with count number {self.count}.")
        if self.count >= 10:
            return FinalizeSummary(self.count)
        else:
            return WebSearch(self.count + 1)


@dataclass
class FinalizeSummary(BaseNode[int]):
    """
    Finalize Summary node.
    """

    count: int = 0

    async def run(self, ctx: GraphRunContext) -> End:
        logger.debug("Running Finalize Summary.")
        return End("End of deep research workflow.")


async def deepresearch_2() -> None:
    """
    Graph use
    """
    logger.info("Starting deep research workflow.")

    # Define the agent graph
    graph = Graph(nodes=[WebSearch, SummarizeSearchResults, ReflectOnSearch, FinalizeSummary])

    # Run the agent graph
    result = await graph.run(start_node=WebSearch(count=1))
    logger.debug(f"Result: {result.output}")

    # Mermaid code
    mermaid_code = graph.mermaid_code(start_node=WebSearch())
    logger.debug(f"Mermaid graph:\n{mermaid_code}")


def main() -> None:
    """
    Main function containing the deep research workflow.
    """

    logger.info("Starting deep research.")
    asyncio.run(deepresearch_2())


if __name__ == "__main__":
    main()
