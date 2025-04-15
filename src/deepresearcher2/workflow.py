#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.mcp import MCPServerStdio

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
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
class NodeA(BaseNode[int]):
    """
    Pass track number on.
    """

    track_number: int = 0

    async def run(self, ctx: GraphRunContext) -> BaseNode:
        logger.debug("Running Node A.")
        return NodeB(self.track_number)


@dataclass
class NodeB(BaseNode[int]):
    """
    Decision node.
    """

    track_number: int = 0

    async def run(self, ctx: GraphRunContext) -> BaseNode | End:
        logger.debug("Running Node B.")
        if self.track_number > 5:
            return End(f"Stop at Node B with track number {self.track_number}")
        else:
            return NodeC(self.track_number)


@dataclass
class NodeC(BaseNode[int]):
    """
    Not always executed.
    """

    track_number: int = 0

    async def run(self, ctx: GraphRunContext) -> End:
        logger.info("Running Node C.")
        return End(f"Stop at Node C with track number {self.track_number}")


async def deepresearch_2() -> None:
    """
    Graph use
    """
    logger.info("Starting deep research 2.")

    # Define the agent graph
    graph = Graph(nodes=[NodeA, NodeB, NodeC])

    # Run the agent graph
    result = await graph.run(start_node=NodeA(track_number=6))
    logger.debug(f"Result: {result.output}")


def main() -> None:
    """
    Main function containing the deep research workflow.
    """

    logger.info("Starting deep research.")
    asyncio.run(deepresearch_2())


if __name__ == "__main__":
    main()
