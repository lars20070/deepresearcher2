#!/usr/bin/env python3

import fastmcp
import logfire
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import config

load_dotenv()

# Most code examples can be found in tests/test_example.py
# The methods here are an exception. They can be executed as scripts via [project.scripts] in pyproject.toml


def basic_chat() -> None:
    """
    Basic chat interface with the agent.
    https://youtu.be/2FsN4f4z2CY
    """

    logfire.info("Starting basic chat.")

    model = "llama3.3"
    # model = "qwen2.5:72b"
    # model = "qwq:32b"
    ollama_model = OpenAIChatModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url=f"{config.ollama_host}/v1",
        ),
    )

    agent = Agent(
        ollama_model,
        system_prompt="You are a helpful assistant.",
    )
    # Instrument the agent with Logfire
    # i.e. this will log the most important events automatically
    logfire.instrument_pydantic_ai(agent)

    result = None
    while True:
        user_message = input(">>> ")
        if user_message.lower() in {"exit", "quit", "bye", "stop"}:
            break

        result = agent.run_sync(
            user_message,
            message_history=result.all_messages() if result else None,
        )
        print(result.output)


def mcp_server() -> None:
    """
    Start the MCP server using the `mcp` package.

    Creates and runs an MCP server with a Claude 3.5 agent inside.
    https://ai.pydantic.dev/mcp/server/

    Test the response of the server with test_mcp_server()
    """
    server = FastMCP("PydanticAI Server")
    server_agent = Agent(
        "anthropic:claude-3-5-haiku-latest",
        system_prompt="Always reply in rhyme.",
    )

    @server.tool()
    async def poet(theme: str) -> str:
        """Poem generator"""
        r = await server_agent.run(f"Write a poem about {theme}.")
        return r.output

    server.run()


def mcp_server_stdio() -> None:
    """
    Start the MCP server using the `fastmcp` package.

    Creates and runs an MCP server with a Claude 3.5 agent inside.
    Test the response of the server in test_mcp_server_stdio()
    """
    server = fastmcp.FastMCP("PydanticAI Server")
    server_agent = Agent(
        "anthropic:claude-3-5-haiku-latest",
        system_prompt="Always reply in rhyme.",
    )

    @server.tool
    async def poet(theme: str) -> str:
        r = await server_agent.run(f"Write a poem about {theme}.")
        return r.output

    server.run()
