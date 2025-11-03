#!/usr/bin/env python3
import textwrap

import fastmcp
import logfire
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from deepresearcher2.logger import logger

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
    Test the response of the server in test_mcp_server_stdio().
    Or add the MCP server to the Claude Desktop app by changing its config.
    ~/Library/Application Support/Claude/claude_desktop_config.json

    {
        "mcpServers": {
            "mcpserver_stdio": {
                "command": "uv",
                "args": [
                    "--directory",
                    "/Users/lars/Code/deepresearcher2",
                    "run",
                    "mcpserver_stdio"
                ]
            }
        }
    }
    """
    server = fastmcp.FastMCP("PydanticAI Server")
    server_agent = Agent(
        "anthropic:claude-3-5-haiku-latest",
        system_prompt="Always reply in rhyme.",
    )

    @server.tool
    async def poet(theme: str) -> str:
        logger.info(f"Calling 'poet' tool with theme: {theme}")
        r = await server_agent.run(f"Write a poem about {theme} and the Golden Gate Bridge.")
        logger.debug(f"Poem generated:\n{r.output}")
        return str(r.output)

    @server.prompt
    def poem_prompt(theme: str) -> str:
        logger.info(f"Calling 'poem_prompt' with theme: {theme}")
        prompt = f"Write a beautiful poem about {theme} and the Eiffel Tower."
        logger.debug(f"Prompt generated:\n{prompt}")
        return prompt

    @server.resource("poetry://guidelines")
    def poetry_guidelines() -> str:
        logger.info("Serving poetry guidelines resource")
        guidelines = textwrap.dedent("""# Poetry Examples and Guidelines

        ## Example 1: Nature Poem
        The morning dew upon the grass,
        Reflects the sun as moments pass,
        A gentle breeze through trees does flow,
        Nature's beauty all aglow.

        ## Example 2: Technology Poem
        In circuits bright and code so clean,
        The future's built on what we've seen,
        Through silicon and logic gates,
        Innovation accelerates.

        ## Poetry Writing Guidelines
        - Use vivid imagery and sensory details
        - Maintain consistent rhythm and meter
        - Employ rhyme schemes (ABAB, AABB, etc.)
        - Create emotional resonance
        - Use metaphors and similes effectively
        - End with a memorable conclusion

        ## Common Themes
        - Nature and seasons
        - Love and relationships
        - Technology and progress
        - Time and memory
        - Dreams and aspirations
        """)
        return guidelines

    server.run()
