#!/usr/bin/env python3

import logfire
import rizaio
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

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
    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1",
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


def chat_with_python() -> None:
    """
    Chat interface with access to Python code execution tool.
    Example by Riza team. https://riza.io

    https://youtu.be/2FsN4f4z2CY
    """

    logfire.info("Starting chat with Python.")

    model = "llama3.3"
    # model = "qwen2.5:72b"
    # model = "qwq:32b"
    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1",
        ),
    )

    agent = Agent(
        ollama_model,
        system_prompt="You are a helpful assistant.",
        retries=5,
    )

    @agent.tool_plain
    def execute_code(code: str) -> str:
        """
        Execute Python code

        Use print() to write the output of your code to stdout.
        Use only the Python standard library and build in modules.
        For example, do not use pandas, but you can use csv.
        Use httpx to make http requests.

        Args:
            code (str): The code to execute.

        Returns:
            str: The output of the code execution.
        """
        load_dotenv()

        logfire.debug(f"Executing code:\n{code}")
        riza = rizaio.Riza()
        result = riza.command.exec(
            language="PYTHON",
            code=code,
        )

        if result.exit_code != 0:
            raise ModelRetry(result.stderr)
        if result.stdout == "":
            raise ModelRetry("Code executed successfully, but no output was returned. Ensure your code includes print statements for output.")

        logfire.debug(f"Execution output:\n{result.stdout}")
        return result.stdout

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
    Start the MCP server.

    Creates and runs an MCP server with a Claude 3.5 agent inside.
    https://ai.pydantic.dev/mcp/server/

    Test the response of the server with test_mcp_server()
    """
    load_dotenv()

    server = FastMCP("PydanticAI Server")
    server_agent = Agent(
        "anthropic:claude-3-5-haiku-latest",
        system_prompt="Always reply in rhyme.",
    )

    @server.tool()
    async def poet(theme: str) -> str:
        """Poem generator"""
        r = await server_agent.run(f"Write a poem about {theme}.")
        return r.data

    server.run()
