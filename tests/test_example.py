#!/usr/bin/env python3

import os
import random
from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Any
from unittest.mock import patch

import logfire
import pytest
from httpx import AsyncClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.mcp import MCPServerHTTP, MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from deepresearcher2 import basic_chat, chat_with_python, logger


@pytest.mark.example
@pytest.mark.paid
@pytest.mark.asyncio
async def test_pydanticai_agent(load_env: None) -> None:
    """
    Test the Agent() class with a cloud model
    https://ai.pydantic.dev/#why-use-pydanticai
    """
    logger.info("Testing PydanticAI Agent() class with a cloud model")

    agent = Agent(
        model="google-gla:gemini-1.5-flash",
        system_prompt="Be concise, reply with one sentence.",
    )

    result = await agent.run('Where does "hello world" come from?')
    logger.debug(f"Result from agent: {result.data}")


@pytest.mark.example
@pytest.mark.ollama
@pytest.mark.asyncio
async def test_pydanticai_ollama() -> None:
    """
    Test the Agent() class with a local Ollama model
    https://ai.pydantic.dev/models/#openai-compatible-models
    """
    logger.info("Testing PydanticAI Agent() class with a local Ollama model")

    class CityLocation(BaseModel):
        city: str
        country: str

    model = "llama3.3"
    # model = "qwq:32b"
    # model = "qwen2.5:72b"
    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1",
        ),
    )

    agent = Agent(
        ollama_model,
        result_type=CityLocation,
    )

    result = await agent.run("Where were the olympics held in 2012?")
    logger.debug(f"Result from agent: {result.data}")
    assert result.data.city == "London"

    usage = result.usage()
    logger.debug(f"Usage statistics: {usage}")
    assert usage.requests == 1
    assert usage.total_tokens > 0


@pytest.mark.example
def test_pydanticai_logfire(load_env: None) -> None:
    """
    Test the basic Logfire functionality
    https://ai.pydantic.dev/logfire/#using-logfire

    Note by default Logfire is disabled inside pytest (send_to_logfire=False)
    https://logfire.pydantic.dev/docs/reference/advanced/testing/
    """
    logfire.configure(
        token=os.environ.get("LOGFIRE_TOKEN"),
        send_to_logfire=True,
    )

    logfire.info("Hello, {place}!", place="World")

    with logfire.span("Asking the user their {question}", question="age"):
        # Simulate user input for testing
        user_input = str(random.randint(1900, 2000)) + "-04-16"
        dob = date.fromisoformat(user_input)
        logfire.debug("{dob=} {age=!r}", dob=dob, age=date.today() - dob)

    # Check the logfire output at https://logfire-eu.pydantic.dev/lars20070/deepresearcher2


@pytest.mark.ollama
def test_basic_chat() -> None:
    """
    Test the basic chat interface
    Note that we mock the user input but not the agent.
    """
    stdout_buffer = StringIO()

    with (
        patch(
            "builtins.input",
            side_effect=[
                "What is the capital of France?",
                "What is the capital of Germany?",
                "exit",
            ],
        ),
        patch("sys.stdout", new=stdout_buffer),
    ):
        basic_chat()

        output = stdout_buffer.getvalue()
        logger.debug(f"Complete output from basic chat: {output}")

        assert "Paris" in output
        assert "Berlin" in output


@pytest.mark.paid
@pytest.mark.ollama
def test_chat_with_python() -> None:
    """
    Test the chat interface with access to Python code execution tool
    """
    stdout_buffer = StringIO()

    with (
        patch(
            "builtins.input",
            side_effect=[
                "What is the largest gap between two successive prime numbers under 10000?",
                "Please determine the prime factorisation of 889966.",
                "exit",
            ],
        ),
        patch("sys.stdout", new=stdout_buffer),
    ):
        chat_with_python()

        output = stdout_buffer.getvalue()
        logger.debug(f"Complete output from basic chat: {output}")

        assert "36" in output
        assert "5779" in output


@pytest.mark.example
@pytest.mark.paid
@pytest.mark.asyncio
async def test_weather_agent(load_env: None) -> None:
    """
    Test Ollama agent with two tools.
    Note that geocode.maps.co has strict request limits. '429 Too Many Requests' is likely.

    Slightly modified example from the Pydantic documentation.
    https://ai.pydantic.dev/examples/weather-agent/
    """

    @dataclass
    class Deps:
        client: AsyncClient
        weather_api_key: str | None
        geo_api_key: str | None

    # TODO: Replace GPT-4o by any Ollama model
    # Model response is <|python_tag|>get_lat_lng(args=["Zurich"])
    # Maybe look into this issue. https://github.com/pydantic/pydantic-ai/issues/437
    # ollama_model = OpenAIModel(
    #     model_name="llama3.3",
    #     provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    # )

    weather_agent = Agent(
        # model=ollama_model,
        model="openai:gpt-4o",
        system_prompt=(
            "Be concise, reply with one sentence. "
            "Use the `get_lat_lng` tool to get the latitude and longitude of the locations, "
            "then use the `get_weather` tool to get the weather."
        ),
        deps_type=Deps,
        retries=2,
        instrument=True,
    )

    @weather_agent.tool
    async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> dict[str, float]:
        """Get the latitude and longitude of a location.

        Args:
            ctx: The context.
            location_description: A description of a location.

        Returns:
            A dictionary containing latitude and longitude.
        """
        if ctx.deps.geo_api_key is None:
            # if no API key is provided, return a dummy response (London)
            return {"lat": 51.1, "lng": -0.1}

        params = {
            "q": location_description,
            "api_key": ctx.deps.geo_api_key,
        }
        with logfire.span("Calling Geocode API", params=params) as span:
            r = await ctx.deps.client.get("https://geocode.maps.co/search", params=params)
            r.raise_for_status()
            data = r.json()
            span.set_attribute("response", data)

        if data:
            return {"lat": data[0]["lat"], "lng": data[0]["lon"]}
        else:
            raise ModelRetry("Could not find the location.")

    @weather_agent.tool
    async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
        """Get the weather at a location.

        Args:
            ctx: The context.
            lat: Latitude of the location.
            lng: Longitude of the location.

        Returns:
            A dictionary containing temperature and weather description.
        """
        if ctx.deps.weather_api_key is None:
            # if no API key is provided, return a dummy response
            return {"temperature": "21 °C", "description": "Sunny"}

        params = {
            "apikey": ctx.deps.weather_api_key,
            "location": f"{lat},{lng}",
            "units": "metric",
        }
        with logfire.span("calling weather API", params=params) as span:
            r = await ctx.deps.client.get("https://api.tomorrow.io/v4/weather/realtime", params=params)
            r.raise_for_status()
            data = r.json()
            span.set_attribute("response", data)

        values = data["data"]["values"]
        # https://docs.tomorrow.io/reference/data-layers-weather-codes
        code_lookup = {
            1000: "Clear, Sunny",
            1100: "Mostly Clear",
            1101: "Partly Cloudy",
            1102: "Mostly Cloudy",
            1001: "Cloudy",
            2000: "Fog",
            2100: "Light Fog",
            4000: "Drizzle",
            4001: "Rain",
            4200: "Light Rain",
            4201: "Heavy Rain",
            5000: "Snow",
            5001: "Flurries",
            5100: "Light Snow",
            5101: "Heavy Snow",
            6000: "Freezing Drizzle",
            6001: "Freezing Rain",
            6200: "Light Freezing Rain",
            6201: "Heavy Freezing Rain",
            7000: "Ice Pellets",
            7101: "Heavy Ice Pellets",
            7102: "Light Ice Pellets",
            8000: "Thunderstorm",
        }
        return {
            "temperature": f"{values['temperatureApparent']:0.0f}°C",
            "description": code_lookup.get(values["weatherCode"], "Unknown"),
        }

    async with AsyncClient() as client:
        # Create a free API key at https://www.tomorrow.io/weather-api/
        weather_api_key = os.getenv("WEATHER_API_KEY")
        # Create a free API key at https://geocode.maps.co/
        geo_api_key = os.getenv("GEO_API_KEY")
        deps = Deps(
            client=client,
            weather_api_key=weather_api_key,
            geo_api_key=geo_api_key,
        )
        result = await weather_agent.run("What is the weather like in Zurich and in Wiltshire?", deps=deps)
        logger.debug(f"Response from weather agent: {result.data}")

    assert weather_agent.model.model_name == "gpt-4o"
    assert "Zurich" in result.data


@pytest.mark.paid
@pytest.mark.example
@pytest.mark.asyncio
async def test_agent_delegation(load_env: None) -> None:
    """
    Test the agent delegation functionality

    Example from the Pydantic documentation.
    https://ai.pydantic.dev/multi-agent-applications/#agent-delegation-and-dependencies
    """

    # model = "llama3.3"
    # ollama_model = OpenAIModel(
    #     model_name=model,
    #     provider=OpenAIProvider(
    #         base_url="http://localhost:11434/v1",
    #     ),
    # )

    @dataclass
    class ClientAndKey:
        http_client: AsyncClient
        api_key: str

    # TODO: The agents cannot use the Ollama Llama3.3 model. Why?
    joke_selection_agent = Agent(
        model="openai:gpt-4o",
        # model=ollama_model,
        deps_type=ClientAndKey,
        system_prompt=(
            "Use the `joke_factory` tool to generate some jokes on the given subject, then choose the best. You must return just a single joke."
        ),
        instrument=True,
    )

    joke_generation_agent = Agent(
        model="openai:gpt-4o",
        # model=ollama_model,
        deps_type=ClientAndKey,
        result_type=list[str],
        system_prompt=('Use the "get_jokes" tool to get some jokes on the given subject, then extract each joke into a list.'),
        instrument=True,
    )

    @joke_selection_agent.tool
    async def joke_factory(ctx: RunContext[ClientAndKey], count: int) -> list[str]:
        r = await joke_generation_agent.run(
            f"Please generate {count} jokes.",
            deps=ctx.deps,
            usage=ctx.usage,
        )
        return r.data[:5]

    @joke_generation_agent.tool
    async def get_jokes(ctx: RunContext[ClientAndKey], count: int) -> str:
        response = await ctx.deps.http_client.get(
            "https://example.com",
            params={"count": count},
            headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
        )
        response.raise_for_status()
        return response.text

    async with AsyncClient() as client:
        deps = ClientAndKey(client, "foobar")
        result = await joke_selection_agent.run("Tell me a joke.", deps=deps)
        logger.debug(result.data)
        logger.debug(result.usage())

        assert isinstance(result.data, str)
        assert result.usage().requests > 0


@pytest.mark.example
@pytest.mark.asyncio
async def test_pydantic_evals() -> None:
    """
    Test the functionality of pydantic-evals.
    We evaluate and score the response of the model, see evaluate() method.
    https://ai.pydantic.dev/evals
    """

    case_1 = Case(
        name="simple_case",
        inputs="What is the capital of France?",
        expected_output="Paris",
        metadata={"difficulty": "easy"},
    )

    case_2 = Case(
        name="another_simple_case",
        inputs="What is the capital of Germany?",
        expected_output="Berlin",
        metadata={"difficulty": "easy"},
    )

    class MyEvaluator(Evaluator[str, str]):
        def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
            # Score the string output
            if ctx.output == ctx.expected_output:
                # Exact match
                return 1.0
            elif isinstance(ctx.output, str) and ctx.expected_output.lower() in ctx.output.lower():
                # Expected output is a substring of the output
                return 0.8
            else:
                # Total failure
                return 0.0

    dataset = Dataset(
        cases=[case_1, case_2],
        evaluators=[
            IsInstance(type_name="str"),  # Vanilla evaluator
            MyEvaluator(),  # Custom evaluator
        ],
    )
    logger.debug(f"Complete evals dataset: {dataset}")

    # Check structure of test dataset
    assert dataset.cases[0].inputs == "What is the capital of France?"
    assert dataset.cases[0].expected_output == "Paris"
    assert dataset.cases[1].inputs == "What is the capital of Germany?"
    assert dataset.cases[1].expected_output == "Berlin"

    async def guess_city(question: str) -> str:
        # Simulate a model response
        return "Paris"

    report = await dataset.evaluate(guess_city)
    report.print(
        include_input=True,
        include_output=True,
        include_durations=False,
    )
    logger.debug(f"Complete evaluation report: {report}")


@pytest.mark.skip(reason="Requires MCP server to be started first.")
@pytest.mark.paid
@pytest.mark.example
@pytest.mark.asyncio
async def test_mcp_sse_client(load_env: None) -> None:
    """
    Test the Pydantic MCP SSE client.
    https://ai.pydantic.dev/mcp/client/#sse-client

    Please start the MCP server first.
    deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python sse
    """

    # model = "llama3.3"
    # ollama_model = OpenAIModel(
    #     model_name=model,
    #     provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    # )

    # MCP server providing run_python_code tool
    mcp_server = MCPServerHTTP(url="http://localhost:3001/sse")
    agent = Agent(
        model="openai:gpt-4o",
        # model=ollama_model,
        mcp_servers=[mcp_server],
        instrument=True,
    )

    async with agent.run_mcp_servers():
        result = await agent.run("How many days between 2000-01-01 and 2025-03-18?")
        logger.debug(f"Result: {result.data}")

        # 9,208 days is the correct answer.
        assert "9,208 days" in result.data


@pytest.mark.paid
@pytest.mark.example
@pytest.mark.asyncio
async def test_mcp_stdio_client(load_env: None) -> None:
    """
    Test the Pydantic MCP SSE client.
    https://ai.pydantic.dev/mcp/client/#sse-client

    Note that unlike in the SSE mode, the MCP server starts up automatically.
    """

    # model = "llama3.3"
    # ollama_model = OpenAIModel(
    #     model_name=model,
    #     provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    # )

    # MCP server providing run_python_code tool
    mcp_server = MCPServerStdio(
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
    agent = Agent(
        model="openai:gpt-4o",
        # model=ollama_model,
        mcp_servers=[mcp_server],
        instrument=True,
    )

    async with agent.run_mcp_servers():
        result = await agent.run("How many days between 2000-01-01 and 2025-03-18?")
        logger.debug(f"Result: {result.data}")

        # 9,208 days is the correct answer.
        assert "9,208 days" in result.data


@pytest.mark.paid
@pytest.mark.example
@pytest.mark.asyncio
async def test_mcp_server(load_env: None) -> None:
    """
    Test the MCP server functionality defined in deepresearcher2.examples.mcp_server()

    The MCP server wraps a Claude 3.5 agent which generates poems.
    The MCP server ist started automatically.
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "mcpserver"],
        env=os.environ,
    )

    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("poet", {"theme": "socks"})
        logger.debug(f"Complete poem:\n{result.content[0].text}")
        assert "socks" in result.content[0].text


@pytest.mark.example
@pytest.mark.asyncio
async def test_pydantic_graph() -> None:
    """
    Define a simple graph and test its traversal.
    See flow chart in tests/README.md
    https://youtu.be/WFvugLf_760

    Node A simply passes the track number on to the next node.
    Node B decides whether to continue to node C or stop.
    Node C passes the track number to the final result.

    track number <= 5: nodes A, B and C executed
    track number > 5: nodes A and B executed, but not C
    """

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

    logger.info("Testing Pydantic Graph")

    # Define the agent graph
    graph = Graph(nodes=[NodeA, NodeB, NodeC])

    # Run the agent graph
    result_1 = await graph.run(start_node=NodeA(track_number=1))
    logger.debug(f"Result: {result_1.output}")
    assert "Node C" in result_1.output

    result_2 = await graph.run(start_node=NodeA(track_number=6))
    logger.debug(f"Result: {result_2.output}")
    assert "Node B" in result_2.output
