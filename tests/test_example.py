#!/usr/bin/env python3

import os
import random
from datetime import date
from io import StringIO
from unittest.mock import patch

import logfire
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

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
