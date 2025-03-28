#!/usr/bin/env python3

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from deepresearcher2 import logger


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

    ollama_model = OpenAIModel(
        model_name="llama3.3",
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
