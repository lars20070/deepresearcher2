#!/usr/bin/env python3

import pytest
from dotenv import load_dotenv
from pydantic_ai import Agent

from deepresearcher2.agents import create_model
from deepresearcher2.config import Config, Provider
from deepresearcher2.logger import logger

load_dotenv()


@pytest.mark.paid
@pytest.mark.ollama
@pytest.mark.lmstudio
@pytest.mark.asyncio
async def test_create_model() -> None:
    """
    Test create_model() functionality for all providers.
    """
    logger.info("Testing create_model() functionality for all providers.")

    # Test creation of models without running them
    for provider in Provider:
        logger.debug(f"Provider: {provider.value}")

        config = Config()
        config.provider = provider
        config.model = "dummy-model"

        model = create_model(config)
        logger.debug(f"Created model: {model}")
        assert model is not None

    # Test creation of models and running a simple agent query
    async def run_agent(provider: Provider, model_name: str) -> None:
        config = Config()
        config.provider = provider
        config.model = model_name

        model = create_model(config)

        agent = Agent(
            model=model,
            system_prompt="Be concise, reply with one sentence.",
        )

        result = await agent.run('Where does "hello world" come from?')
        logger.debug(f"Result from agent: {result.output}")

    # Test Ollama provider
    await run_agent(
        provider=Provider.ollama,
        model_name="llama3.3",
    )

    # Test OpenAI provider
    await run_agent(
        provider=Provider.openai,
        model_name="gpt-4o",
    )
