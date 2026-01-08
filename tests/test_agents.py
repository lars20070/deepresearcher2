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

    async def run_agent(provider: Provider, model_name: str) -> None:
        """Run an agent with the specified provider and model.

        Args:
            provider (Provider): The provider to use.
            model_name (str): The model name to use.
        """
        config = Config(provider=provider, model=model_name)
        model = create_model(config)
        agent = Agent(model=model, system_prompt="Be concise, reply with one sentence.")

        result = await agent.run('Where does "hello world" come from?')
        logger.debug(f"Result from agent: {result.output}")

    # We test each provider with a different model.
    models_for_testing = {
        Provider.ollama: "llama3.3",
        Provider.lmstudio: "qwen/qwen3-8b",
        Provider.openrouter: "meta-llama/llama-3.3-70b-instruct",
        Provider.openai: "gpt-4o",
        Provider.together: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        Provider.deepinfra: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    }

    # Check that all providers are covered.
    assert len(models_for_testing) == len(Provider)

    # Run agents for all providers
    for provider, model_name in models_for_testing.items():
        await run_agent(
            provider=provider,
            model_name=model_name,
        )
