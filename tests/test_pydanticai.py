#!/usr/bin/env python3

import pytest
from pydantic_ai import Agent

from deepresearcher2 import logger


@pytest.mark.example
@pytest.mark.paid
@pytest.mark.asyncio
async def test_pydanticai_agent(load_env: None) -> None:
    """
    Test the Agent() class with a cloud model
    """
    logger.info("Testing PydanticAI Agent() class with a cloud model")

    agent = Agent(
        model="google-gla:gemini-1.5-flash",
        system_prompt="Be concise, reply with one sentence.",
    )

    result = await agent.run('Where does "hello world" come from?')
    logger.debug(f"Result from agent: {result.data}")


@pytest.mark.example
@pytest.mark.paid
@pytest.mark.asyncio
async def test_pydanticai_agent_2(load_env: None) -> None:
    """
    Test the Agent() class with a cloud model
    """
    logger.info("Testing PydanticAI Agent() class with a cloud model")

    agent = Agent(
        model="google-gla:gemini-1.5-flash",
        system_prompt="Be concise, reply with one sentence.",
    )

    result = await agent.run('Where does "hello world" come from?')
    logger.debug(f"Result from agent: {result.data}")
