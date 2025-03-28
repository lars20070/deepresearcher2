#!/usr/bin/env python3

import pytest
from pydantic_ai import Agent

from deepresearcher2 import logger


@pytest.mark.example
@pytest.mark.paid
@pytest.mark.asyncio
async def test_basic_pydanticai(load_env: None) -> None:
    """
    Test basic pydanticai
    """
    logger.info("Testing basic pydanticai functionality.")

    agent = Agent(
        "google-gla:gemini-1.5-flash",
        system_prompt="Be concise, reply with one sentence.",
    )

    result = await agent.run('Where does "hello world" come from?')
    logger.info(f"Result from agent: {result.data}")
