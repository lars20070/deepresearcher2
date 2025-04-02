#!/usr/bin/env python3

import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from deepresearcher2 import logger


def riza_example() -> None:
    """
    Example by Riza team. https://riza.io
    https://youtu.be/2FsN4f4z2CY

    Returns:
        None
    """

    logfire.info("Starting Rizza example.")

    model = "llama3.3"
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

    user_message = "What is the capital of France?"
    result = agent.run_sync(user_message)

    logger.info(f"Result: {result.data}")


def main() -> None:
    """
    Main function for the script.

    Returns:
        None
    """

    logger.info("Starting main function.")
    logfire.info("Starting main function.")

    riza_example()


if __name__ == "__main__":
    main()
