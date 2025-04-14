#!/usr/bin/env python3
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from deepresearcher2 import logger


@dataclass
class DeepState:
    topic: str | None = None
    loop_count: int = 0


def deepresearch() -> None:
    """
    Deep research workflow.
    """
    model = "llama3.3"
    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1",
        ),
    )

    agent = Agent(
        model=ollama_model,
        # model="openai:gpt-4o",
        result_type=str,
        instrument=True,
    )
    logger.debug(f"Agent: {agent}")

    result = agent.run_sync("What is the capital of France?")
    logger.debug(f"Result: {result.data}")


def main() -> None:
    """
    Main function containing the deep research workflow.
    """

    logger.info("Starting deep research.")
    deepresearch()


if __name__ == "__main__":
    main()
