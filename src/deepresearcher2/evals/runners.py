#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Dataset

from deepresearcher2.config import config
from deepresearcher2.logger import logger


async def run() -> None:
    """
    Run dark humor detection.
    """
    logger.info("Run dark humor detection.")

    # Model for both recipe and judge
    model = "llama3.3"
    # model = "qwq:32b"
    # model = "qwen2.5:72b"
    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url=f"{config.ollama_host}/v1",
        ),
    )

    # Agent for recipe generation
    joke_detector = Agent(
        ollama_model,
        output_type=bool,
        system_prompt="Determine whether the text is a joke or not.",
    )

    async def transform_text(text: str) -> bool:
        r = await joke_detector.run(text)
        return r.output

    path = Path("data/dark_humor_detection/task.json")
    dataset = Dataset[str, bool, Any].from_file(path)

    # Run the evaluation
    report = await dataset.evaluate(transform_text)
    report.print(
        include_input=True,
        include_output=True,
        include_durations=True,
    )
    logger.debug(f"Complete evaluation report:\n{report}")


def main() -> None:
    """
    Main function running evaluations.
    """
    logger.info("Run evaluation.")
    asyncio.run(run())


if __name__ == "__main__":
    main()
