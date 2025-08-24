#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

from deepresearcher2.config import config
from deepresearcher2.evals.import_bigbench import Response
from deepresearcher2.logger import logger


class ExactMatch(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext[Response, Response]) -> float:
        if ctx.output == ctx.expected_output:
            return 1.0
        else:
            return 0.0


async def run() -> None:
    """
    Run dark humor detection.
    """
    logger.info("Run dark humor detection.")

    # Model for both recipe and judge
    # model = "llama3.3"
    # model = "qwq:32b"  # Not reliable. Does not respond with conform JSON. Let the model respond with free form `str` instead.
    model = "qwen2.5:72b"
    # model = "qwen3:30b"
    # model = "magistral:latest"  # Not reliable.
    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url=f"{config.ollama_host}/v1",
        ),
    )

    # Agent for recipe generation
    joke_detector = Agent(
        ollama_model,
        output_type=Response,
        system_prompt="Determine whether the text is a joke or not. Respond with 'joke' or 'no joke' in JSON format.",
    )

    async def transform_text(text: str) -> Response:
        r = await joke_detector.run(text)
        return r.output

    path = Path("data/dark_humor_detection/task.json")
    dataset = Dataset[str, Response, Any].from_file(path)
    dataset = Dataset[str, Response, Any](
        cases=dataset.cases,
        evaluators=[
            IsInstance(type_name="Response"),
            ExactMatch(),
        ],
    )

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
