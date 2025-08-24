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
    async def evaluate(self, ctx: EvaluatorContext[Any, Any]) -> float:
        return float(ctx.output == ctx.expected_output)


class ExactMatchAny(Evaluator[Any, list[Any]]):
    async def evaluate(self, ctx: EvaluatorContext[Any, list[Any]]) -> float:
        return float(ctx.output in ctx.expected_output)


async def eval_codenames(model: str = "qwen2.5:72b", max_cases: int | None = None) -> None:
    """
    Runs evaluation for codenames benchmark.

    Args:
        model (str): The model to use for evaluation.
        max_cases (int | None): The maximum number of cases to evaluate. Defaults to None.

    Returns:
        float: The evaluation score.
    """
    logger.info("Runs evaluation for codenames benchmark.")

    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url=f"{config.ollama_host}/v1",
        ),
    )

    # Agent for word detection
    word_detector = Agent(
        ollama_model,
        output_type=str,
        system_prompt="Find the associated word or words described below. Respond with a comma-separated list of small caps words.",
    )

    async def transform_text(text: str) -> Response:
        r = await word_detector.run(text)
        return r.output

    # Load the benchmark cases
    path = Path("benchmarks/codenames/task.json")
    dataset = Dataset[str, str, Any].from_file(path)
    cases = dataset.cases
    if max_cases is not None:
        cases = cases[:max_cases]
    # Add the benchmark evaluators
    dataset = Dataset[str, str, Any](
        cases=cases,
        evaluators=[
            IsInstance(type_name="str"),  # Pointless here since the evaluation crashes anyhow if the output type is incorrect.
            ExactMatch(),
        ],
    )

    # Run the evaluation
    report = await dataset.evaluate(transform_text)
    # report.print(include_input=True, include_output=True, include_durations=True)
    logger.debug(f"Complete evaluation report:\n{report}")

    score = report.averages().scores.get("ExactMatch", 0)
    logger.info(f"Evaluation score: {score}")

    return score


async def eval_darkhurmordetection(model: str = "qwen2.5:72b", max_cases: int | None = None) -> None:
    """
    Runs evaluation for dark humor detection.

    Tested with the following models:
    * llama3.3
    * qwq:32b  # Not reliable. Does not respond with conform JSON. Let the model respond with free form `str` instead.
    * qwen2.5:72b
    * qwen3:30b
    * magistral:latest  # Not reliable.

    Args:
        model (str): The model to use for evaluation.
        max_cases (int | None): The maximum number of cases to evaluate. Defaults to None.

    Returns:
        float: The evaluation score.
    """
    logger.info("Runs evaluation for dark humor detection.")

    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url=f"{config.ollama_host}/v1",
        ),
    )

    # Agent for joke detection
    joke_detector = Agent(
        ollama_model,
        output_type=Response,
        system_prompt="Determine whether the text is a joke or not. Respond with 'joke' or 'no joke' in JSON format.",
    )

    async def transform_text(text: str) -> Response:
        r = await joke_detector.run(text)
        return r.output

    # Load the benchmark cases
    path = Path("benchmarks/dark_humor_detection/task.json")
    dataset = Dataset[str, Response, Any].from_file(path)
    cases = dataset.cases
    if max_cases is not None:
        cases = cases[:max_cases]
    # Add the benchmark evaluators
    dataset = Dataset[str, Response, Any](
        cases=cases,
        evaluators=[
            IsInstance(type_name="Response"),  # Pointless here since the evaluation crashes anyhow if the output type is incorrect.
            ExactMatch(),
        ],
    )

    # Run the evaluation
    report = await dataset.evaluate(transform_text)
    # report.print(include_input=True, include_output=True, include_durations=True)
    logger.debug(f"Complete evaluation report:\n{report}")

    score = report.averages().scores.get("ExactMatch", 0)
    logger.info(f"Evaluation score: {score}")

    return score


async def eval_rephrase(model: str = "qwen2.5:72b", max_cases: int | None = None) -> None:
    """
    Runs evaluation for rephrase benchmark

    Args:
        model (str): The model to use for evaluation.
        max_cases (int | None): The maximum number of cases to evaluate. Defaults to None.

    Returns:
        float: The evaluation score.
    """
    logger.info("Runs evaluation for rephrase benchmark.")

    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url=f"{config.ollama_host}/v1",
        ),
    )

    # Agent for rephrasing
    rephraser = Agent(
        ollama_model,
        output_type=str,
        system_prompt=(
            "Rephrase the given sentence so that it retains its meaning, but contains the given keyword. Answer ONLY with the rephrased sentence."
        ),
    )

    async def transform_text(text: str) -> str:
        r = await rephraser.run(text)
        return r.output

    # Load the benchmark cases
    path = Path("benchmarks/rephrase/task.json")
    dataset = Dataset[str, list[str], Any].from_file(path)
    cases = dataset.cases
    if max_cases is not None:
        cases = cases[:max_cases]
    # Add the benchmark evaluators
    dataset = Dataset[str, list[str], Any](
        cases=cases,
        evaluators=[
            IsInstance(type_name="str"),  # Pointless here since the evaluation crashes anyhow if the output type is incorrect.
            ExactMatchAny(),
        ],
    )

    # Run the evaluation
    report = await dataset.evaluate(transform_text)
    # report.print(include_input=True, include_output=True, include_durations=True)
    logger.debug(f"Complete evaluation report:\n{report}")

    score = report.averages().scores.get("ExactMatchAny", 0)
    logger.info(f"Evaluation score: {score}")

    return score


def main() -> None:
    """
    Main function running evaluations.
    """
    logger.info("Run evaluation.")
    model = "qwen2.5:72b"
    max_cases = 10
    asyncio.run(eval_codenames(model=model, max_cases=max_cases))
    asyncio.run(eval_darkhurmordetection(model=model, max_cases=max_cases))
    asyncio.run(eval_rephrase(model=model, max_cases=max_cases))


if __name__ == "__main__":
    main()
