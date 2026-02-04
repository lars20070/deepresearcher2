#!/usr/bin/env python3
from __future__ import annotations as _annotations

import random  # noqa: F401
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: F401
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: F401
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset

from deepresearcher2.agents import EVALUATION_AGENT, model  # noqa: F401
from deepresearcher2.config import config
from deepresearcher2.logger import logger
from deepresearcher2.plugin import AssayContext, PairwiseEvaluator


def generate_evaluation_cases() -> Dataset[dict[str, str], str, Any]:
    """
    Generate a list of Cases containing topics as input.
    """
    logger.info("Creating new assay dataset.")

    topics = [
        "pangolin trafficking networks",
        "molecular gastronomy",
        "dark kitchen economics",
        "kintsugi philosophy",
        "nano-medicine delivery systems",
        "Streisand effect dynamics",
        "Anne Brorhilker",
        "bioconcrete self-healing",
        "bacteriophage therapy revival",
        "Habsburg jaw genetics",
    ]

    cases: list[Case[dict[str, str], str, Any]] = []
    for idx, topic in enumerate(topics):
        logger.info(f"Case {idx + 1} / {len(topics)} with topic: {topic}")
        case = Case(
            name=f"case_{idx:03d}",
            inputs={"topic": topic},
            expected_output="",
        )
        cases.append(case)

    return Dataset[dict[str, str], str, Any](cases=cases)


# Model for the Bradley-Terry evaluation agent
evaluation_model = OpenAIChatModel(
    model_name="Qwen/Qwen2.5-72B-Instruct",
    provider=OpenAIProvider(
        base_url=config.deepinfra_base_url,
        api_key=config.deepinfra_api_key,
    ),
)


@pytest.mark.skip(reason="Run only locally with DeepInfra cloud inference. PROVIDER='deepinfra' MODEL='Qwen/Qwen2.5-72B-Instruct'")
@pytest.mark.assay(
    generator=generate_evaluation_cases,
    evaluator=PairwiseEvaluator(
        model=evaluation_model,
        criterion="Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
    ),
    # evaluator=BradleyTerryEvaluator(
    #     model=evaluation_model,
    #     criterion="Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
    #     max_standard_deviation=2.1,
    # ),
)
@pytest.mark.asyncio
@pytest.mark.usefixtures("timer_for_tests")
async def test_search_queries(context: AssayContext) -> None:
    """
    Run the agent workflow once.

    Args:
        context: The assay context containing the evaluation dataset `context.dataset`and other information.
    """

    logger.debug(f"assay path: {context.path}")
    logger.debug(f"assay dataset: {context.dataset}")

    # Agent for generating search queries using a local Ollama server
    model_for_queries = OpenAIChatModel(
        model_name="qwen3:14b",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),  # Local Ollama server
    )
    # model_for_queries = model  # Use the model defined in .env (Not possible for VCR recording!)

    query_agent = Agent(
        model=model_for_queries,
        output_type=str,
        system_prompt="Please generate a concise web search query for the given research topic. You must respond only in English."
        + " Never use Chinese characters or any non-English text. Reply with ONLY the query string. Do NOT use quotes.",
        retries=5,
        instrument=True,
    )

    logger.info("Use case for EvalTournament, EvalGame and EvalPlayer classes.")

    # Generate model outputs
    for case in context.dataset.cases:
        logger.info(f"Case {case.name} with topic: {case.inputs['topic']}")

        # prompt = f"Please generate a useful search query for the following research topic: <TOPIC>{case.inputs['topic']}</TOPIC>"
        prompt = (
            f"Please generate a very creative search query for the research topic: <TOPIC>{case.inputs['topic']}</TOPIC>\n"
            "The query should show genuine originality and interest in the topic. AVOID any generic or formulaic phrases.\n\n"
            "Examples of formulaic queries for the topic 'molecular gastronomy' (BAD):\n"
            "- 'Definition molecular gastronomy'\n"
            "- 'Molecular gastronomy techniques and applications'\n"
            "Examples of curious, creative queries for the topic 'molecular gastronomy' (GOOD):\n"
            "- 'Use of liquid nitrogen instead of traditional freezing for food texture'\n"
            "- 'Failed molecular gastronomy experiments that led to new dishes'\n\n"
            f"Now generate one creative search query for: <TOPIC>{case.inputs['topic']}</TOPIC>"
        )
        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt,
                model_settings=ModelSettings(
                    # temperature=0.0,
                    temperature=1.0,
                    timeout=300,
                ),
            )

        logger.debug(f"Generated query: {result.output}")
