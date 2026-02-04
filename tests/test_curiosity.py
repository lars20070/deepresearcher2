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

from deepresearcher2.config import config  # noqa: F401
from deepresearcher2.logger import logger
from deepresearcher2.plugin import AssayContext, BradleyTerryEvaluator, PairwiseEvaluator


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
        logger.debug(f"Case {idx + 1} / {len(topics)} with topic: {topic}")
        case = Case(
            name=f"case_{idx:03d}",
            inputs={"topic": topic},
            expected_output="",
        )
        cases.append(case)

    return Dataset[dict[str, str], str, Any](cases=cases)


# Model for both (1) the search query generation and (2) the evaluation of the generated queries.
model = OpenAIChatModel(
    model_name="qwen3:14b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),  # Local Ollama server
)
# model = OpenAIChatModel(
#     model_name="Qwen/Qwen3-14B",
#     provider=OpenAIProvider(base_url=config.deepinfra_base_url, api_key=config.deepinfra_api_key),  # DeepInfra cloud server
# )


BASIC_PROMPT = "Please generate a useful search query for the following research topic: <TOPIC>{topic}</TOPIC>"

CREATIVE_PROMPT = """\
Please generate a very creative search query for the research topic: <TOPIC>{topic}</TOPIC>
The query should show genuine originality and interest in the topic. AVOID any generic or formulaic phrases.

Examples of formulaic queries for the topic 'molecular gastronomy' (BAD):
- 'Definition molecular gastronomy'
- 'Molecular gastronomy techniques and applications'

Examples of curious, creative queries for the topic 'molecular gastronomy' (GOOD):
- 'Use of liquid nitrogen instead of traditional freezing for food texture'
- 'Failed molecular gastronomy experiments that led to new dishes'

Now generate one creative search query for: <TOPIC>{topic}</TOPIC>"""


async def _run_query_generation(context: AssayContext) -> None:
    """Run query generation for all cases in the dataset."""
    query_agent = Agent(
        model=model,
        output_type=str,
        system_prompt="Please generate a concise web search query for the given research topic. You must respond only in English.",
        retries=5,
        instrument=True,
    )

    for case in context.dataset.cases:
        logger.info(f"Case {case.name} with topic: {case.inputs['topic']}")

        # prompt = BASIC_PROMPT.format(topic=case.inputs["topic"])  # for assay-mode 'new_baseline'
        prompt = CREATIVE_PROMPT.format(topic=case.inputs["topic"])  # for assay-mode 'evaluate'

        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt,
                model_settings=ModelSettings(temperature=0.0, timeout=300),
            )

        logger.debug(f"Generated search query: {result.output}")


@pytest.mark.vcr()
@pytest.mark.assay(
    generator=generate_evaluation_cases,
    evaluator=PairwiseEvaluator(
        model=model,
        criterion="Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
    ),
)
@pytest.mark.asyncio
async def test_integration_pairwiseevaluator(context: AssayContext) -> None:
    """
    Integration test for the 'assay' pytest plugin with PairwiseEvaluator evaluator.

    An agent generates search queries for various research topics.
    PairwiseEvaluator then evaluates the creativity of the generated queries.

    Args:
        context: The assay context containing the evaluation dataset `context.dataset`and other information.
    """
    logger.info("Integration test for assay pytest plugin with PairwiseEvaluator.")
    await _run_query_generation(context)


@pytest.mark.vcr()
@pytest.mark.assay(
    generator=generate_evaluation_cases,
    evaluator=BradleyTerryEvaluator(
        model=model,
        criterion="Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
        max_standard_deviation=2.1,
    ),
)
@pytest.mark.asyncio
async def test_integration_bradleyterryevaluator(context: AssayContext) -> None:
    """
    Integration test for the 'assay' pytest plugin with BradleyTerryEvaluator evaluator.

    An agent generates search queries for various research topics.
    BradleyTerryEvaluator then evaluates the creativity of the generated queries.

    Args:
        context: The assay context containing the evaluation dataset `context.dataset`and other information.
    """
    logger.info("Integration test for assay pytest plugin with BradleyTerryEvaluator.")
    await _run_query_generation(context)
