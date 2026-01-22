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
from deepresearcher2.logger import logger
from deepresearcher2.plugin import AssayContext, bradley_terry_evaluation


def generate_evaluation_cases() -> Dataset[dict[str, str], type[None], Any]:
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
        "Anne Brorhilke",
        "bioconcrete self-healing",
        "bacteriophage therapy revival",
        "Habsburg jaw genetics",
    ]

    cases: list[Case[dict[str, str], type[None], Any]] = []
    for idx, topic in enumerate(topics):
        logger.info(f"Case {idx + 1} / {len(topics)} with topic: {topic}")
        case = Case(
            name=f"case_{idx:03d}",
            inputs={"topic": topic},
        )
        cases.append(case)

    return Dataset[dict[str, str], type[None], Any](cases=cases)


@pytest.mark.skip(reason="Run only locally with DeepInfra cloud inference. PROVIDER='deepinfra' MODEL='Qwen/Qwen2.5-72B-Instruct'")
@pytest.mark.assay(generator=generate_evaluation_cases, evaluator=bradley_terry_evaluation)
@pytest.mark.asyncio
@pytest.mark.usefixtures("timer_for_tests")
async def test_search_queries(assay: AssayContext) -> None:
    """
    Run the agent workflow once.
    """

    logger.debug(f"assay path: {assay.path}")
    logger.debug(f"assay dataset: {assay.dataset}")

    # Agent for generating search queries using a local Ollama server
    model_for_queries = OpenAIChatModel(
        model_name="qwen2.5:72b",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
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

    cases_new: list[Case[dict[str, str], type[None], Any]] = []
    logger.info("")
    for case in assay.dataset.cases:
        logger.info(f"Case {case.name} with topic: {case.inputs['topic']}")

        # prompt = f"Please generate a query for the research topic: <TOPIC>{case.inputs['topic']}</TOPIC>"
        prompt = (
            f"Please generate a very creative search query for the research topic: <TOPIC>{case.inputs['topic']}</TOPIC>\n"
            "The query should show genuine originality and interest in the topic. AVOID any generic or formulaic phrases."
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
        case_new = Case(
            name=case.name,
            inputs={"topic": case.inputs["topic"], "query": result.output},
        )
        cases_new.append(case_new)

    assert cases_new is not None

    # Update assay dataset in place
    # Required for automatic serialisation by pytest-assay plugin
    assay.dataset.cases.clear()
    assay.dataset.cases.extend(cases_new)
