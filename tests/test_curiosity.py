#!/usr/bin/env python3
from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

from deepresearcher2.plugin import PLAYERS_KEY

if TYPE_CHECKING:
    from pathlib import Path

import random  # noqa: F401

import numpy as np
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: F401
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: F401
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset

from deepresearcher2.agents import EVALUATION_AGENT, model  # noqa: F401
from deepresearcher2.evals.evals import (
    EvalGame,
    EvalPlayer,
    EvalTournament,
    adaptive_uncertainty_strategy,
)
from deepresearcher2.logger import logger
from deepresearcher2.plugin import AssayContext

MODEL_SETTINGS = ModelSettings(
    temperature=0.0,
    timeout=300,
)

MODEL_SETTINGS_CURIOUS = ModelSettings(
    temperature=1.0,  # TODO: At higher temperature, qwen2.5:72b starts talking in Chinese despite the system prompt telling it not to.
    timeout=300,
)


# @pytest.mark.vcr()
@pytest.mark.skip(reason="Run only locally with DeepInfra cloud inference. PROVIDER='deepinfra' MODEL='Qwen/Qwen2.5-72B-Instruct'")
@pytest.mark.assay
@pytest.mark.asyncio
@pytest.mark.usefixtures("timer_for_tests")
async def test_search_queries_1(request: pytest.FixtureRequest, assay_path: Path, assay_dataset: Dataset) -> None:
    """
    Use case for EvalTournament, EvalGame and EvalPlayer classes.

    The code demonstrates how the evaluation framework can be used in practice. It is not intended as test for individual components.
    In this use case, we are provided with a list of topics. The objective is to generate creative web search queries for these topics.
    We have a baseline implementation in the `main` branch and a novel implementation in some `feature` branch. In this simple example,
    the implementations differ merely in the prompt (`prompt_baseline` vs. `prompt_novel`) and temperature. We want to check whether the
    novel implementation does indeed generate more creative queries.

    The use case proceeds in three steps:
    (1) We generate an evaluation `Dataset` containing the topics.
    (2) We run the baseline implementation and store the generated queries in the dataset. This code could be run as part of the
        CI/CD pipeline whenever the `main` branch changes.
    (3) We run the novel implementation, score both baseline and novel queries in one go using a Bradley-Terry tournament,
        and check whether the scores have improved.
    """

    logger.debug(f"assay path: {assay_path}")
    logger.debug(f"assay dataset: {assay_dataset}")

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

    # (1) Generate Cases and serialise them

    dataset = assay_dataset

    # (2) Generate base line model outputs

    # dataset = Dataset[dict[str, str], type[None], Any].from_file(assay_path)
    cases_new: list[Case[dict[str, str], type[None], Any]] = []
    logger.info("")
    for case in dataset.cases:
        logger.info(f"Case {case.name} with topic: {case.inputs['topic']}")

        prompt_baseline = f"Please generate a query for the research topic: <TOPIC>{case.inputs['topic']}</TOPIC>"
        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt_baseline,
                model_settings=MODEL_SETTINGS,
            )

        logger.debug(f"Generated query: {result.output}")
        case_new = Case(
            name=case.name,
            inputs={"topic": case.inputs["topic"], "query": result.output},
        )
        cases_new.append(case_new)
    dataset_new: Dataset[dict[str, str], type[None], Any] = Dataset[dict[str, str], type[None], Any](cases=cases_new)

    assay_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_new.to_file(assay_path, schema_path=None)

    # (3) Generate novel model outputs and score them

    dataset = Dataset[dict[str, str], type[None], Any].from_file(assay_path)
    players: list[EvalPlayer] = []
    logger.info("")
    for idx, case in enumerate(dataset.cases):
        logger.info(f"Case {case.name} with topic: {case.inputs['topic']}")

        prompt_novel = (
            f"Please generate a very creative search query for the research topic: <TOPIC>{case.inputs['topic']}</TOPIC>\n"
            "The query should show genuine originality and interest in the topic. AVOID any generic or formulaic phrases."
        )
        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt_novel,
                model_settings=MODEL_SETTINGS_CURIOUS,
            )

        logger.debug(f"Generated query: {result.output}")

        player_baseline = EvalPlayer(idx=idx, item=case.inputs["query"])
        player_novel = EvalPlayer(idx=idx + len(dataset.cases), item=result.output)
        players.append(player_baseline)
        players.append(player_novel)

    # Pass new responses to PyTest hook
    request.node.stash[PLAYERS_KEY] = players

    # Run the Bradley-Terry tournament to score both baseline and novel queries
    game = EvalGame(criterion="Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?")
    tournament = EvalTournament(players=players, game=game)
    players_scored = await tournament.run(
        agent=EVALUATION_AGENT,
        model_settings=MODEL_SETTINGS,
        strategy=adaptive_uncertainty_strategy,
        max_standard_deviation=2.0,
    )

    # Players sorted by score
    players_sorted = sorted(players_scored, key=lambda p: p.score if p.score is not None else float("-inf"))
    for player in players_sorted:
        logger.debug(f"Player {player.idx:4d}   score: {player.score:7.4f}   query: {player.item}")

    # Average score for both baseline and novel queries
    scores_baseline = [tournament.get_player_by_idx(idx=i).score or 0.0 for i in range(len(dataset.cases))]
    scores_novel = [tournament.get_player_by_idx(idx=i + len(dataset.cases)).score or 0.0 for i in range(len(dataset.cases))]
    logger.debug(f"Average score for baseline queries (Players 0 to 9): {np.mean(scores_baseline):7.4f}")
    logger.debug(f"Average score for novel queries  (Players 10 to 19): {np.mean(scores_novel):7.4f}")
    # Not every novel query will have scored higher than the baseline case. But on average the novel queries should have improved scores.
    assert np.mean(scores_novel) > np.mean(scores_baseline)
    # The sum of all Bradley-Terry scores is zero.
    assert np.isclose(np.mean(scores_novel) + np.mean(scores_baseline), 0)


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
@pytest.mark.assay(generator=generate_evaluation_cases)
@pytest.mark.asyncio
@pytest.mark.usefixtures("timer_for_tests")
async def test_search_queries_2(assay: AssayContext) -> None:
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

        prompt_baseline = f"Please generate a query for the research topic: <TOPIC>{case.inputs['topic']}</TOPIC>"
        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt_baseline,
                model_settings=MODEL_SETTINGS,
            )

        logger.debug(f"Generated query: {result.output}")
        case_new = Case(
            name=case.name,
            inputs={"topic": case.inputs["topic"], "query": result.output},
        )
        cases_new.append(case_new)

    assert cases_new is not None

    # Update assay dataset in place (automatic serialisation by assay plugin)
    assay.dataset.cases.clear()
    assay.dataset.cases.extend(cases_new)
