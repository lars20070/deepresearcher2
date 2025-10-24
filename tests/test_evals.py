#!/usr/bin/env python3
from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset

from deepresearcher2.agents import evaluation_agent
from deepresearcher2.evals.evals import (
    EvalGame,
    EvalPlayer,
    EvalTournament,
    adaptive_uncertainty_strategy,
    random_sampling_strategy,
    round_robin_strategy,
)
from deepresearcher2.logger import logger

EVAL_MODEL_SETTINGS = ModelSettings(
    temperature=0.0,  # Model needs to be deterministic for VCR recording to work.
    timeout=300,
)


def test_evalplayer() -> None:
    """
    Test the EvalPlayer class.
    """
    logger.info("Testing EvalPlayer() class")

    player = EvalPlayer(
        idx=42,
        item="toasted rice & miso caramel ice cream",
    )
    assert player.idx == 42
    assert player.item == "toasted rice & miso caramel ice cream"


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_evalgame(ice_cream_players: list[EvalPlayer]) -> None:
    """
    Test the EvalGame class.
    """
    logger.info("Testing EvalGame() class")

    game = EvalGame(criterion="Which of the two ice cream flavours A or B is more creative?")
    assert game.criterion == "Which of the two ice cream flavours A or B is more creative?"

    result = await game.run(
        players=(ice_cream_players[0], ice_cream_players[4]),
        agent=evaluation_agent,
        model_settings=EVAL_MODEL_SETTINGS,
    )
    logger.debug(f"Game result: {result}")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(r, int) for r in result)
    assert result[0] == 4  # Toasted rice & miso caramel ice cream flavour is more creative.


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_evaltournament(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
    """
    Test the EvalTournament class.
    """
    logger.info("Testing EvalTournament() class")

    tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)

    assert len(tournament.players) == len(ice_cream_players)
    assert tournament.game.criterion == ice_cream_game.criterion

    # Test player retrieval
    player = tournament.get_player_by_idx(1)
    assert player is not None
    assert player.item == ice_cream_players[1].item

    # Test the default strategy
    players_with_scores = await tournament.run(
        agent=evaluation_agent,
        model_settings=EVAL_MODEL_SETTINGS,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")

    # Test the random sampling strategy
    players_with_scores = await tournament.run(
        agent=evaluation_agent,
        model_settings=EVAL_MODEL_SETTINGS,
        strategy=random_sampling_strategy,
        fraction_of_games=0.3,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_random_sampling_strategy(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
    """
    Test the random sampling tournament strategy.
    """
    logger.info("Testing random_sampling_strategy()")

    players_with_scores = await random_sampling_strategy(
        players=ice_cream_players,
        game=ice_cream_game,
        agent=evaluation_agent,
        model_settings=EVAL_MODEL_SETTINGS,
        fraction_of_games=0.3,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_round_robin_strategy(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
    """
    Test the round robin tournament strategy.
    """
    logger.info("Testing round_robin_strategy()")

    players_with_scores = await round_robin_strategy(
        players=ice_cream_players,
        game=ice_cream_game,
        agent=evaluation_agent,
        model_settings=EVAL_MODEL_SETTINGS,
        number_of_rounds=1,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_adaptive_uncertainty_strategy(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
    """
    Test the adaptive uncertainty tournament strategy.
    """
    logger.info("Testing adaptive_uncertainty_strategy()")

    players_with_scores = await adaptive_uncertainty_strategy(
        players=ice_cream_players,
        game=ice_cream_game,
        agent=evaluation_agent,
        model_settings=EVAL_MODEL_SETTINGS,
        max_standard_deviation=1.0,
        alpha=0.01,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")


@pytest.mark.asyncio
async def test_evaltournament_usecase(tmp_path: Path) -> None:
    """
    Use case for EvalTournament, EvalGame and EvalPlayer classes.

    The code demonstrates how the evaluation framework can be used in practice.
    It is not intended as test for individual components.
    """

    # Path to store the evaluation dataset
    path_out = tmp_path / "dataset.json"

    # Agent for generating search queries using a local Ollama server
    query_agent = Agent(
        model=OpenAIChatModel(
            model_name="qwen2.5:72b",
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        ),
        output_type=str,
        system_prompt="Please generate a concise web search query for the given research topic. Reply with ONLY the query string.",
        retries=5,
        instrument=True,
    )

    logger.info("Use case for EvalTournament, EvalGame and EvalPlayer classes.")

    # (1) Generate Cases and serialise them

    topics = [
        "pangolin trafficking networks",
        "molecular gastronomy",
        "dark kitchen economics",
        "kintsugi philosophy",
        "nano-medicine delivery systems",
        "Streisand effect dynamics",
        "social cooling phenomenon",
        "Anne Brorhilke",
        "bioconcrete self-healing",
        "bacteriophage therapy revival",
    ]

    cases: list[Case[dict[str, str], type[None], Any]] = []
    for idx, topic in enumerate(topics):
        logger.info(f"Case {idx + 1} / {len(topics)} with topic: {topic}")
        case = Case(
            name=f"case_{idx:03d}",
            inputs={"topic": topic},
        )
        cases.append(case)
    dataset: Dataset[dict[str, str], type[None], Any] = Dataset[dict[str, str], type[None], Any](cases=cases)
    dataset.to_file(path_out)

    # (2) Generate base line model outputs

    dataset = Dataset[dict[str, str], type[None], Any].from_file(path_out)
    cases_new: list[Case[dict[str, str], type[None], Any]] = []
    for case in dataset.cases:
        logger.info(f"Case {case.name} with topic: {case.inputs['topic']}")

        prompt = f"Please generate a query for the research topic: {case.inputs['topic']}"
        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt,
                model_settings=ModelSettings(
                    temperature=0.0,
                    timeout=300,
                ),
            )

        logger.debug(f"Generated query: {result.output}")
        case_new = Case(
            name=case.name,
            inputs={"topic": case.inputs["topic"], "query": result.output},
        )
        cases_new.append(case_new)
    dataset_new: Dataset[dict[str, str], type[None], Any] = Dataset[dict[str, str], type[None], Any](cases=cases_new)
    dataset_new.to_file(path_out)
