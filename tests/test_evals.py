#!/usr/bin/env python3
from __future__ import annotations as _annotations

import pytest
from pydantic_ai.settings import ModelSettings

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
        model_settings=ModelSettings(
            temperature=0.0,
            timeout=300,
        ),
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
        model_settings=ModelSettings(
            temperature=0.0,
            timeout=300,
        ),
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
        model_settings=ModelSettings(
            temperature=0.0,
            timeout=300,
        ),
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
        model_settings=ModelSettings(
            temperature=0.0,
            timeout=300,
        ),
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
        model_settings=ModelSettings(
            temperature=0.0,
            timeout=300,
        ),
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
        model_settings=ModelSettings(
            temperature=0.0,
            timeout=300,
        ),
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
