#!/usr/bin/env python3
from __future__ import annotations as _annotations

import pytest
from pydantic_ai.settings import ModelSettings

from deepresearcher2.agents import evaluation_agent
from deepresearcher2.evals.evals import EvalGame, EvalPlayer, EvalTournament
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


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_evalgame() -> None:
    """
    Test the EvalGame class.
    """
    logger.info("Testing EvalGame() class")

    game = EvalGame(criterion="Which of the two ice cream flavours A or B is more creative?")
    assert game.criterion == "Which of the two ice cream flavours A or B is more creative?"

    player_a = EvalPlayer(idx=0, item="vanilla")
    player_b = EvalPlayer(idx=1, item="toasted rice & miso caramel ice cream")

    result = await game.run(
        players=(player_a, player_b),
        agent=evaluation_agent,
        model_settings=ModelSettings(
            temperature=1.0,
            timeout=300,
        ),
    )
    logger.debug(f"Game result: {result}")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(r, int) for r in result)


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_evaltournament() -> None:
    """
    Test the EvalTournament class.
    """
    logger.info("Testing EvalTournament() class")

    players = [
        EvalPlayer(idx=0, item="vanilla"),
        EvalPlayer(idx=1, item="toasted rice & miso caramel ice cream"),
        EvalPlayer(idx=2, item="chocolate"),
    ]
    game = EvalGame(criterion="Which of the two ice cream flavours A or B is more creative?")
    tournament = EvalTournament(players=players, game=game)

    assert len(tournament.players) == 3
    assert tournament.game.criterion == "Which of the two ice cream flavours A or B is more creative?"

    players_with_scores = await tournament.run(
        agent=evaluation_agent,
        model_settings=ModelSettings(
            temperature=1.0,
            timeout=300,
        ),
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        # assert isinstance(player, EvalPlayer)
        # assert hasattr(player, "score")
        # assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")
