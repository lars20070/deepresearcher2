#!/usr/bin/env python3
from __future__ import annotations as _annotations

from deepresearcher2.evals.evals import EvalGame, EvalPlayer, EvalTournament


def test_evalplayer() -> None:
    """
    Test the EvalPlayer class.
    """
    player = EvalPlayer(
        idx=42,
        item="toasted rice & miso caramel ice cream",
    )
    assert player.idx == 42
    assert player.item == "toasted rice & miso caramel ice cream"


def test_evalgame() -> None:
    """
    Test the EvalGame class.
    """
    game = EvalGame(criterion="Which of the two ice cream flavours A or B is more creative?")
    assert game.criterion == "Which of the two ice cream flavours A or B is more creative?"


def test_evaltournament() -> None:
    """
    Test the EvalTournament class.
    """
    players = [
        EvalPlayer(idx=0, item="vanilla"),
        EvalPlayer(idx=1, item="chocolate"),
        EvalPlayer(idx=2, item="toasted rice & miso caramel ice cream"),
    ]
    game = EvalGame(criterion="Which of the two ice cream flavours A or B is more creative?")
    tournament = EvalTournament(players=players, game=game)

    assert len(tournament.players) == 3
    assert tournament.game.criterion == "Which of the two ice cream flavours A or B is more creative?"
