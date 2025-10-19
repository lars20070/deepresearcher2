#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
import json
import math
import random
import textwrap
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import choix
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

from deepresearcher2.agents import evaluation_agent
from deepresearcher2.config import config
from deepresearcher2.evals.import_bigbench import Response
from deepresearcher2.logger import logger
from deepresearcher2.models import GameResult


class ExactMatch(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext[Any, Any]) -> float:
        return float(ctx.output == ctx.expected_output)


class ExactMatchAny(Evaluator[str, list[str], Any]):
    async def evaluate(self, ctx: EvaluatorContext[str, list[str]]) -> float:
        """
        We simply check that the output is in the list of expected outputs.
        Note that the `out` list is always of length 1.
        """
        out = ctx.output or []
        exp_out = ctx.expected_output or []
        return float(any(o in exp_out for o in out))


async def eval_codenames(model: str = "qwen2.5:72b", max_cases: int | None = None) -> float:
    """
    Runs evaluation for codenames benchmark.

    Args:
        model (str): The model to use for evaluation.
        max_cases (int | None): The maximum number of cases to evaluate. Defaults to None.

    Returns:
        float: The evaluation score.
    """
    logger.info("Runs evaluation for codenames benchmark.")

    ollama_model = OpenAIChatModel(
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

    async def transform_text(text: str) -> str:
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

    averages = report.averages()
    if averages is not None:
        score = averages.scores.get("ExactMatch", 0)
    else:
        score = 0.0
    logger.info(f"Evaluation score: {score}")

    return score


async def eval_darkhurmordetection(model: str = "qwen2.5:72b", max_cases: int | None = None) -> float:
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

    ollama_model = OpenAIChatModel(
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

    averages = report.averages()
    if averages is not None:
        score = averages.scores.get("ExactMatch", 0)
    else:
        score = 0.0
    logger.info(f"Evaluation score: {score}")

    return score


async def eval_rephrase(model: str = "qwen2.5:72b", max_cases: int | None = None) -> float:
    """
    Runs evaluation for rephrase benchmark

    Args:
        model (str): The model to use for evaluation.
        max_cases (int | None): The maximum number of cases to evaluate. Defaults to None.

    Returns:
        float: The evaluation score.
    """
    logger.info("Runs evaluation for rephrase benchmark.")

    ollama_model = OpenAIChatModel(
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

    async def transform_text(text: str) -> list[str]:
        r = await rephraser.run(text)
        return [r.output]

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
            IsInstance(type_name="list[str]"),  # Pointless here since the evaluation crashes anyhow if the output type is incorrect.
            ExactMatchAny(),
        ],
    )

    # Run the evaluation
    report = await dataset.evaluate(transform_text)
    # report.print(include_input=True, include_output=True, include_durations=True)
    logger.debug(f"Complete evaluation report:\n{report}")

    averages = report.averages()
    if averages is not None:
        score = averages.scores.get("ExactMatch", 0)
    else:
        score = 0.0
    logger.info(f"Evaluation score: {score}")

    return score


def make_knowledge_gap_agent(model_name: str = "qwen2.5:72b") -> Agent:
    """
    Generates an agent for identifying knowledge gaps in research summaries.

    Args:
        model_name (str): The name of the model to use.
    """
    ollama_model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(base_url=f"{config.ollama_host}/v1"),
    )

    prompt = """
    You are diligent research assistant working on a specific research topic. You are given a summary of a web search.
    Based on the provided summary, generate a single subtopic which is relevant to the main topic but not sufficiently covered
    in the summary. The new subtopic will be the basis for the next web search query. You can think of the new subtopic as
    a 'knowledge gap' in the current summary.

    <OUTPUT_FORMAT>
    - Respond with a single phrase containing the new subtopic.
    - The subtopic should be five or less words long.
    - Avoid XML tags and markdown formatting.
    </OUTPUT_FORMAT>
    """

    return Agent(
        model=ollama_model,
        output_type=str,
        system_prompt=prompt,
    )


async def generate_knowledge_gap(topic: str, summary: str, generator: Agent, settings: ModelSettings) -> str:
    """
    Generates a knowledge gap.

    Args:
        topic (str): The main research topic.
        summary (str): Summary of web search results.
        generator (Agent): The agent to use for generation.
        settings (ModelSettings): The settings to use for the model.

    Returns:
        str: The generated knowledge gap.
    """

    prompt = (
        f"Please come up with a new subtopic for the topic <TOPIC>{topic}</TOPIC>.\n"
        f"Base your response on the following summary:\n<SUMMARY>{summary}</SUMMARY>\n"
        "Respond with a single phrase containing the new subtopic."
    )

    async with generator:
        result = await generator.run(user_prompt=prompt, model_settings=settings)
        logger.debug(f"New subtopic for {topic}:\n{result.output}")

    return result.output


class EvalPlayer(BaseModel):
    idx: int = Field(..., description="unique identifier for the player")
    item: str = Field(..., description="item to be scored")
    score: float | None = Field(default=None, description="Bradley-Terry strength score for the item")


class EvalGame(BaseModel):
    criterion: str = Field(..., description="evaluation criterion on which players should be judged")

    async def run(self, players: tuple[EvalPlayer, EvalPlayer], agent: Agent, model_settings: ModelSettings) -> tuple[int, int]:
        prompt = textwrap.dedent(f"""
            <QUESTION> {self.criterion} </QUESTION>
            <A> {players[0].item} </A>
            <B> {players[1].item} </B>
        """)

        async with agent:
            result = await agent.run(
                user_prompt=prompt,
                model_settings=model_settings,
            )

        if result.output == GameResult.A:
            return (players[0].idx, players[1].idx)
        else:
            return (players[1].idx, players[0].idx)


TournamentStrategy = Callable[
    [list[EvalPlayer], EvalGame, Agent, ModelSettings],
    Awaitable[list[EvalPlayer]],
]


async def round_robin_strategy(
    players: list[EvalPlayer],
    game: EvalGame,
    agent: Agent,
    model_settings: ModelSettings,
    number_of_rounds: int = 2,
) -> list[EvalPlayer]:
    """
    Round-robin tournament strategy.

    Each player plays against a randomly selected opponent for a given number of rounds.
    The scores are calculated from the game outcomes using the Bradley-Terry algorithm.
    The strategy ensures that each player plays at least number_of_rounds games.
    The strategy is simple but not efficient.

    Args:
        players: List of players in the tournament.
        game: Game defining the pairwise comparisons.
        agent: Agent for the game.
        model_settings: Model settings for the game.
        number_of_rounds: Number of rounds.

    Returns:
        List of players with Bradley-Terry scores.
    """
    scoreboard: list[tuple[int, int]] = []

    logger.info(f"Round-robin strategy: {len(players)} players, {number_of_rounds} rounds")

    for n in range(number_of_rounds):
        logger.debug(f"Starting round {n + 1} / {number_of_rounds}")

        for player in players:
            # Pick a random opponent (excluding self)
            idx = random.randrange(len(players))
            while idx == player.idx:
                idx = random.randrange(len(players))
            player_2 = players[idx]
            logger.debug(f"Game: Player {player.idx} vs Player {player_2.idx}")

            # Play the game
            result = await game.run(
                players=(player, player_2),
                agent=agent,
                model_settings=model_settings,
            )
            scoreboard.append(result)
            logger.debug(f"Result: {result}")

    # Calculate Bradley-Terry scores and update players
    scores = choix.ilsr_pairwise(len(players), scoreboard, alpha=0.01)
    for i, player in enumerate(players):
        player.score = float(scores[i])

    return players


async def random_sampling_strategy(
    players: list[EvalPlayer],
    game: EvalGame,
    agent: Agent,
    model_settings: ModelSettings,
    fraction_of_games: float | None = None,
) -> list[EvalPlayer]:
    """
    Random sampling tournament strategy.

    In a tournament with n players, there are n*(n-1) possible pairwise games. We consider
    (i, j) and (j, i) as different games in order to ensure that the evaluation agent does
    not introduce any ordering bias. The strategy plays all possible games in random order.
    The strategy is simple and not efficient. But when all games are played, it returns the
    best possible scores.

    Args:
        players: List of players in the tournament.
        game: Game defining the pairwise comparisons.
        agent: Agent for the game.
        model_settings: Model settings for the game.
        fraction_of_games: Fraction of all possible games to be played. Between 0 and 1. If None, all games are played.

    Returns:
        List of players with Bradley-Terry scores.
    """
    scoreboard: list[tuple[int, int]] = []

    # Generate all possible games
    n = len(players)
    matches = [(i, j) for i in range(n) for j in range(n) if i != j]
    random.shuffle(matches)
    if fraction_of_games is not None and 0 < fraction_of_games <= 1:
        number_of_games = int(len(matches) * fraction_of_games)
        matches = matches[:number_of_games]
    logger.info(f"Random sampling strategy: {n} players playing {len(matches)} games")

    # Play all games
    for i, match in enumerate(matches):
        player_1, player_2 = players[match[0]], players[match[1]]
        logger.debug(f"Game {i + 1} / {len(matches)}: Player {player_1.idx} vs Player {player_2.idx}")

        result = await game.run(
            players=(player_1, player_2),
            agent=agent,
            model_settings=model_settings,
        )
        scoreboard.append(result)
        logger.debug(f"Result: {result}")

    # Calculate Bradley-Terry scores and update players
    scores = choix.ilsr_pairwise(len(players), scoreboard, alpha=0.01)
    for i, player in enumerate(players):
        player.score = float(scores[i])

    return players


async def adaptive_uncertainty_strategy(
    players: list[EvalPlayer],
    game: EvalGame,
    agent: Agent,
    model_settings: ModelSettings,
    max_standard_deviation: float = 2.0,
    alpha: float = 0.1,
) -> list[EvalPlayer]:
    """
    Adaptive uncertainty tournament strategy.

    The strategy consists of two phases:
    (1) Bootstrap phase: The Bradley-Terry model requires the comparison graph to be strongly connected i.e.
        there must be a path between any two players. We therefore start by playing n/2*log(n) random games where
        n is the number of players. With fewer games, the scores are most likely unreliable.
    (2) Optimization phase: In this phase, we iteratively calculate the Bradley-Terry scores and their
        covariance matrix, and play the game for which the player scores are the most uncertain.

        Let s_i and s_j the Bradley-Terry scores of players i and j respectively. The uncertainty in their
        relative strength is then given by

        Var(s_i - s_j) = Var(s_i) + Var(s_j) - 2*Cov(s_i, s_j)

        We stop when the standard deviation sqrt(Var(s_i - s_j)) of the most uncertain pair drops below
        the threshold max_standard_deviation, or when all possible pairs have been played.

    Comment on max_standard_deviation parameter:
        Typically, a standard deviation below 1.0 is a good stopping condition. However, the uncertainty
        depends greatly on the evaluation problem. For a problem such as "Which of the following ice cream
        flavours is the most creative one? Vanilla or Chocolate or Strawberry?", the uncertainty will remain
        high even after many games.

    Comment on alpha parameter:
        The alpha parameter is the prior strength for the Bradley-Terry model. Higher alpha (e.g. 0.8) is a
        strong prior towards equal player strengths. The games have a smaller influence on the scores, and
        the scores remain close to the mean of 0. Lower alpha (e.g. 0.1) on the other hand lets the games
        influence the scores more strongly. However, for a sparse comparison graph, the scores can become
        less stable. Typical values are between 0.1 and 0.3.

    Args:
        players: List of players in the tournament.
        game: Game defining the pairwise comparisons.
        agent: Agent for the game.
        model_settings: Model settings for the game.
        max_standard_deviation: Maximum standard deviation for the most uncertain pair. See also above.
        alpha: Prior strength for the Bradley-Terry model. Between 0 and 1. See also above.

    Returns:
        List of players with Bradley-Terry scores.
    """
    scoreboard: list[tuple[int, int]] = []
    n = len(players)

    logger.info(f"Adaptive uncertainty strategy: {n} players")

    # (1) Bootstrap phase
    number_of_bootstrap_games = max(2 * n, int(n / 2 * np.log(n)))
    matches = [(i, j) for i in range(n) for j in range(n) if i != j]
    random.shuffle(matches)
    matches = matches[:number_of_bootstrap_games]
    for idx, match in enumerate(matches):
        player_1, player_2 = players[match[0]], players[match[1]]
        logger.debug(f"Bootstrap game {idx + 1} / {len(matches)}: Player {player_1.idx} vs Player {player_2.idx}")

        result = await game.run(
            players=(player_1, player_2),
            agent=agent,
            model_settings=model_settings,
        )
        scoreboard.append(result)
        logger.debug(f"Result: {result}")

    # (2) Optimization phase
    max_number_of_games = int(n * (n - 1) / 2.0)
    previous_scores: np.ndarray | None = None
    for idx in range(max_number_of_games):
        logger.debug(f"Optimization game {idx + 1} / {max_number_of_games}")

        # Calculate the Bradley-Terry scores and covariance matrix
        scores, cov_matrix = choix.ep_pairwise(n_items=n, data=scoreboard, alpha=alpha, model="logit")

        # For monitoring only, check the absolute score changes.
        # Note that this change is not decreasing monotonically.
        if previous_scores is not None:
            absolute_change = np.abs(scores - previous_scores)
            max_change = np.max(absolute_change)
            logger.debug(f"Maximum absolute change in the scores since last iteration: {max_change}")
        previous_scores = scores.copy()

        # Find most uncertain pair which has not yet been played.
        max_uncertainty = -1.0
        next_pair: tuple[int, int] | None = None
        for i in range(n):
            for j in range(i + 1, n):
                # Check if the pair has already been played.
                # Here we assume that games are symmetric which is not quite correct but good enough.
                if (players[i].idx, players[j].idx) in scoreboard or (players[j].idx, players[i].idx) in set(scoreboard):
                    continue

                # Uncertainty of the pair
                uncertainty = cov_matrix[i, i] + cov_matrix[j, j] - 2 * cov_matrix[i, j]
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    next_pair = (i, j)

        # Terminate optimization phase?
        if next_pair is None:
            logger.info("All pairs have been played. Ending optimization phase.")
            break
        if math.sqrt(max_uncertainty) < max_standard_deviation:
            logger.info(
                f"The standard deviation of the most uncertain pair {math.sqrt(max_uncertainty)} is below the threshold {max_standard_deviation}. "
                f"Ending optimization phase."
            )
            break

        # Play the most uncertain pair
        logger.debug(
            f"Most uncertain pair: Player {players[next_pair[0]].idx} vs Player {players[next_pair[1]].idx} (uncertainty: {max_uncertainty})"
        )
        player_1, player_2 = players[next_pair[0]], players[next_pair[1]]
        result = await game.run(
            players=(player_1, player_2),
            agent=agent,
            model_settings=model_settings,
        )
        scoreboard.append(result)
        logger.debug(f"Result: {result}")

    # Final calculation of Bradley-Terry scores and update players
    scores, _ = choix.ep_pairwise(n_items=n, data=scoreboard, alpha=alpha, model="logit")
    for i, player in enumerate(players):
        player.score = float(scores[i])

    return players


class EvalTournament(BaseModel):
    game: EvalGame = Field(..., description="game to be played in the tournament")
    players: list[EvalPlayer] = Field(..., description="players participating in the tournament")

    def get_player_by_idx(self, idx: int) -> EvalPlayer:
        """
        Return player with unique identifier idx

        Args:
            idx: Unique identifier of the player.

        Returns:
            Player with the specified unique identifier.
        """
        for player in self.players:
            if player.idx == idx:
                return player
        raise ValueError(f"Player with unique identifier {idx} not found.")

    async def run(
        self,
        agent: Agent,
        model_settings: ModelSettings,
        strategy: TournamentStrategy | None = None,
        **strategy_kwargs: Any,  # noqa: ANN401
    ) -> list[EvalPlayer]:
        """
        Runs the evaluation tournament using the specified strategy.

        The strategy function handles game sampling, game execution and scoring
        allowing complete flexibility in the tournament algorithms.

        Args:
            agent: Agent for the evaluation game.
            model_settings: Model settings for the evaluation game.
            strategy: Function with the tournament algorithm.
            **strategy_kwargs: Additional arguments passed to the strategy function.

        Returns:
            List of players with scores.
        """
        # Use default strategy if none provided
        if strategy is None:
            strategy = adaptive_uncertainty_strategy

        logger.info(f"Starting tournament with {len(self.players)} players using {strategy.__name__}")

        # Run the tournament strategy (returns players with scores)
        self.players = await strategy(
            self.players,
            self.game,
            agent,
            model_settings,
            **strategy_kwargs,
        )

        logger.info(f"Tournament complete using {strategy.__name__}")
        return self.players


async def eval_knowledge_gap(models: list[str] | None = None, max_cases: int | None = None) -> None:
    """
    Runs evaluation for knowledge gap benchmark

    We use multiple judges with different underlying models to score the generated knowledge gaps.

    Args:
        models (list[str] | None): The models to use for scoring the knowledge gaps.
        max_cases (int | None): The maximum number of cases to evaluate. Defaults to None.

    Returns:
        float: The evaluation score.
    """
    logger.info("Runs evaluation for knowledge gap benchmark.")

    # Load the benchmark cases
    path = Path("benchmarks/knowledge_gap/task.json")
    dataset = Dataset[dict[str, str], str, Any].from_file(path)
    cases = dataset.cases
    if max_cases is not None:
        cases = cases[:max_cases]
    # Add the benchmark evaluators
    dataset = Dataset[dict[str, str], str, Any](
        cases=cases,
        evaluators=[IsInstance(type_name="str")],
    )
    logger.debug(f"Loaded dataset with {len(dataset.cases)} cases.")

    # # Generate knowledge gaps
    # generator = make_knowledge_gap_agent("qwen2.5:72b")
    # generator_settings = ModelSettings(temperature=1.0, timeout=600)
    # gaps: list[str] = []
    # for case in dataset.cases:
    #     topic = case.inputs.get("topic", "")
    #     summary = case.inputs.get("summary", "")
    #     logger.debug(f"Generating knowledge gap for case: {case.name} with topic: {topic}")

    #     gap = await generate_knowledge_gap(
    #         topic=topic,
    #         summary=summary,
    #         generator=generator,
    #         settings=generator_settings,
    #     )
    #     gaps.append(gap)

    # # Serialize the knowledge gaps to a file
    # output_path = Path("benchmarks/knowledge_gap/knowledge_gaps.json")
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # with output_path.open("w", encoding="utf-8") as f:
    #     json.dump(gaps, f, indent=2, ensure_ascii=False)

    # Deserialize knowledge gaps from file
    input_path = Path("benchmarks/knowledge_gap/knowledge_gaps.json")
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        raise FileNotFoundError(f"Cannot find the file: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        gaps: list[str] = json.load(f)
    if max_cases is not None:
        gaps = gaps[:max_cases]
    logger.info(f"Loaded {len(gaps)} knowledge gaps from {input_path}")

    # Create players and tournament
    players = [EvalPlayer(idx=i, item=gap) for i, gap in enumerate(gaps)]
    tournament = EvalTournament(
        players=players,
        game=EvalGame(criterion="Which of the two search queries (A or B) shows more genuine curiosity and creativity, and is less formulaic?"),
    )

    # Run the tournament
    players_scored = await tournament.run(
        agent=evaluation_agent,
        model_settings=ModelSettings(
            temperature=1.0,
            timeout=config.model_timeout,
        ),
        strategy=adaptive_uncertainty_strategy,
    )

    # Print the scores
    for player in players_scored:
        logger.debug(f"Score for Player {player.idx}: {player.score:0.4f}")

    # Print players sorted by score
    scores = [player.score for player in players_scored if player.score is not None]
    idx_sorted = np.argsort(scores)
    for i in idx_sorted:
        player = tournament.get_player_by_idx(i)
        logger.debug(f"Player {player.idx:2d}   score: {player.score:7.4f}   item: {player.item}")


def main() -> None:
    """
    Main function running evaluations.
    """
    logger.info("Run evaluation.")
    # model = "llama3.3"
    # model = "qwen2.5:72b"
    models = ["llama3.3", "qwen2.5:72b"]
    max_cases = None
    # max_cases = None
    # asyncio.run(eval_codenames(model=model, max_cases=max_cases))
    # asyncio.run(eval_darkhurmordetection(model=model, max_cases=max_cases))
    # asyncio.run(eval_rephrase(model=model, max_cases=max_cases))
    asyncio.run(eval_knowledge_gap(models=models, max_cases=max_cases))


if __name__ == "__main__":
    main()
