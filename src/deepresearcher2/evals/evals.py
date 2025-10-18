#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
import json
import random
import textwrap
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
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance, LLMJudge

from deepresearcher2.agents import evaluation_agent
from deepresearcher2.config import config
from deepresearcher2.evals.import_bigbench import Response
from deepresearcher2.logger import logger


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

        if result.output == "A":
            return (players[0].idx, players[1].idx)
        else:
            return (players[1].idx, players[0].idx)


class EvalTournament(BaseModel):
    game: EvalGame = Field(..., description="game to be played in the tournament")
    players: list[EvalPlayer] = Field(..., description="players participating in the tournament")
    scoreboard: list[tuple[int, int]] = Field(default_factory=list, description="results of the games played")

    async def run(self, agent: Agent, model_settings: ModelSettings) -> list[EvalPlayer]:
        # Start with round robin
        for player in self.players:
            # Pick a random second player for the game
            idx = random.randrange(len(self.players))
            while idx == player.idx:
                idx = random.randrange(len(self.players))
            player_2 = self.players[idx]
            logger.debug(f"Playing game between Player {player.idx} and Player {player_2.idx}")

            # Play the game
            result = await self.game.run(
                players=(player, player_2),
                agent=agent,
                model_settings=model_settings,
            )
            self.scoreboard.append(result)
            logger.debug(f"Game result: {result}")

        # Calculate Bradley-Terry scores and update players
        scores = choix.ilsr_pairwise(len(self.players), self.scoreboard, alpha=0.01)
        for i, player in enumerate(self.players):
            player.score = scores[i]

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

    # Create the knowledge gap generator
    generator = make_knowledge_gap_agent("qwen2.5:72b")
    generator_settings = ModelSettings(temperature=1.0, timeout=600)

    # Create the judges
    # Two independent judges with two different models.
    if models is None:
        models = ["qwen2.5:72b"]
    judges: list[LLMJudge] = []
    for model in models:
        ollama_model = OpenAIChatModel(
            model_name=model,
            provider=OpenAIProvider(
                base_url=f"{config.ollama_host}/v1",
            ),
        )
        judge = LLMJudge(
            rubric="The new subtopic should point to a research area which is clearly missing in the summary.",
            include_input=True,
            model=ollama_model,
        )
        judges.append(judge)

    async def transform_knowledge_gap(payload: dict[str, str]) -> str:
        knowledge_gap = await generate_knowledge_gap(
            topic=payload.get("topic", ""),
            summary=payload.get("summary", ""),
            generator=generator,
            settings=generator_settings,
        )
        return knowledge_gap

    # Load the benchmark cases
    path = Path("benchmarks/knowledge_gap/task.json")
    dataset = Dataset[dict[str, str], str, Any].from_file(path)
    cases = dataset.cases
    if max_cases is not None:
        cases = cases[:max_cases]
    # Add the benchmark evaluators
    dataset = Dataset[dict[str, str], str, Any](
        cases=cases,
        evaluators=[
            IsInstance(type_name="str"),
            *judges,
        ],
    )
    logger.debug(f"Loaded dataset with {len(dataset.cases)} cases.")

    # # Generate knowledge gaps
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
    logger.info(f"Loaded {len(gaps)} knowledge gaps from {input_path}")

    # # Run evaluation
    # prompt = """
    # <QUESTION> Which of the two books is a better present for a history enthusiast? </QUESTION>

    # <A> 'Sapiens: A Brief History of Humankind' by Yuval Noah Harari </A>

    # <B> 'Solaris' by Stanislaw Lem </B>
    # """
    # async with evaluation_agent:
    #     result = await evaluation_agent.run(
    #         user_prompt=prompt,
    #         model_settings=ModelSettings(
    #             timeout=config.model_timeout,
    #         ),
    #     )
    #     logger.debug(f"Game result: {result.output.value}")

    # Loop over knowledge gaps
    game_results = []
    for idx in range(len(dataset.cases)):
        gap = gaps[idx]

        # Pick a random second case for comparison
        idx_2 = random.randrange(len(dataset.cases))
        while idx_2 == idx:
            idx_2 = random.randrange(len(dataset.cases))
        gap_2 = gaps[idx_2]

        logger.debug(f"Gap {idx} vs Gap {idx_2}")

        prompt = textwrap.dedent(f"""
            <QUESTION>
            Which of the two search queries (A or B) shows more genuine curiosity and creativity, and is less formulaic?
            </QUESTION>

            <A> {gap} </A>

            <B> {gap_2} </B>
        """)
        # logger.debug(f"Prompt for evaluation:\n{prompt}")

        async with evaluation_agent:
            result = await evaluation_agent.run(
                user_prompt=prompt,
                model_settings=ModelSettings(
                    temperature=1.0,
                    timeout=config.model_timeout,
                ),
            )
            logger.debug(f"Game result: {result.output.value}")

        if result.output.value == "A":
            game_results.append((idx, idx_2))
        else:
            game_results.append((idx_2, idx))

        prompt = textwrap.dedent(f"""
            <QUESTION>
            Which of the two search queries (A or B) shows more genuine curiosity and creativity, and is less formulaic?
            </QUESTION>

            <A> {gap_2} </A>

            <B> {gap} </B>
        """)
        # logger.debug(f"Prompt for evaluation:\n{prompt}")

        async with evaluation_agent:
            result = await evaluation_agent.run(
                user_prompt=prompt,
                model_settings=ModelSettings(
                    temperature=1.0,
                    timeout=config.model_timeout,
                ),
            )
            logger.debug(f"Game result: {result.output.value}")

        if result.output.value == "B":
            game_results.append((idx, idx_2))
        else:
            game_results.append((idx_2, idx))

        logger.debug(f"Current games:\n{game_results}")

    # Calculate Bradley-Terry scores
    scores = choix.ilsr_pairwise(len(dataset.cases), game_results, alpha=0.01)
    idx_sorted = np.argsort(scores)
    gaps_sorted = [gaps[i] for i in idx_sorted]
    for i in range(len(dataset.cases)):
        logger.debug(f"Score for Gap {i}: {scores[i]:0.4f}")
    logger.debug(f"Sorted indices: {idx_sorted}")
    logger.debug("Sorted gaps (from worst to best):")
    for i in range(len(gaps_sorted)):
        logger.debug(gaps_sorted[i])


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
