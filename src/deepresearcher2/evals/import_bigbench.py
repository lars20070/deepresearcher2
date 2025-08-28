#!/usr/bin/env python3
from __future__ import annotations as _annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset

from deepresearcher2.logger import logger


def import_codenames(path: Path = Path("../BIG-bench")) -> None:
    """
    Import BIG-bench dataset and convert it to PydanticAI format.
    https://github.com/google/BIG-bench

    Args:
        path (Path): Path to the BIG-bench repository.
    """
    logger.info("Importing BIG-bench dataset.")
    path = path / "bigbench/benchmark_tasks/codenames/task.json"
    data_json = json.loads(path.read_text(encoding="utf-8"))
    data: list[dict[str, Any]] = data_json.get("examples", [])

    logger.info("Converting BIG-bench dataset to PydanticAI format.")
    cases: list[Case[str, str, Any]] = []
    for idx, example in enumerate(data):
        text = example.get("input", "")
        target = example.get("target", "")
        logger.debug(f"Case {idx:02d}: {text[:50]} ...    Target: {target}")

        case = Case(
            name=f"codenames{idx:02d}",
            inputs=text,
            expected_output=target,
        )
        cases.append(case)

    dataset: Dataset[str, str, Any] = Dataset[str, str, Any](cases=cases)

    logger.info("Serializing benchmark dataset to file.")
    out_path = Path("benchmarks/codenames/task.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_file(out_path)


class Response(str, Enum):
    joke = "joke"
    nojoke = "no joke"


def import_darkhumordetection(path: Path = Path("../BIG-bench")) -> None:
    """
    Import BIG-bench dataset and convert it to PydanticAI format.
    https://github.com/google/BIG-bench

    Args:
        path (Path): Path to the BIG-bench repository.
    """
    logger.info("Importing BIG-bench dataset.")
    path = path / "bigbench/benchmark_tasks/dark_humor_detection/task.json"
    data_json = json.loads(path.read_text(encoding="utf-8"))
    data: list[dict[str, Any]] = data_json.get("examples", [])

    logger.info("Converting BIG-bench dataset to PydanticAI format.")
    cases: list[Case[str, Response, Any]] = []
    for idx, example in enumerate(data):
        text = example.get("input", "")
        scores = example.get("target_scores", {})
        joke: Response = Response.joke if scores.get("joke", 0) else Response.nojoke
        logger.debug(f"Case {idx:02d}: {text[:50]} ...    Response: {joke.value}")

        case = Case(
            name=f"dark_humor_detection_{idx:02d}",
            inputs=text,
            expected_output=joke,
        )
        cases.append(case)

    logger.info("Add evaluators.")
    dataset: Dataset[str, Response, Any] = Dataset[str, Response, Any](cases=cases)

    logger.info("Serializing benchmark dataset to file.")
    out_path = Path("benchmarks/dark_humor_detection/task.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_file(out_path)


def import_rephrase(path: Path = Path("../BIG-bench")) -> None:
    """
    Import BIG-bench dataset and convert it to PydanticAI format.
    https://github.com/google/BIG-bench

    Args:
        path (Path): Path to the BIG-bench repository.
    """
    logger.info("Importing BIG-bench dataset.")
    path = path / "bigbench/benchmark_tasks/rephrase/task.json"
    data_json = json.loads(path.read_text(encoding="utf-8"))
    data: list[dict[str, Any]] = data_json.get("examples", [])

    logger.info("Converting BIG-bench dataset to PydanticAI format.")
    cases: list[Case[str, list[str], Any]] = []
    for idx, example in enumerate(data):
        text = example.get("input", "")
        targets = example.get("target", "")
        if isinstance(targets, str):
            targets = [targets]
        logger.debug(f"Case {idx:02d}: {text[:50]} ...    Targets: {targets}")

        case = Case(
            name=f"rephrase_{idx:02d}",
            inputs=text,
            expected_output=targets,
        )
        cases.append(case)

    dataset: Dataset[str, list[str], Any] = Dataset[str, list[str], Any](cases=cases)

    logger.info("Serializing benchmark dataset to file.")
    out_path = Path("benchmarks/rephrase/task.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_file(out_path)


def main() -> None:
    """
    Main function containing all import workflows.
    """
    logger.info("Starting import of benchmark datasets.")
    import_codenames()
    import_darkhumordetection()
    import_rephrase()


if __name__ == "__main__":
    main()
