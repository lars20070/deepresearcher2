#!/usr/bin/env python3
from __future__ import annotations as _annotations

import json
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset

from deepresearcher2.logger import logger


def import_bigbench(path: Path = Path("../BIG-bench")) -> None:
    """
    Import BIG-bench dataset and convert it to PydanticAI format.
    https://github.com/google/BIG-bench

    Args:
        path (Path): Path to the BIG-bench repository.
    """
    logger.info("Importing BIG-bench dataset.")
    path = path / "bigbench/benchmark_tasks/dark_humor_detection/task.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data: list[dict[str, Any]] = data.get("examples", [])

    logger.info("Converting BIG-bench dataset to PydanticAI format.")
    cases: list[Case[str, bool, Any]] = []
    for idx, example in enumerate(data):
        text = example.get("input", "")
        scores = example.get("target_scores", {})
        joke = bool(scores.get("joke", 0))
        logger.debug(f"Case {idx:02d}: {text[:50]} ...    Joke: {joke}")

        case = Case(
            name=f"dark_humor_detection_{idx:02d}",
            inputs=text,
            expected_output=joke,
        )
        cases.append(case)

    dataset: Dataset[str, bool, Any] = Dataset[str, bool, Any](cases=cases)

    logger.info("Serializing benchmark dataset to file.")
    out_path = Path("data/dark_humor_detection/task.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_file(out_path)


def main() -> None:
    """
    Main function containing all import workflows.
    """
    logger.info("Starting import workflow.")
    import_bigbench()


if __name__ == "__main__":
    main()
