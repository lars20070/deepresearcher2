#!/usr/bin/env python3
import json
from pathlib import Path

from deepresearcher2.logger import logger


def generate_search_summaries() -> None:
    logger.info("Generating search summaries.")

    path = Path("benchmarks/knowledge_gap/topics.json")
    topics = json.loads(path.read_text(encoding="utf-8"))
    logger.debug(f"Loaded topics: {topics}")


def main() -> None:
    generate_search_summaries()


if __name__ == "__main__":
    main()
