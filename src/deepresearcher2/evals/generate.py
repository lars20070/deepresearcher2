#!/usr/bin/env python3
import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic_ai import format_as_xml
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset

from deepresearcher2.agents import summary_agent_evals
from deepresearcher2.config import config
from deepresearcher2.logger import logger
from deepresearcher2.utils import (
    remove_reasoning_tags,
    searxng_search,
)


async def generate_search_summaries() -> None:
    logger.info("Generating search summaries.")

    path = Path("benchmarks/knowledge_gap/topics.json")
    topics = json.loads(path.read_text(encoding="utf-8"))

    # Loop over topics
    cases: list[Case[str, str, Any]] = []
    for idx, topic in enumerate(topics[:5]):
        logger.info(f"Topic: {topic}")

        # Run web search
        results = searxng_search(topic + " -filetype:pdf", max_results=3)
        logger.debug(f"number of results: {len(results)}")
        xml = format_as_xml(results, root_tag="search_results")

        prompt = f"Please summarize the provided web search results for the topic <TOPIC>{topic}</TOPIC>.\n List of web search results:\n{xml}"

        # Generate the summary
        async with summary_agent_evals:
            result = await summary_agent_evals.run(
                user_prompt=prompt,
                model_settings=ModelSettings(
                    temperature=config.temperature_summary,
                    timeout=config.model_timeout,
                ),
            )

            result.output = remove_reasoning_tags(result.output)
            logger.debug(f"Web search summary:\n{result.output}")

        case = Case(
            name=f"knowledge_gap_{idx:03d}",
            inputs=result.output,
            metadata={"topic": topic},
        )
        cases.append(case)

        dataset: Dataset[str, str, Any] = Dataset[str, str, Any](cases=cases)

        logger.info("Serializing benchmark dataset to file.")
        out_path = Path("benchmarks/knowledge_gap/task.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_file(out_path)


def main() -> None:
    asyncio.run(generate_search_summaries())


if __name__ == "__main__":
    main()
