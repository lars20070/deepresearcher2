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


async def generate_search_summaries(
    path: Path = Path("benchmarks/knowledge_gap/topics.json"),
    path_out: Path = Path("benchmarks/knowledge_gap/task.json"),
) -> None:
    """
    Generates a set of search summaries for the Knowledge Gap benchmark.

    Note that the generated benchmark contains only inputs i.e. the search summaries and no expected outputs.
    Note that we use the `summary_agent_evals` agent and not the production agent. This agent generates shorter
    summaries. Short summaries are easier to score due to the context window limitations of the models.

    Recommended model: qwen2.5:72b
    Please set in the .env file.

    Args:
        path (Path): The path to the input JSON file containing topics.
        path_out (Path): The path to the output JSON file for the generated summaries.
    """
    logger.info("Generating search summaries for Knowledge Gap benchmark.")

    logger.info("Loading topics from file.")
    topics = json.loads(path.read_text(encoding="utf-8"))

    # Loop over topics
    cases: list[Case[str, str, Any]] = []
    # for idx, topic in enumerate(topics[:3]):
    for idx, topic in enumerate(topics):
        logger.info(f"Case {idx}/{len(topics)} with topic: {topic}")

        # Run web search
        results = searxng_search(
            query=topic + " -filetype:pdf",
            max_results=3,
            max_content_length=2000,
        )
        logger.debug(f"number of web search results: {len(results)}")
        xml = format_as_xml(results, root_tag="search_results")

        logger.info(f"Generating summary for case {idx}/{len(topics)}")
        prompt = (
            f"Please summarize the provided web search results for the topic <TOPIC>{topic}</TOPIC>.\n"
            f"List of web search results:\n{xml}\n"
            "Do NOT include bullet points! Do NOT use Markdown formatting!"
        )
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

        logger.info("Adding new case to the benchmark dataset.")
        case = Case(
            name=f"knowledge_gap_{idx:03d}",
            inputs=result.output,
            metadata={"topic": topic},
        )
        cases.append(case)
        dataset: Dataset[str, str, Any] = Dataset[str, str, Any](cases=cases)

        logger.info("Serializing benchmark dataset to file.")
        path_out.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_file(path_out)


def main() -> None:
    asyncio.run(generate_search_summaries())


if __name__ == "__main__":
    main()
