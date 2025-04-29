#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from pydantic_ai import Agent, format_as_xml
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from .logger import logger
from .prompts import (
    final_summary_instructions,
    query_instructions_with_reflection,
    query_instructions_without_reflection,
    reflection_instructions,
    summary_instructions,
)

load_dotenv()

# Config parameters
topic = "petrichor"
max_research_loops = 2
max_web_search_results = 3


# Models
class DeepState(BaseModel):
    topic: str = Field(default="petrichor", description="main research topic")
    search_query: WebSearchQuery | None = Field(default=None, description="single search query for the current loop")
    search_results: list[WebSearchResult] | None = Field(default=None, description="list of search results in the current loop")
    search_summaries: list[WebSearchSummary] | None = Field(default=None, description="list of all search summaries of the past loops")
    reflection: Reflection | None = Field(default=None, description="reflection on the search results of the previous current loop")
    count: int = Field(default=0, description="counter for tracking iteration count")


class WebSearchQuery(BaseModel):
    query: str = Field(..., description="search query")
    aspect: str = Field(..., description="aspect of the topic being researched")
    rationale: str = Field(..., description="rationale for the search query")


class WebSearchResult(BaseModel):
    title: str = Field(..., description="short descriptive title of the web search result")
    url: str = Field(..., description="URL of the web search result")
    summary: str | None = Field(None, description="summary of the web search result")
    content: str | None = Field(None, description="main content of the web search result in Markdown format")


class Reference(BaseModel):
    title: str = Field(..., description="title of the reference")
    url: str = Field(..., description="URL of the reference")


class WebSearchSummary(BaseModel):
    summary: str = Field(..., description="summary of multiple web search results")
    aspect: str = Field(..., description="aspect of the topic being summarized")


class Reflection(BaseModel):
    knowledge_gaps: list[str] = Field(..., description="aspects of the topic which require further exploration")
    covered_topics: list[str] = Field(..., description="aspects of the topic which have already been covered sufficiently")


class FinalSummary(BaseModel):
    summary: str = Field(..., description="summary of the topic for the final report")


# Agents
model = OpenAIModel(model_name="llama3.3", provider=OpenAIProvider(base_url="http://localhost:11434/v1"))
query_agent = Agent(model=model, output_type=WebSearchQuery, system_prompt="")
summary_agent = Agent(model=model, output_type=WebSearchSummary, system_prompt=summary_instructions)
reflection_agent = Agent(model=model, output_type=Reflection, system_prompt=reflection_instructions)
final_summary_agent = Agent(model=model, output_type=FinalSummary, system_prompt=final_summary_instructions)


def duckduckgo_search(query: str) -> list[WebSearchResult]:
    """
    Perform a web search using DuckDuckGo and return a list of results.

    Args:
        query (str): The search query to execute.

    Returns:
        list[WebSearchResult]: list of search results
    """
    logger.info(f"DuckDuckGo web search for: {query}")

    # Run the search
    with DDGS() as ddgs:
        ddgs_results = list(ddgs.text(query, max_results=max_web_search_results))

    # Convert to pydantic objects
    results = []
    for r in ddgs_results:
        result = WebSearchResult(title=r.get("title"), url=r.get("href"), content=r.get("body"))
        results.append(result)

    return results


def export_report(report: str, topic: str = "Report") -> None:
    """
    Export the report to markdown.

    Args:
        report (str): The report content in markdown format.
        topic (str): The topic of the report. Defaults to "Report".
    """
    file_name = re.sub(r"[^a-zA-Z0-9]", "_", topic).lower()
    path_md = os.path.join("reports/", f"{file_name}.md")
    with open(path_md, "w", encoding="utf-8") as f:
        f.write(report)


# Nodes
@dataclass
class WebSearch(BaseNode[DeepState]):
    """
    Web Search node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> SummarizeSearchResults:
        logger.debug(f"Running Web Search with count number {ctx.state.count}.")

        topic = ctx.state.topic

        @query_agent.system_prompt
        def add_reflection() -> str:
            """
            Add reflection from the previous loop to the system prompt.
            """
            if ctx.state.reflection:
                xml = format_as_xml(ctx.state.reflection, root_tag="reflection")
                return query_instructions_with_reflection + f"Reflection on existing knowledge:\n{xml}\n" + "Provide your response in JSON format."
            else:
                return query_instructions_without_reflection

        # Generate the query
        async with query_agent.run_mcp_servers():
            prompt = f"Please generate a web search query for the following topic: <TOPIC>{topic}</TOPIC>"
            result = await query_agent.run(prompt)
            ctx.state.search_query = result.output
            logger.debug(f"Web search query:\n{ctx.state.search_query.model_dump_json(indent=2)}")

        # Run the search
        ctx.state.search_results = duckduckgo_search(ctx.state.search_query.query)

        return SummarizeSearchResults()


@dataclass
class SummarizeSearchResults(BaseNode[DeepState]):
    """
    Summarize Search Results node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> ReflectOnSearch:
        logger.debug(f"Running Summarize Search Results with count number {ctx.state.count}.")

        @summary_agent.system_prompt
        def add_web_search_results() -> str:
            """
            Add web search results to the system prompt.
            """
            xml = format_as_xml(ctx.state.search_results, root_tag="search_results")
            return f"List of web search results:\n{xml}"

        # Generate the summary
        async with summary_agent.run_mcp_servers():
            summary = await summary_agent.run(
                user_prompt=f"Please summarize the provided web search results for the topic <TOPIC>{ctx.state.topic}</TOPIC>."
            )
            logger.debug(f"Web search summary:\n{summary.output.model_dump_json(indent=2)}")

            # Append the summary to the list of all search summaries
            ctx.state.search_summaries = ctx.state.search_summaries or []
            ctx.state.search_summaries.append(
                WebSearchSummary(
                    summary=summary.output.summary,
                    aspect=summary.output.aspect,
                )
            )

        return ReflectOnSearch()


@dataclass
class ReflectOnSearch(BaseNode[DeepState]):
    """
    Reflect on Search node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> WebSearch | FinalizeSummary:
        logger.debug(f"Running Reflect on Search with count number {ctx.state.count}.")

        # Flow control
        # Should we ponder on the next web search or compile the final report?
        if ctx.state.count < max_research_loops:
            ctx.state.count += 1

            xml = format_as_xml(ctx.state.search_summaries, root_tag="search_summaries")
            logger.debug(f"Search summaries:\n{xml}")

            @reflection_agent.system_prompt
            def add_search_summaries() -> str:
                """
                Add search summaries to the system prompt.
                """
                xml = format_as_xml(ctx.state.search_summaries, root_tag="search_summaries")
                return f"List of search summaries:\n{xml}"

            # Reflect on the summaries so far
            async with reflection_agent.run_mcp_servers():
                reflection = await reflection_agent.run(
                    user_prompt=f"Please reflect on the provided web search summaries for the topic <TOPIC>{ctx.state.topic}</TOPIC>."
                )
                logger.debug(f"Reflection knowledge gaps:\n{reflection.output.knowledge_gaps}")
                logger.debug(f"Reflection covered topics:\n{reflection.output.covered_topics}")

                ctx.state.reflection = Reflection(
                    knowledge_gaps=reflection.output.knowledge_gaps,
                    covered_topics=reflection.output.covered_topics,
                )

            return WebSearch()
        else:
            return FinalizeSummary()


@dataclass
class FinalizeSummary(BaseNode[DeepState]):
    """
    Finalize Summary node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> End:
        logger.debug("Running Finalize Summary.")

        topic = ctx.state.topic

        xml = format_as_xml(ctx.state.search_summaries, root_tag="search_summaries")
        logger.debug(f"Search summaries:\n{xml}")

        @final_summary_agent.system_prompt
        def add_search_summaries() -> str:
            """
            Add search summaries to the system prompt.
            """
            xml = format_as_xml(ctx.state.search_summaries, root_tag="search_summaries")
            return f"List of search summaries:\n{xml}"

        # Finalize the summary of the entire report
        async with final_summary_agent.run_mcp_servers():
            final_summary = await final_summary_agent.run(
                user_prompt=f"Please summarize all web search summaries for the topic <TOPIC>{ctx.state.topic}</TOPIC>."
            )
            report = f"## {topic}\n\n" + final_summary.output.summary
            logger.debug(f"Final report:\n{report}")

        # Export the report
        export_report(report=report, topic=topic)

        return End("End of deep research workflow.\n\n")


# Workflow
async def deepresearch() -> None:
    """
    Graph use
    """
    logger.info("Starting deep research workflow.")

    # Define the agent graph
    graph = Graph(nodes=[WebSearch, SummarizeSearchResults, ReflectOnSearch, FinalizeSummary])

    # Run the agent graph
    state = DeepState(topic=topic, count=1)
    result = await graph.run(WebSearch(), state=state)
    logger.debug(f"Result: {result.output}")


def main() -> None:
    """
    Main function containing the deep research workflow.
    """

    logger.info("Starting deep research.")
    asyncio.run(deepresearch())


if __name__ == "__main__":
    main()
