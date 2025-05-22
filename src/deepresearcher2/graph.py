#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import format_as_xml
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from .agents import final_summary_agent, query_agent, reflection_agent, summary_agent
from .config import SearchEngine, config
from .logger import logger
from .models import DeepState, Reference, Reflection, WebSearchSummary
from .prompts import query_instructions_with_reflection, query_instructions_without_reflection
from .utils import duckduckgo_search, export_report, perplexity_search, remove_reasoning_tags, tavily_search

load_dotenv()


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
        search_params = {
            "query": ctx.state.search_query.query,
            "max_results": config.max_web_search_results,
            "max_content_length": 12000,
        }
        if config.search_engine == SearchEngine.duckduckgo:
            ctx.state.search_results = duckduckgo_search(**search_params)
        elif config.search_engine == SearchEngine.tavily:
            ctx.state.search_results = tavily_search(**search_params)
        elif config.search_engine == SearchEngine.perplexity:
            ctx.state.search_results = perplexity_search(ctx.state.search_query.query)
        else:
            message = f"Unsupported search engine: {config.search_engine}"
            logger.error(message)
            raise ValueError(message)

        logger.debug(f"Web search results:\n{format_as_xml(ctx.state.search_results, root_tag='search_results')}")

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
            result = await summary_agent.run(
                user_prompt=f"Please summarize the provided web search results for the topic <TOPIC>{ctx.state.topic}</TOPIC>."
            )
            result.output = remove_reasoning_tags(result.output)
            logger.debug(f"Web search summary:\n{result.output}")

            # Transfer search result references to the summary
            references = []
            for ref in ctx.state.search_results:
                if ref.title and ref.url:
                    references.append(Reference(title=ref.title, url=ref.url))

            summary = WebSearchSummary(
                summary=result.output,  # Summary from the agent
                aspect=ctx.state.search_query.aspect,  # Aspect from the search query
                references=references,  # References from the search results
            )
            logger.debug(f"Summary result:\n{format_as_xml(summary, root_tag='single_search_summary')}")

            # Append the summary to the list of all search summaries
            ctx.state.search_summaries = ctx.state.search_summaries or []
            ctx.state.search_summaries.append(summary)

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
        if ctx.state.count < config.max_research_loops:
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
                reflection_run_output = await reflection_agent.run( # Name changed for clarity
                    user_prompt=f"Please reflect on the provided web search summaries for the topic <TOPIC>{ctx.state.topic}</TOPIC>."
                )
                # reflection_run_output.output should now contain knowledge_gaps, covered_topics, AND exploratory_paths
                logger.debug(f"Reflection knowledge gaps:\n{reflection_run_output.output.knowledge_gaps}")
                logger.debug(f"Reflection covered topics:\n{reflection_run_output.output.covered_topics}")
                logger.debug(f"Reflection exploratory paths:\n{reflection_run_output.output.exploratory_paths}")

                ctx.state.reflection = Reflection( # This is our model from models.py
                    knowledge_gaps=reflection_run_output.output.knowledge_gaps,
                    covered_topics=reflection_run_output.output.covered_topics,
                    exploratory_paths=reflection_run_output.output.exploratory_paths,
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
            logger.debug(f"Final summary:\n{final_summary.output.summary}")

        # Compile the final report
        report = f"# {topic}\n\n"
        report += f"{final_summary.output.summary}\n\n"  # Overall summary
        for summary in ctx.state.search_summaries:  # Summaries of individual searches
            report += f"\n## {summary.aspect}\n\n"
            report += f"{summary.summary}\n\n"
            report += "### References\n"
            for ref in summary.references:
                report += f"- {ref.title} [{ref.url}]({ref.url})\n"
            report += "\n"

        # Export the report
        export_report(report=report, topic=topic, output_dir=config.reports_folder)

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
    state = DeepState(topic=config.topic, count=1)
    result = await graph.run(WebSearch(), state=state)
    logger.debug(f"Result: {result.output}")

    # Mermaid code
    # mermaid_code = graph.mermaid_code(start_node=WebSearch())
    # logger.debug(f"Mermaid graph:\n{mermaid_code}")


def main() -> None:
    """
    Main function containing the deep research workflow.
    """

    logger.info("Starting deep research.")
    asyncio.run(deepresearch())


if __name__ == "__main__":
    main()
