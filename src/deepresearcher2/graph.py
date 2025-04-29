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

load_dotenv()

# Config parameters
topic = "petrichor"
max_research_loops = 2
max_web_search_results = 3

# Prompts
query_instructions_without_reflection = """
Please generate a targeted web search query for a specific topic.

<REQUIREMENTS>
1. **Specificity:** The query must be specific and focused on a single aspect of the topic.
2. **Relevance:** Ensure the query directly relates to the core topic.
3. **Conciseness:** The query string must not exceed 100 characters.
4. **Aspect Definition:** The 'aspect' value must describe the specific focus of the query, excluding the main topic itself.
5. **Rationale:** Briefly explain why this query is relevant for researching the topic.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "query": The generated search query string.
- "aspect": The specific aspect targeted by the query.
- "rationale": A brief justification for the query's relevance.
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "query": "Rosalind Franklin DNA structure contributions",
    "aspect": "DNA structure contributions",
    "rationale": "Focuses on her specific scientific contributions rather than general biography, addressing a key area of her work."
}
```
</EXAMPLE_OUTPUT>

Provide your response in JSON format."""

query_instructions_with_reflection = """
Please generate a targeted web search query for a specific topic. The query will gather information related to a specific topic
based on specific knowledge gaps.

<INPUT_FORMAT>
You will receive reflections in XML with `<reflections>` tags containing:
- `<knowledge_gaps>`: information that has not been covered in the previous search results
- `<covered_topics>`: information that has been covered and should not be repeated
</INPUT_FORMAT>

<REQUIREMENTS>
1. The knowledge gaps form the basis of the search query.
2. Identify the most relevant point in the knowledge gaps and use it to create a focused search query.
3. Pick only one item from the list of knowledge gaps. Do not include all knowledge gaps in the construction of the query.
4. Check that the query is not covering any aspects listed in the list of covered topics.
5. Check that the query is at least vaguely related to the topic.
6. Do not include the topic in the aspect of the query, since this is too broad.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "query": The generated search query string.
- "aspect": The specific aspect targeted by the query.
- "rationale": A brief justification for the query's relevance.
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "query": "Rosalind Franklin DNA structure contributions",
    "aspect": "DNA structure contributions",
    "rationale": "Focuses on her specific scientific contributions rather than general biography, addressing a key area of her work."
}
```
</EXAMPLE_OUTPUT>

Provide your response in JSON format."""

summary_instructions = """
You are a search results summarizer. Your task is to generate a comprehensive summary from web search results that is relevant to the user's topic.

<INPUT_FORMAT>
You will receive web search results in XML with `<WebSearchResult>` tags containing:
- `<title>`: Descriptive title
- `<url>`: Source URL
- `<summary>`: Brief summary 
- `<content>`: Raw content
</INPUT_FORMAT>

<REQUIREMENTS>
1. Compile all topic-relevant information from search results
2. Create a summary at least 1000 words long
3. Ensure coherent information flow
4. Keep content relevant to the user topic
5. The "aspect" value must be specific to the information and must NOT include the topic itself
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "summary": Direct compilation of ALL information (minimum 1000 words) without preamble, XML tags, or Markdown
- "aspect": The specific aspect of the topic being researched (excluding the topic itself)
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "summary": "Petrichor refers to the earthy scent produced when rain falls on dry soil or ground, often experienced as a pleasant smell.
    It is characterized by its distinct aroma, which is typically associated with the smell of rain on dry earth.",
    "aspect": "definition and meaning",
}
```
</EXAMPLE_OUTPUT>

Provide your response in JSON format."""

reflection_instructions = """
You analyze web search summaries to identify knowledge gaps and coverage areas.

<INPUT_FORMAT>
You will receive web search summaries in XML with `<WebSearchSummary>` tags containing:
- `<summary>`: Summary of the search result as text
- `<aspect>`: Specific aspect discussed in the summary
</INPUT_FORMAT>

<REQUIREMENTS>
1. Analyze all summaries thoroughly
2. Identify knowledge gaps needing deeper exploration
3. Identify well-covered topics to avoid repetition in future searches
4. Be curious and creative with knowledge gaps! Never return "None" or "Nothing".
5. Use keywords and phrases only, not sentences
6. Return only the JSON object - no explanations or formatting
7. Consider technical details, implementation specifics, and emerging trends
8. Consider second and third-order effects or implications of the topic when exploring knowledge gaps
9. Be thorough yet concise
10. Ensure that the list of knowledgae gaps and the list of covered topics are distinct and do not overlap.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "knowledge_gaps": List of specific aspects requiring further research
- "covered_topics": List of aspects already thoroughly covered
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "knowledge_gaps": [
        "scientific mechanisms",
        "psychological effects",
        "regional variations",
        "commercial applications",
        "cultural significance"
    ],
    "covered_topics": [
        "basic definition",
        "etymology",
        "general description"
    ]
}
```
</EXAMPLE_OUTPUT>

Provide your response in JSON format."""

final_summary_instructions = """
You are a precise information compiler that transforms web search summaries into comprehensive reports. Follow these instructions carefully.

<INPUT_FORMAT>
You will receive web search summaries in XML with `<WebSearchSummary>` tags containing:
- `<summary>`: Summary of the search result as text
- `<aspect>`: Specific aspect discussed in the summary
</INPUT_FORMAT>

<REQUIREMENTS>
1. Extract and consolidate all relevant information from the provided summaries
2. Create a coherent, well-structured report that flows logically
3. Focus on delivering comprehensive information relevant to the implied topic
4. When search results contain conflicting information, present both perspectives and indicate the discrepancy
5. Structure your report into 3-5 paragraphs of reasonable length (150-300 words each)
6. Avoid redundancy while ensuring all important information is included
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "summary": The comprehensive report, starting directly with the information without preamble.
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "summary": "Your comprehensive report here. Start directly with the information without preamble.
    Write multiple cohesive paragraphs with logical flow."
}
```
</EXAMPLE_OUTPUT>

The JSON response must be properly formatted with quotes escaped within the summary value. Do not include any text outside the JSON object.
"""


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

        # Run the search
        ctx.state.search_results = duckduckgo_search(ctx.state.search_query.query)

        return SummarizeSearchResults()


@dataclass
class SummarizeSearchResults(BaseNode[DeepState]):
    """
    Summarize Search Results node.
    """

    async def run(self, ctx: GraphRunContext[DeepState]) -> ReflectOnSearch:
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
        # Flow control
        # Should we ponder on the next web search or compile the final report?
        if ctx.state.count < max_research_loops:
            ctx.state.count += 1

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
        topic = ctx.state.topic

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

        # Export the report
        export_report(report=report, topic=topic)

        return End("End of deep research workflow.\n\n")


# Workflow
async def deepresearch() -> None:
    # Define the agent graph
    graph = Graph(nodes=[WebSearch, SummarizeSearchResults, ReflectOnSearch, FinalizeSummary])

    # Run the agent graph
    await graph.run(WebSearch(), state=DeepState(topic=topic, count=1))


def main() -> None:
    asyncio.run(deepresearch())


if __name__ == "__main__":
    main()
