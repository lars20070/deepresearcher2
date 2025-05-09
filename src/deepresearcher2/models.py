#!/usr/bin/env python3
from __future__ import annotations as _annotations

from pydantic import BaseModel, Field


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
    references: list[Reference] | None = Field(None, description="list of references for the summary")

    # TODO: The model struggles with summaries that are too short. It seems the model does not understand the returned value error.
    # @field_validator("summary")
    # @classmethod
    # def validate_summary_length(cls, text: str) -> str:
    #     word_count = len(text.split())
    #     if word_count < 100:
    #         raise ValueError(
    #             f"The summary you have written is too short: {word_count} words. You are required to write a summary of at least 100 words."
    #         )
    #     if word_count > 400:
    #         raise ValueError(f"The summary you have written is too long: {word_count} words. The maximum allowed length is 400 words.")
    #     return text


class Reflection(BaseModel):
    knowledge_gaps: list[str] = Field(..., description="aspects of the topic which require further exploration")
    covered_topics: list[str] = Field(..., description="aspects of the topic which have already been covered sufficiently")


class FinalSummary(BaseModel):
    summary: str = Field(..., description="summary of the topic for the final report")
