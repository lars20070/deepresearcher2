#!/usr/bin/env python3
from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class DeepState:
    topic: str = "petrichor"
    search_query: WebSearchQuery | None = field(default_factory=lambda: None)
    search_results: list[WebSearchResult] | None = field(default_factory=lambda: None)
    count: int = 0
    summary: str | None = None


class WebSearchQuery(BaseModel):
    query: str = Field(..., description="search query")
    aspect: str = Field(..., description="aspect of the topic being researched")
    rationale: str = Field(..., description="rationale for the search query")


class WebSearchResult(BaseModel):
    title: str = Field(..., description="short descriptive title of the web search result")
    url: str = Field(..., description="URL of the web search result")
    content: str = Field(..., description="main content of the web search result in Markdown format")
