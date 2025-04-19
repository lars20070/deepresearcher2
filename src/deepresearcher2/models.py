#!/usr/bin/env python3
from pydantic import BaseModel, Field, HttpUrl


class WebSearchQuery(BaseModel):
    query: str = Field(..., description="search query")
    aspect: str = Field(..., description="aspect of the topic being researched")
    rationale: str = Field(..., description="rationale for the search query")


class WebSearchResult(BaseModel):
    title: str = Field(..., description="short descriptive title of the web search result")
    url: HttpUrl = Field(..., description="URL of the web search result")
    content: str = Field(..., description="main content of the web search result in Markdown format")
