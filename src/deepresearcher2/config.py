#!/usr/bin/env python3
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Configuration settings for the application.
    Read configuration from environment variables or .env file.
    """

    # Load environment variables from .env file
    load_dotenv()

    # Workflow settings
    topic: str = Field(
        default=os.getenv("TOPIC", "petrichor"),
        description="topic to be researched",
        min_length=2,
    )
    max_research_loops: int = Field(
        default=int(os.getenv("MAX_RESEARCH_LOOPS", "3")),
        description="number of search-summary-reflection loops",
    )
    max_web_search_results: int = Field(
        default=int(os.getenv("MAX_WEB_SEARCH_RESULTS", "2")),
        description="number of results in a single web search",
    )

    # API keys
    tavily_api_key: str | None = Field(default=os.getenv("TAVILY_API_KEY"))
    gemini_api_key: str | None = Field(default=os.getenv("GEMINI_API_KEY"))
    logfire_token: str | None = Field(default=os.getenv("LOGFIRE_TOKEN"))
    riza_api_key: str | None = Field(default=os.getenv("RIZA_API_KEY"))
    openai_api_key: str | None = Field(default=os.getenv("OPENAI_API_KEY"))
    weather_api_key: str | None = Field(default=os.getenv("WEATHER_API_KEY"))
    geo_api_key: str | None = Field(default=os.getenv("GEO_API_KEY"))
    anthropic_api_key: str | None = Field(default=os.getenv("ANTHROPIC_API_KEY"))
    brave_api_key: str | None = Field(default=os.getenv("BRAVE_API_KEY"))


config = Config()
