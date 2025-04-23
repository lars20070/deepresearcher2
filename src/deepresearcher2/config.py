#!/usr/bin/env python3
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import env  # noqa: F401


class Config(BaseSettings):
    """
    Configuration settings for the application.
    """

    # Workflow settings
    topic: str = Field(default="petrichor", description="topic to be researched", min_length=2)
    max_research_loops: int = Field(default=3, description="number of search-summary-reflection loops")
    max_web_search_results: int = Field(default=2, description="number of results in a single web search")

    # API keys
    tavily_api_key: str | None = None
    gemini_api_key: str | None = None
    logfire_token: str | None = None
    riza_api_key: str | None = None
    openai_api_key: str | None = None
    weather_api_key: str | None = None
    geo_api_key: str | None = None
    anthropic_api_key: str | None = None
    brave_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


config = Config()
