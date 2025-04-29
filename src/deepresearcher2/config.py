#!/usr/bin/env python3
from enum import Enum

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class SearchEngine(str, Enum):
    duckduckgo = "duckduckgo"
    tavily = "tavily"
    perplexity = "perplexity"


class Config(BaseSettings):
    """
    Configuration settings for the application.
    """

    # Workflow settings
    topic: str = Field(default="petrichor", description="topic to be researched", min_length=2)
    max_research_loops: int = Field(default=3, description="number of search-summary-reflection loops")
    max_web_search_results: int = Field(default=2, description="number of results in a single web search")
    search_engine: SearchEngine = Field(default=SearchEngine.duckduckgo, description="search engine for the web searches")
    reports_folder: str = Field(default="reports/", description="output directory for the final reports")
    logs2logfire: bool = Field(default=False, description="Post all logs to Logfire. If false, some logs are written to a local log file.")

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
    perplexity_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


config = Config()
