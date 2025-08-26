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
    brave = "brave"
    serper = "serper"
    searxng = "searxng"


class Model(str, Enum):
    llama33 = "llama3.3"
    qwen25_72b = "qwen2.5:72b"
    qwen3_8b = "qwen3:8b"
    qwen3_32b = "qwen3:32b"
    gptoss = "gpt-oss"
    gpt4o = "openai:gpt-4o"
    gpt4omini = "openai:gpt-4o-mini"


class Config(BaseSettings):
    """
    Configuration settings for the application.
    """

    # Workflow settings
    topic: str = Field(default="petrichor", description="Topic to be researched", min_length=2)
    max_research_loops: int = Field(default=3, description="Number of search-summary-reflection loops")
    max_web_search_results: int = Field(default=2, description="Number of results in a single web search")
    search_engine: SearchEngine = Field(default=SearchEngine.searxng, description="Search engine for the web searches")
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama host URL")
    searxng_host: str = Field(default="http://localhost:8080", description="SearXNG host URL")
    model: Model = Field(default=Model.llama33, description="Model to be used by all agents")
    model_timeout: int = Field(default=600, description="Timeout in seconds for the model requests")
    reports_folder: str = Field(default="reports/", description="Output directory for the final reports")
    logs2logfire: bool = Field(default=False, description="Post all logs to Logfire. If false, some logs are written to a local log file.")
    temperature_query: float = Field(default=1.0, description="Temperature for the model generating the web queries")
    temperature_summary: float = Field(default=1.0, description="Temperature for the model generating the summaries of the web search results")
    temperature_reflection: float = Field(default=1.0, description="Temperature for the model generating the reflection on the summaries")
    temperature_final_summary: float = Field(default=1.0, description="Temperature for the model generating the final summary of the research report")

    # API keys
    tavily_api_key: str | None = None
    gemini_api_key: str | None = None
    logfire_token: str | None = None
    openai_api_key: str | None = None
    weather_api_key: str | None = None
    geo_api_key: str | None = None
    anthropic_api_key: str | None = None
    brave_api_key: str | None = None
    perplexity_api_key: str | None = None
    serper_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


config = Config()
