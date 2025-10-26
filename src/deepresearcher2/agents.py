#!/usr/bin/env python3
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import config
from .models import FinalSummary, GameResult, Reflection, WebSearchQuery
from .prompts import EVALUATION_INSTRUCTIONS, FINAL_SUMMARY_INSTRUCTIONS, REFLECTION_INSTRUCTIONS, SUMMARY_INSTRUCTIONS, SUMMARY_INSTRUCTIONS_EVALS

load_dotenv()

# Models
if "openai" in config.model.value:
    # Cloud model
    model = config.model.value
else:
    # Local Ollama model
    model = OpenAIChatModel(
        model_name=config.model.value,
        provider=OpenAIProvider(base_url=f"{config.ollama_host}/v1"),
    )

# MCP serves
mcp_server_duckduckgo = MCPServerStdio("uvx", args=["duckduckgo-mcp-server"])

# Agents
# Note that we provide internet access to the query writing agent. This might be a bit circular.
# TODO: Check whether this improves the queries or is just a waste of time.
query_agent = Agent(
    model=model,
    # toolsets=[mcp_server_duckduckgo],
    output_type=WebSearchQuery,
    system_prompt="",
    retries=5,
    instrument=True,
)

# Note that we provide internet access to the summary agent. Maybe the agent wants to clarify some facts.
# TODO: Check whether this improves the queries or is just a waste of time.
# Sometimes the model fails to reply with JSON. In this case, the model tries to google for a fix. Better switch off the internet access.
summary_agent = Agent(
    model=model,
    # toolsets=[mcp_server_duckduckgo],
    output_type=str,
    system_prompt=SUMMARY_INSTRUCTIONS,
    retries=5,
    instrument=True,
)

# This agent is specifically for the generation of the knowledge_gap benchmark. Not used for production.
summary_agent_evals = Agent(
    model=model,
    # toolsets=[mcp_server_duckduckgo],
    output_type=str,
    system_prompt=SUMMARY_INSTRUCTIONS_EVALS,
    retries=5,
    instrument=True,
)

reflection_agent = Agent(
    model=model,
    output_type=Reflection,
    system_prompt=REFLECTION_INSTRUCTIONS,
    retries=5,
    instrument=True,
)

final_summary_agent = Agent(
    model=model,
    output_type=FinalSummary,
    system_prompt=FINAL_SUMMARY_INSTRUCTIONS,
    retries=5,
    instrument=True,
)

evaluation_agent = Agent(
    model=model,
    output_type=GameResult,
    system_prompt=EVALUATION_INSTRUCTIONS,
    retries=5,
    instrument=True,
)
