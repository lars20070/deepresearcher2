#!/usr/bin/env python3

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .models import FinalSummary, Reflection, WebSearchQuery, WebSearchSummary
from .prompts import final_summary_instructions, reflection_instructions, summary_instructions

# Models
model_name = "llama3.3"
# model_name = "firefunction-v2"  # fails during reflection task
# model_name = "mistral-nemo"  # Nemo is terrible in replying in JSON.
ollama_model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
)

# MCP serves
mcp_server_duckduckgo = MCPServerStdio("uvx", args=["duckduckgo-mcp-server"])

# Agents
# Note that we provide internet access to the query writing agent. This might be a bit circular.
# TODO: Check whether this improves the queries or is just a waste of time.
query_agent = Agent(
    model=ollama_model,
    # model="openai:gpt-4o",
    mcp_servers=[mcp_server_duckduckgo],
    output_type=WebSearchQuery,
    system_prompt="",
    retries=5,
    instrument=True,
)

# Note that we provide internet access to the summary agent. Maybe the agent wants to clarify some facts.
# TODO: Check whether this improves the queries or is just a waste of time.
# Sometimes the model fails to reply with JSON. In this case, the model tries to google for a fix. Better switch off the internet access.
summary_agent = Agent(
    model=ollama_model,
    # model="openai:gpt-4o",
    # mcp_servers=[mcp_server_duckduckgo],
    output_type=WebSearchSummary,
    system_prompt=summary_instructions,
    retries=5,
    instrument=True,
)

reflection_agent = Agent(
    model=ollama_model,
    # model="openai:gpt-4o",
    output_type=Reflection,
    system_prompt=reflection_instructions,
    retries=5,
    instrument=True,
)

final_summary_agent = Agent(
    model=ollama_model,
    # model="openai:gpt-4o",
    output_type=FinalSummary,
    system_prompt=final_summary_instructions,
    retries=5,
    instrument=True,
)
