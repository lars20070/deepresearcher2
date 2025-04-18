#!/usr/bin/env python3

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from deepresearcher2.models import WebSearchQuery
from deepresearcher2.prompts import query_instructions

# Models
model_name = "llama3.3"
# model_name = "firefunction-v2"
# model_name = "mistral-nemo"
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
    system_prompt=query_instructions,
    retries=5,
    instrument=True,
)
