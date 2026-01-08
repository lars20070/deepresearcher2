#!/usr/bin/env python3
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Config, Provider, config
from .logger import logger
from .models import FinalSummary, GameResult, Reflection, WebSearchQuery
from .prompts import EVALUATION_INSTRUCTIONS, FINAL_SUMMARY_INSTRUCTIONS, REFLECTION_INSTRUCTIONS, SUMMARY_INSTRUCTIONS, SUMMARY_INSTRUCTIONS_EVALS

load_dotenv()

Model_ = str | OpenAIChatModel


def create_model(config: Config) -> Model_:
    """
    Create a model instance based on provider configuration.

    For custom providers (Ollama, LM Studio), returns an OpenAIChatModel.
    For providers with native pydantic-ai support, returns a shorthand string.

    Args:
        config (Config): Configuration object containing provider and model info.

    Returns:
        Model_: Configured model instance.
    """
    logger.info(f"Creating a model for provider: {config.provider.value}")

    match config.provider:
        # Custom OpenAI-compatible endpoints for local models
        case Provider.ollama:
            return OpenAIChatModel(
                model_name=config.model.value,
                provider=OpenAIProvider(base_url=f"{config.ollama_host}/v1"),
            )
        case Provider.lmstudio:
            return OpenAIChatModel(
                model_name=config.model.value,
                provider=OpenAIProvider(base_url=f"{config.lmstudio_host}/v1"),
            )

        # Native pydantic-ai shorthand for cloud models
        # API keys are automatically read from env variables.
        case Provider.openrouter:
            return f"openrouter:{config.model.value}"
        case Provider.openai:
            return f"openai:{config.model.value}"
        case Provider.together:
            return f"together:{config.model.value}"
        case Provider.deepinfra:
            return f"deepinfra:{config.model.value}"
        case _:
            error_msg = f"Unsupported provider: {config.provider.value}"
            logger.error(error_msg)
            raise ValueError(error_msg)


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
MCP_SERVER_DUCKDUCKGO = MCPServerStdio("uvx", args=["duckduckgo-mcp-server"])

# Agents
# Note that we provide internet access to the query writing agent. This might be a bit circular.
# TODO: Check whether this improves the queries or is just a waste of time.
QUERY_AGENT = Agent(
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
SUMMARY_AGENT = Agent(
    model=model,
    # toolsets=[mcp_server_duckduckgo],
    output_type=str,
    system_prompt=SUMMARY_INSTRUCTIONS,
    retries=5,
    instrument=True,
)

# This agent is specifically for the generation of the knowledge_gap benchmark. Not used for production.
SUMMARY_AGENT_EVALS = Agent(
    model=model,
    # toolsets=[mcp_server_duckduckgo],
    output_type=str,
    system_prompt=SUMMARY_INSTRUCTIONS_EVALS,
    retries=5,
    instrument=True,
)

REFLECTION_AGENT = Agent(
    model=model,
    output_type=Reflection,
    system_prompt=REFLECTION_INSTRUCTIONS,
    retries=5,
    instrument=True,
)

FINAL_SUMMARY_AGENT = Agent(
    model=model,
    output_type=FinalSummary,
    system_prompt=FINAL_SUMMARY_INSTRUCTIONS,
    retries=5,
    instrument=True,
)

EVALUATION_AGENT = Agent(
    model=model,
    output_type=GameResult,
    system_prompt=EVALUATION_INSTRUCTIONS,
    retries=5,
    instrument=True,
)
