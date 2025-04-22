# Copilot Instructions for Deep Researcher 2

This document provides guidance for GitHub Copilot when generating code for the Deep Researcher 2 project - a fully local web research and report writing assistant.

## Project Overview

Deep Researcher 2 is an AI agent that autonomously conducts deep research on arbitrary topics by searching the web. The agent uses a local-first approach where possible, with cloud services as fallbacks.

## Tech Stack

- **Package Management**: uv (modern Python package manager)
- **Linting**: ruff (fast Python linter)
- **Testing**: pytest with xdist for parallel testing
- **LLM Runtime**: Ollama for running local large language models
- **Agent Framework**: PydanticAI for creating AI agents and tools
- **Language**: Python 3.11+
- **Logging**: Loguru for local logging, Logfire for cloud logging
- **Documentation**: Google docstring style
- **Type Hints**: Strict type hints throughout the codebase

## Code Organization

- `src/deepresearcher2/`: Main package code
- `tests/`: Test suite
- CI/CD: GitHub Actions for continuous integration

## Coding Standards

### General Practices

1. **Type Hints**: Always use proper type hints for function parameters and return values
2. **Docstrings**: Follow Google style docstrings for all classes and functions
3. **Logging**: Use the project's logger instead of print statements
4. **Imports**: Use explicit imports (avoid wildcard imports)
5. **Error Handling**: Use proper exception handling with specific exception types
6. **Configuration**: Use dotenv for environment variables and config management

### PydanticAI Patterns

1. **Agent Definition**: Prefer defining agents with strict result types when possible
2. **Tool Definition**: Use the `@agent.tool` decorator for adding capabilities to agents
3. **Error Handling**: Use `ModelRetry` for recoverable errors in tools
4. **Instrumentation**: Enable instrumentation with Logfire for production tracing

```python
# Example agent pattern
agent = Agent(
    model="model_name",
    system_prompt="Your system prompt here",
    output_type=YourOutputType,  # If applicable
    retries=3,
    instrument=True,
)

@agent.tool
async def your_tool(ctx: RunContext[YourDeps], param: str) -> dict:
    """Tool description here.

    Args:
        ctx: The run context
        param: Parameter description

    Returns:
        Description of return value
    """
    # Implementation here
```

### Testing Standards

1. **Test Markers**: Use appropriate markers (`paid`, `ollama`, `example`)
2. **Mocking**: Mock external dependencies for unit tests
3. **Skip Rules**: Skip tests requiring paid APIs or Ollama in CI with proper markers

```python
@pytest.mark.paid  # Skip in CI if it requires paid API
@pytest.mark.ollama  # Skip in CI if it requires local Ollama
@pytest.mark.asyncio  # For async tests
async def test_your_function() -> None:
    """Test description."""
    # Test implementation
```

## Local Environment Setup

1. Copy `.env.example` to `.env` and fill in your API keys
2. Install dependencies with `uv sync`
3. Run tests with `uv run pytest -m "not paid"`

## Web Research Implementation

When implementing web research functionality:

1. Use the Tavily API for web search when appropriate
2. Implement proper rate limiting for external APIs
3. Store search results in a structured format (Pydantic models)
4. Use async functions for network operations
5. Add appropriate retries and error handling for web requests

## Report Generation

For report generation capabilities:

1. Use Markdown as the primary format for reports
2. Consider using pypandoc for format conversion
3. Structure reports with clear sections and headers
4. Add metadata to reports (date, sources, etc.)

## Security Considerations

1. Never hardcode API keys or credentials
2. Validate and sanitize all user inputs
3. Use proper escaping for any data displayed to users
4. Follow least privilege principles when accessing APIs

## Dependency Management

1. Keep dependencies minimal and well-defined in pyproject.toml
2. Pin dependencies to specific versions for reproducibility
3. Use uv for dependency management

## File Templates

### Agent Implementation

```python
#!/usr/bin/env python3

from typing import Any
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from deepresearcher2.logger import logger

class ResultType(BaseModel):
    """Result model for the agent."""
    field1: str
    field2: int

class ResearchAgent:
    """Agent that performs research on a given topic."""
    
    def __init__(self) -> None:
        """Initialize the research agent."""
        logger.info("Initializing ResearchAgent")
        self.agent = Agent(
            model="llama3.3",
            system_prompt="You are a research assistant focused on providing accurate information.",
            output_type=OutputType,
        )
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self) -> None:
        """Register tools for the agent."""
        
        @self.agent.tool
        async def search_web(ctx: RunContext[Any], query: str) -> dict[str, Any]:
            """Search the web for information.
            
            Args:
                ctx: The run context
                query: The search query
                
            Returns:
                A dictionary containing search results
            """
            # Implementation
            logger.debug(f"Searching web for: {query}")
            # Add actual implementation
            return {"results": ["result1", "result2"]}
```

## Closing Notes

- Favor composition over inheritance
- Keep functions small and focused on a single responsibility
- Use GitHub issues for feature requests and bug reports
- Document breaking changes in commit messages