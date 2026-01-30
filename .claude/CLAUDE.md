# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepResearcher2 is a fully local web research and report writing assistant. The tool uses local LLM models via Ollama to perform web searches with DuckDuckGo, analyze search results, and automatically generate comprehensive reports on a given topic. The system employs a multi-agent architecture to manage different aspects of the research process.

## Development Commands

### Environment Setup

```bash
# Install Ollama (required for local models)
# See https://ollama.com for installation instructions

# Pull the default model
ollama pull qwen3:8b

# Create environment file from template
cp .env.example .env
# (Edit .env to set your TOPIC and any API keys)

# Install dependencies
uv sync
```

### Running the Application

```bash
# Run the research workflow with default settings
uv run research

# Generate UML diagrams
uv run uml
```

### Testing

```bash
# Run all tests
# Excluding tests marked 'paid'. See `addopts` in pyproject.toml for details.
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_utils.py

# Run specific test
uv run pytest tests/test_utils.py::test_duckduckgo_search

# Run tests in parallel
uv run pytest -n auto

# Run tests with coverage report
uv run pytest --cov=src/deepresearcher2 --cov-report=term-missing
```

### Code Quality

```bash
# Format code
uvx ruff format .

# Check and fix linting issues (ALWAYS run with --fix)
uvx ruff check --fix .

# Type checking (ALWAYS run after code changes)
uvx pyright .
```

### Before Committing

Run these checks:

```bash
# 1. Format code
uvx ruff format .

# 2. Check and fix linting issues
uvx ruff check --fix .

# 3. Type checking
uvx pyright .

# 4. Run tests
uv run pytest -n auto
```

## Architecture

DeepResearcher2 is built around a directed graph workflow where each node represents a step in the research process:

1. **WebSearch**: Generates search queries and executes web searches using configured search engines
2. **SummarizeSearchResults**: Takes search results and creates a comprehensive summary
3. **ReflectOnSearch**: Analyzes the summaries to identify knowledge gaps and decide next steps
4. **FinalizeSummary**: Compiles all summaries into a final report document

### Key Components

- **Agents**: Specialized LLM agents that handle different parts of the workflow (query generation, summarization, reflection, final report generation)
- **Models**: Pydantic models defining the data structures for state management (DeepState, WebSearchQuery, WebSearchResult, etc.)
- **Config**: Central configuration handling environment variables and runtime settings
- **Graph**: The main workflow implementation using pydantic_graph for managing the research process
- **Utils**: Helper functions for web search, content fetching, and report generation

### Search Engine Options

DeepResearcher2 supports multiple search engines:
- DuckDuckGo (default, no API key required)
- Tavily (requires API key)
- Perplexity (requires API key)

### LLM Model Options

The system can use:
- Local models via Ollama (llama3.3, qwen3:8b, qwen3:32b)
- Cloud models (OpenAI's gpt-4o, gpt-4o-mini)

## MCP Servers

This project uses Model Context Protocol (MCP) servers to extend AI capabilities. These are automatically invoked when relevant.

### Context7 Documentation Server

**When to use:**
- Looking up library documentation (e.g., "How do I use pydantic-ai streaming?")
- Checking API references for dependencies
- Finding code examples from official docs
- Verifying correct usage of third-party packages

**Examples:**
- "What's the latest pydantic-ai agent syntax?"
- "Show me httpx async client examples"
- "How do I configure pytest-asyncio?"

### GitHub Repository Server

**When to use:**
- Checking open/closed issues in this repository
- Reviewing pull requests and their status
- Reading issue comments and discussions
- Finding related issues or PRs
- Understanding project history and decisions

**Examples:**
- "What are the open issues about curiosity?"
- "Show me recent PRs related to PDF support"
- "Are there any issues about MLX integration?"
- "What's the status of issue #13?"

### Best Practices

- **Be specific:** "Check issue #15" is better than "check issues"
- **Context first:** Read codebase before checking issues
- **Combine sources:** Use Context7 for "how to use X" and GitHub for "what's our approach to X"

## Python Coding Standards

### Python Version

- **Required:** Python 3.12 (no 3.13+ features)
- **Check:** `requires-python = ">=3.12,<3.13"` in pyproject.toml
- Avoid features introduced in Python 3.13

### Code Style & Formatting

Use Ruff for formatting and linting (configured in `pyproject.toml`):

**Key rules enabled:**
- `E`, `F` - pycodestyle, pyflakes (essential errors)
- `I` - isort (import sorting)
- `UP` - pyupgrade (modern Python syntax)
- `ANN` - type annotations (required)
- `B` - bugbear (common bugs)
- `PL` - pylint rules

**Line Length:** Maximum 150 characters (configured in ruff)

### Import Order

```python
# 1. Standard library
from collections.abc import Generator
import os

# 2. Third-party packages
import pytest
from pydantic_ai import Agent

# 3. Local imports
from deepresearcher2.config import config
from deepresearcher2.models import WebSearchQuery
```

### Type Hints

Type hints are **required** for all functions (enforced by Pyright):

```python
# Good
def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    ...

async def fetch_content(url: str) -> str:
    ...
```

Use Python 3.12+ type syntax:

```python
# Good (3.12+)
def process(items: list[str]) -> dict[str, int]:
    ...

# Avoid (old style)
from typing import List, Dict
def process(items: List[str]) -> Dict[str, int]:
    ...
```

### Async/Await Patterns

Use async for:
- Network I/O (web searches, API calls)
- File I/O with async libraries
- Concurrent operations

```python
# Good - concurrent operations
async def fetch_multiple(urls: list[str]) -> list[str]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.text for r in responses]
```

### Pydantic AI Agents

All agents are async:

```python
from pydantic_ai import Agent

agent = Agent(
    model=model,
    output_type=WebSearchQuery,
    system_prompt=QUERY_INSTRUCTIONS,
    retries=5,
    instrument=True,
)

# Usage
async with agent:
    result = await agent.run(user_prompt="Generate query")
    print(result.output)
```

### Dependencies

Use `uv` (not pip or poetry):

```bash
# Install dependencies
uv sync

# Add new dependency - edit pyproject.toml manually, then:
uv sync

# Run command in venv
uv run pytest
```

### Error Handling & Logging

Use `loguru` for structured logging:

```python
from deepresearcher2.logger import logger

logger.info("Starting web search for topic: {}", topic)
logger.debug("Received {} results", len(results))

try:
    result = await fetch_content(url)
except Exception as e:
    logger.error("Failed to fetch {}: {}", url, e)
    raise
```

### Configuration

Access via centralized config:

```python
from deepresearcher2.config import config

# Good
max_loops = config.max_research_loops
model_name = config.model.value

# Avoid hardcoded values
max_loops = 5  # Bad
```

### Docstrings

Use Google-style docstrings (not Sphinx style). No backticks in docstrings:

```python
def complex_function(param1: str, param2: int = 5) -> dict[str, Any]:
    """Short one-line description.

    Longer description if needed, explaining the function's purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 5.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param2 is negative.
    """
    ...
```

### Module and Package Structure

`__init__.py` files should be kept minimal. Use for imports, `__all__`, and package-level docstrings only. Avoid defining functions, classes, or complex logic directly in `__init__.py`.

## Testing Guidelines

### Testing Principles

- **Reuse, Don't Replicate**: Tests should reuse as much functional code as possible. Avoid reimplementing application logic within a test.
- **Mock Fundamental Processes**: Mock the most fundamental external interaction (e.g., `asyncio.create_subprocess_exec` for CLI tools).
- **Cover All Failure Modes**: Cover not just the "happy path" but also all conceivable failure modes using `pytest.raises`.

### Test Structure

```python
@pytest.mark.vcr()  # For tests using VCR cassettes
@pytest.mark.asyncio  # For async tests
async def test_feature() -> None:
    """Test description."""
    # Arrange
    ...
    
    # Act
    result = await some_function()
    
    # Assert
    assert result is not None
```

### Markers

- `@pytest.mark.paid` - Requires paid API keys (skipped in CI)
- `@pytest.mark.ollama` - Requires local Ollama (skipped in CI)
- `@pytest.mark.searxng` - Requires local SearXNG (skipped in CI)
- `@pytest.mark.vcr()` - Uses VCR cassettes for HTTP recording

### VCR Cassettes

- Location: `tests/cassettes/`
- Record mode: `'none'` (playback only by default)
- Deterministic tests: set `temperature=0.0` in `MODEL_SETTINGS`

### Coverage Requirements

After writing or modifying tests, verify coverage targets from `.codecov.yaml`:

```bash
# Full project coverage
uv run pytest --cov=src/deepresearcher2 --cov-report=term-missing

# Specific module coverage
uv run pytest --cov=src/deepresearcher2/MODULE_NAME --cov-report=term-missing tests/test_MODULE_NAME.py
```
