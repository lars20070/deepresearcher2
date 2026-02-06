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
uv run ruff format .

# Check and fix linting issues (ALWAYS run with --fix)
uv run ruff check --fix .

# Type checking (ALWAYS run after code changes)
uv run pyright .
```

### Before Committing

Run these checks:

```bash
# 1. Format code
uv run ruff format .

# 2. Check and fix linting issues
uv run ruff check --fix .

# 3. Type checking
uv run pyright .

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

## Coding Standards

See the rules files for detailed coding standards:

- `.claude/rules/python.md` - Python coding standards, type hints, async patterns
- `.claude/rules/python-testing.md` - Testing conventions, markers, coverage requirements
