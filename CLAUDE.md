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
# Run all tests (excluding tests marked 'paid')
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_utils.py

# Run specific test
pytest tests/test_utils.py::test_function_name

# Run tests in parallel
pytest -xvs -n auto

# Run tests with coverage report
pytest --cov
```

### Code Quality

```bash
# Run linting with ruff
ruff check .

# Format code with ruff
ruff format .
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

## Testing Guidelines

- Tests are organized with pytest markers:
  - `paid`: Tests requiring paid API keys (skipped by default)
  - `ollama`: Tests requiring a local Ollama instance 
  - `example`: Examples not testing core functionality

- When adding new features, create corresponding tests in the appropriate test file
- Use fixtures in conftest.py for common testing scenarios