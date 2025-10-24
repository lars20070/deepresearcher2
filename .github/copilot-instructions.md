# Project Overview

DeepResearcher2 is a fully local web research and report writing assistant that prioritizes user privacy. It leverages PydanticAI as the core agent framework to orchestrate a multi-step research workflow. The system runs AI models locally using Ollama and performs web searches through SearXNG, ensuring no data leaves the user's environment by default. The application supports both local and cloud-based configurations, allowing users to choose between complete privacy or enhanced performance with API-based services.

The research workflow follows a state-machine pattern with four main stages:
1. **Web Search**: Initiates targeted searches based on the research topic
2. **Summarize Search Results**: Processes and condenses the retrieved information
3. **Reflect on Summary**: Evaluates the quality and completeness of gathered information
4. **Finalize Summary**: Produces the final research report with citations

This project uses PydanticAI's agent capabilities for structured responses, dependency injection, and tool integration. The UV package manager handles Python dependencies and virtual environment management.

## Folder Structure

```
deepresearcher2/
├── benchmarks
│   ├── codenames
│   │   ├── task_schema.json
│   │   └── task.json
│   ├── dark_humor_detection
│   │   ├── task_schema.json
│   │   └── task.json
│   ├── README.md
│   └── rephrase
│       ├── task_schema.json
│       └── task.json
├── deepresearcher2.log
├── dist
├── LICENSE
├── pyproject.toml
├── README.md
├── reports
├── src
│   └── deepresearcher2
│       ├── __init__.py
│       ├── agents.py
│       ├── cli.py
│       ├── config.py
│       ├── evals
│       │   ├── evals.py
│       │   ├── import_bigbench.py
│       │   └── README.md
│       ├── examples.py
│       ├── graph.py
│       ├── logger.py
│       ├── models.py
│       ├── prompts.py
│       ├── py.typed
│       └── utils.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── data
│   │   ├── state_1.json
│   │   ├── state_2.json
│   │   └── state_3.json
│   ├── README.md
│   ├── reports
│   │   ├── petrichor.md
│   │   └── petrichor.pdf
│   ├── test_config.py
│   ├── test_example.py
│   ├── test_graph.py
│   ├── test_logger.py
│   └── test_utils.py
├── uml
│   ├── classes.dot
│   ├── classes.png
│   ├── packages.dot
│   ├── packages.png
│   └── README.md
└── uv.lock
```

## Libraries and Frameworks

### Core Dependencies
- **pydantic-ai**: Main agent framework for building the research assistant
- **pydantic**: Data validation and settings management using Python type annotations
- **ollama**: Python client for interacting with local Ollama models
- **httpx**: Async HTTP client for web requests and API calls
- **aiohttp**: Alternative async HTTP library for specific integrations

### Search and Web Tools
- **searxng**: Integration library for SearXNG meta-search engine
- **beautifulsoup4**: HTML parsing for web content extraction
- **markdownify**: Convert HTML content to Markdown format
- **lxml**: XML and HTML processing with XPath support

### Development Tools
- **python-dotenv**: Load environment variables from .env files
- **rich**: Terminal formatting for better CLI output
- **loguru**: Advanced logging with structured output
- **pytest**: Testing framework
- **pytest-asyncio**: Async test support for PydanticAI agents
- **ruff**: Fast Python linter and formatter

### Environment Management
- **uv**: Modern Python package and project manager
- **docker**: Container runtime for SearXNG deployment

## Coding Standards

### Python Style Guidelines
- Follow PEP 8 with a line length limit of 100 characters
- Use Python 3.11+ features including type hints for all function signatures
- Prefer async/await patterns for I/O operations and agent interactions
- Use Ruff for automatic code formatting and linting

### PydanticAI Agent Patterns
- Define agents with explicit `deps_type` and `output_type` for type safety
- Use dependency injection for passing configuration and services to agents
- Implement structured outputs using Pydantic models with Field descriptions
- Handle tool registration through PydanticAI's tool decorator pattern
- Use RunContext for accessing dependencies within agent functions

### Code Organization
- Keep agent definitions focused on a single responsibility
- Separate tool implementations from agent logic
- Use Pydantic models for all data structures and API contracts
- Implement proper error handling with custom exception classes
- Log all agent decisions and tool calls for debugging

### Testing Requirements
- Write unit tests for all agent tools and utility functions
- Use pytest fixtures for agent setup and teardown
- Mock external services (Ollama, SearXNG) in tests using pytest-mock
- Or use VCR recordings with pytest-recording
- Test both success and failure paths for agent workflows
- Maintain test coverage above 80% for core functionality

### Environment Configuration
- Never commit `.env` files or API keys to version control
- Use environment variables for all configuration values
- Provide sensible defaults for optional settings
- Document all required environment variables in `.env.example`
- Support both local (Ollama, SearXNG) and cloud (OpenAI, Anthropic) configurations

### Documentation Standards
- Use docstrings for all classes, methods, and functions
- Include type hints in function signatures
- Document agent system prompts and tool descriptions clearly
- Maintain up-to-date README with setup instructions
- Include examples of agent usage and output formats

### Error Handling
- Implement retry logic for network requests and LLM calls
- Use PydanticAI's built-in error handling for agent failures
- Provide meaningful error messages for user-facing issues
- Log errors with full context for debugging
- Gracefully degrade when optional services are unavailable

### Security Considerations
- Default to local-only operation without external API calls
- Validate and sanitize all user inputs
- Use HTTPS for any external API connections
- Store sensitive configuration in environment variables only
- Implement rate limiting for external service calls

### Performance Optimization
- Use async operations for concurrent web searches
- Implement caching for repeated searches within a session
- Stream LLM responses when possible for better UX
- Batch process search results for efficient summarization
- Monitor and log agent execution times for optimization