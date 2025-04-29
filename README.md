# Deep Researcher 2

Fully local web research and report writing assistant. The models run locally in Ollama. The web searches are performed with DuckDuckGo. No API keys are required.

``` mermaid
stateDiagram-v2
    WebSearch: Web Search
    SummarizeSearchResults: Summarize Search Results
    ReflectOnSummary: Reflect on Summary
    FinalizeSummary: Finalize Summary
    [*] --> WebSearch
    WebSearch --> SummarizeSearchResults: web search result
    SummarizeSearchResults --> ReflectOnSummary: web search summary
    ReflectOnSummary --> WebSearch: reflection
    ReflectOnSummary --> FinalizeSummary
    FinalizeSummary --> [*]
```
<br>*Deep Researcher 2 workflow*
<br>
<br>

## Getting started
1. Install [Ollama](https://ollama.com) and pull a model.
   ```bash
   ollama pull llama3.3
   ```
2. Start up the workflow.
   ```bash
   uv run research
   ```

