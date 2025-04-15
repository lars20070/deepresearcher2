# Deep Researcher 2

fully local web research and report writing assistant

## UML diagrams

![package diagram](./uml/packages.png "Deep Researcher 2 package structure")
<br>*Deep Researcher 2 package structure*

<br>

![class diagram](./uml/classes.png "Deep Researcher 2 class structure")
<br>*Deep Researcher 2 class structure*

``` mermaid
stateDiagram-v2
    direction LR
    WebSearch: Web Search
    SummarizeSearchResults: Summarize Search Results
    ReflectOnSummary: Reflect on Summary
    FinalizeSummary: Finalize Summary
    [*] --> WebSearch
    WebSearch --> SummarizeSearchResults
    SummarizeSearchResults --> ReflectOnSummary
    ReflectOnSummary --> WebSearch
    ReflectOnSummary --> FinalizeSummary
    FinalizeSummary --> [*]
```
<br>*Deep Researcher 2 design*
<br>
<br>
<br>
<br>

``` mermaid
flowchart LR
    CoordinatorModel["coordinator model<br>Llama 3.3"]
    SearchMCP["web search MCP<br>DuckDuckGo"]
    ReasoningMCP["reasoning MCP<br>DeepSeek R1"]
    CoordinatorModel --> SearchMCP
    CoordinatorModel --> ReasoningMCP
```
<br>*MCPs in Deep Researcher 2*
<br>
<br>
<br>
<br>

``` mermaid
classDiagram
    DeepState: string topic
    DeepState: string search_query
    DeepState: list search_results
    DeepState: int loop_count
    DeepState: string running_summary
```
<br>*state class*

## Models and MCPs

* coordinator models for tool/MCP use
  * Llama 3.3 `llama3.3` (Meta)
  * Firefunction v2 `firefunction-v2` (Fireworks AI)
  * Mistral Nemo `mistral-nemo` (Mistral + Nvidia)
    * fails to make proper use of the Python execution MCP
* search MCP
  * [DuckDuckGo MCP](https://github.com/nickclyde/duckduckgo-mcp-server)
    * `search` tool
    * `fetch_content` tool
