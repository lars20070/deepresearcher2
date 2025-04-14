# Deep Researcher 2

fully local web research and report writing assistant

## UML diagrams

![package diagram](./uml/packages.png "Deep Researcher 2 package structure")
<br>*Deep Researcher 2 package structure*

<br>

![class diagram](./uml/classes.png "Deep Researcher 2 class structure")
<br>*Deep Researcher 2 class structure*

``` mermaid
flowchart LR
    CoordinatorModel["coordinator model
    (Llama 3.3)"]
    SearchMCP["web search MCP
    (DuckDuckGo)"]
    ReasoningMCP["reasoning MCP
    (DeepSeek R1)"]
    CoordinatorModel --> SearchMCP
    CoordinatorModel --> ReasoningMCP
```
<br>*Deep Researcher 2 design*

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
  * Mistral Nemo `mistral-nemo` (Mistral + Nvidia)
  * Firefunction v2 `firefunction-v2` (Fireworks AI)
* search MCP
  * [DuckDuckGo MCP](https://github.com/nickclyde/duckduckgo-mcp-server)
