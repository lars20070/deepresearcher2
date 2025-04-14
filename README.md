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