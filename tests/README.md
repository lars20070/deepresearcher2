## Examples

``` mermaid
stateDiagram-v2
    direction LR
    NodeA: Node A
    NodeB: Node B
    NodeC: Node C
    [*] --> NodeA 
    NodeA --> NodeB
    NodeB --> NodeC
    NodeB --> [*]
    NodeC --> [*]
```
<br>*state diagram for example `test_pydantic_graph`*
<br>
<br>
<br>
<br>

``` mermaid
stateDiagram-v2
    direction LR
    [*] --> WriteEmail
    WriteEmail --> Feedback
    Feedback --> WriteEmail
    Feedback --> [*]
```
<br>*state diagram for example `test_email`*
