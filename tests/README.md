## Examples

The repository contains many code examples which are unrelated to the Deep Researcher 2 functionality. They can be found in `tests/test_example.py` and `src/deeprearcher2/examples.py`. The later can be executed as scripts. For example:
```bash
uv run chat
uv run mcpserver
```
<br>
<br>
<br>
<br>

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
