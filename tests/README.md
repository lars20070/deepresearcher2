``` mermaid
flowchart LR
    Start(["start"])
    NodeA["Node A"]
    NodeB["Node B"]
    NodeC["Node C"]
    End(["end"])
    Start --> NodeA
    NodeA --> NodeB
    NodeB --> NodeC
    NodeB --> End
    NodeC --> End
```
<br>*flow chart for example `test_pydantic_graph`*
<br>
<br>
<br>
<br>

``` mermaid
flowchart LR
    Start(["start"])
    WriteEmail
    Feedback
    End(["end"])
    Start --> WriteEmail
    WriteEmail --> Feedback
    Feedback --> WriteEmail
    Feedback --> End
```
<br>*flow chart for example `test_email`*