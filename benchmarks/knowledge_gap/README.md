## Knowledge Gap benchmark

One of the most important steps in the Deep Research workflow is the generation of new queries based on previous search results. Or more precisely, the step from a summary of a search result to a gap in knowledge i.e. a new direction in the deep research process.

``` mermaid
stateDiagram-v2
    direction LR
    Ellipsis1: ...
    Result: Search Result
    Summary: Summary of Search Result
    KnowledgeGap: Knowledge Gap
    Query: Search Query
    Ellipsis2: ...
    Ellipsis1 --> Result
    Result --> Summary
    Summary --> KnowledgeGap
    KnowledgeGap --> Query
    Query --> Ellipsis2
```
The benchmark compiles a varied set of *search summaries*. 