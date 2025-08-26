# Benchmarks

The folder contains benchmark data for evaluating models on specific tasks. Each benchmark is defined in a separate JSON file.

| Benchmark                 | Type | Source | Description |
|----------------------|-------------|-------------|-------------|
| `codenames` | Static input. Static output. | Google [BIG-bench](https://github.com/google/BIG-bench) | A word association game where players try to pick the right words from a given set. |
| `dark_humor_detection` | Static input. Static output. | Google [BIG-bench](https://github.com/google/BIG-bench) | A task for detecting dark humor in text. |
| `rephrase` | Static input. Static output. | Google [BIG-bench](https://github.com/google/BIG-bench) | A task for rephrasing sentences while retaining their meaning. |
| `knowledge_gap` | Static input. Dynamic scoring. | `generate.py` script | Set of *search summaries* for benchmarking *knowledge gap* generation |