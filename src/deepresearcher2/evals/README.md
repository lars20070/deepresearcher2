# Evaluations

The folder contains scripts for evaluating models on various benchmarks. Any benchmark data are located in the `benchmarks/` directory.

| File                 | Description |
|----------------------|-------------|
| `import_bigbench.py` | Imports the three datasets `codenames`, `dark_humor_detection`, and `rephrase` from the 2022 Google [BIG-bench](https://github.com/google/BIG-bench) benchmark to `benchmarks/`. |
| `generate.py`        | Generates a set of *search summaries* for the *knowledge gap* benchmark. |
| `evals.py`           | Runs evaluations on models. |

Run the following commands to set up and run the evaluations:

```bash
uv run import    # Import BIG-bench datasets
uv run generate  # Generate search summaries dataset
uv run evals     # Run evaluations on models
```