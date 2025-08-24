# Evaluations

The folder contains scripts for evaluating models on various benchmarks. Any benchmark data are located in the `benchmarks/` directory.

| File                 | Description |
|----------------------|-------------|
| `import_bigbench.py` | Imports the three datasets `codenames`, `dark_humor_detection`, and `rephrase` from the 2022 Google [BIG-bench](https://github.com/google/BIG-bench) benchmark to `benchmarks/`. |
| `evals.py`            | Runs evaluations on models. |