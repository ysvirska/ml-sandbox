# ml-sandbox

Small, reproducible machine learning experiments in Python.

The aim is to understand how basic models work: prepare data, implement the
algorithm, evaluate predictions, and explain the results. Experiments start with
the standard library so the calculations remain easy to inspect.

## Scope

- Implement basic models and evaluation metrics.
- Use small synthetic datasets before introducing external data.
- Keep training and evaluation data separate.
- Explain assumptions, results, and limitations alongside the code.

## Layout

```text
experiments/    Runnable Python experiments
tests/          Tests for models and metrics
docs/           Task descriptions and notes
```

## First experiment

[Linear regression from scratch](docs/linear_regression.md) fits a line to noisy
synthetic observations and reports mean squared error on a separate test set.

Python 3.10 or newer is required. No third-party dependencies are needed.
Run from the repository root:

```bash
python experiments/linear_regression.py
python -m unittest discover -s tests -v
```

## Development approach

Build the sandbox incrementally, with one focused change per commit. Each
experiment should include a short explanation, runnable code, and tests for
important behavior. API keys, local environment files, and private key files
must stay out of version control.
