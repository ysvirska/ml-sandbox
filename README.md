# ml-sandbox

Small, reproducible machine learning experiments in Python.

The aim is to understand how basic models work: prepare data, implement the
algorithm, evaluate predictions, and explain the results. Experiments will start
with the standard library so the calculations remain easy to inspect.

## Scope

- Implement basic models and evaluation metrics.
- Use small synthetic datasets before introducing external data.
- Keep training and evaluation data separate.
- Explain assumptions, results, and limitations alongside the code.

## Planned layout

```text
experiments/    Runnable Python experiments
tests/          Tests for models and metrics
docs/           Task descriptions and notes
```

## Development approach

Build the sandbox incrementally, with one focused change per commit. Each
experiment should include a short explanation, runnable code, and tests for
important behavior.

## Status

The project currently contains this overview. The directory structure and first
experiment will be added next. Python 3.10 or newer is the intended baseline.
