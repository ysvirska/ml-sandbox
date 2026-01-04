# First task: linear regression from scratch

Fit a line to a small synthetic dataset, then evaluate it on held-out observations.
The experiment uses only the Python standard library.

## Data

Generate 50 observations from `y = 2*x + 1 + noise`, where noise is uniformly sampled
between -1 and 1. A local random generator with seed 7 makes the run reproducible.
Shuffle the observations and use 40 for training and 10 for testing.

The test observations must not be used to fit the model. There is no external
dataset to download or store.

## Model

For training observations `(x, y)`, calculate the means `x_mean` and `y_mean`:

```text
slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)**2)
intercept = y_mean - slope * x_mean
prediction = slope * x + intercept
```

At least two training observations and non-zero input variance are required.
The fitted slope and intercept should be close to 2 and 1, rather than exactly
equal, because the training targets contain noise.

## Evaluation

Report the training and test sample counts, fitted parameters, and mean squared
error on the test observations:

```text
MSE = sum((actual - predicted)**2) / number_of_observations
```

Run from the repository root with Python 3.10 or newer:

```bash
python experiments/linear_regression.py
python -m unittest discover -s tests -v
```

Tests cover exact lines, negative slopes, a large input offset, degenerate and
non-finite data, known metric results, and reproducibility.

This is an educational one-feature baseline. It does not provide multivariate
regression, regularization, or uncertainty estimates. Normal floating-point
precision limits apply, and these synthetic results do not establish performance
on real data.
