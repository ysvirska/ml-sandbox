"""Fit and evaluate a one-feature linear regression using the standard library."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from random import Random
from statistics import fmean


@dataclass(frozen=True)
class LinearModel:
    slope: float
    intercept: float

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept


def fit_line(points: Sequence[tuple[float, float]]) -> LinearModel:
    """Fit y = slope * x + intercept by ordinary least squares.

    At least two finite observations and non-zero input variance are required.
    Centering the observations avoids subtracting large uncentered sums.
    """
    if len(points) < 2:
        raise ValueError("At least two training observations are required.")
    if any(not isfinite(x) or not isfinite(y) for x, y in points):
        raise ValueError("Training values must be finite.")

    x_mean = fmean(x for x, _ in points)
    y_mean = fmean(y for _, y in points)
    variance = fsum((x - x_mean) ** 2 for x, _ in points)
    if variance == 0:
        raise ValueError("Training inputs must have non-zero variance.")
    covariance = fsum((x - x_mean) * (y - y_mean) for x, y in points)
    slope = covariance / variance
    return LinearModel(slope, y_mean - slope * x_mean)


def mean_squared_error(
    actual: Sequence[float], predicted: Sequence[float]
) -> float:
    """Return the average squared error for matching, finite observations."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Expected non-empty sequences of equal length.")
    if any(not isfinite(value) for values in (actual, predicted) for value in values):
        raise ValueError("Metric inputs must be finite.")
    return fsum(
        (target - estimate) ** 2 for target, estimate in zip(actual, predicted)
    ) / len(actual)


def make_dataset(size: int = 50, seed: int = 7) -> list[tuple[float, float]]:
    """Generate shuffled observations of y = 2*x + 1 with bounded noise."""
    if type(size) is not int or size < 3:
        raise ValueError("Dataset size must be an integer of at least three.")
    generator = Random(seed)
    points = [
        (float(x), 2.0 * x + 1.0 + generator.uniform(-1.0, 1.0))
        for x in range(size)
    ]
    generator.shuffle(points)
    return points


def main() -> None:
    points = make_dataset()
    split = int(len(points) * 0.8)
    training, test = points[:split], points[split:]
    model = fit_line(training)
    actual = [y for _, y in test]
    predicted = [model.predict(x) for x, _ in test]

    print(f"Training samples: {len(training)}")
    print(f"Test samples: {len(test)}")
    print(f"Slope: {model.slope:.4f}")
    print(f"Intercept: {model.intercept:.4f}")
    print(f"Test MSE: {mean_squared_error(actual, predicted):.4f}")


if __name__ == "__main__":
    main()
