"""Check the regression mathematics and invalid-input behavior."""

import unittest

from experiments.linear_regression import fit_line, make_dataset, mean_squared_error


class LinearRegressionTests(unittest.TestCase):
    def test_fit_exact_line(self):
        model = fit_line([(0, 1), (1, 3), (2, 5), (3, 7)])
        self.assertAlmostEqual(model.slope, 2)
        self.assertAlmostEqual(model.intercept, 1)
        self.assertAlmostEqual(model.predict(4), 9)

    def test_fit_negative_slope(self):
        model = fit_line([(-2, 7), (0, 3), (2, -1)])
        self.assertAlmostEqual(model.slope, -2)
        self.assertAlmostEqual(model.intercept, 3)

    def test_centered_fit_handles_large_input_offset(self):
        points = [(1_000_000_000 + x, 3 * (1_000_000_000 + x) - 7) for x in range(5)]
        model = fit_line(points)
        self.assertAlmostEqual(model.slope, 3)
        self.assertAlmostEqual(model.intercept, -7)

    def test_reject_constant_training_inputs(self):
        with self.assertRaisesRegex(ValueError, "variance"):
            fit_line([(2, 3), (2, 5)])

    def test_reject_insufficient_training_observations(self):
        for points in ([], [(1, 2)]):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    fit_line(points)

    def test_reject_non_finite_training_values(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for point in ((value, 1), (1, value)):
                with self.subTest(point=point):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        fit_line([(0, 1), point])

    def test_mean_squared_error_known_result(self):
        self.assertAlmostEqual(mean_squared_error([1, 2, 3], [1, 2, 5]), 4 / 3)

    def test_mean_squared_error_zero_for_exact_predictions(self):
        self.assertEqual(mean_squared_error([-1, 0, 2], [-1, 0, 2]), 0)

    def test_reject_empty_or_mismatched_metric_inputs(self):
        for actual, predicted in (([], []), ([1], []), ([1, 2], [1])):
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    mean_squared_error(actual, predicted)

    def test_reject_non_finite_metric_inputs(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for actual, predicted in (([value], [1]), ([1], [value])):
                with self.subTest(actual=actual, predicted=predicted):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        mean_squared_error(actual, predicted)

    def test_dataset_is_reproducible_and_has_unique_inputs(self):
        points = make_dataset(seed=7)
        self.assertEqual(points, make_dataset(seed=7))
        self.assertNotEqual(points, make_dataset(seed=8))
        self.assertEqual(len({x for x, _ in points}), 50)
        training, test = points[:40], points[40:]
        self.assertTrue(set(training).isdisjoint(test))
        for x, y in points:
            self.assertLessEqual(abs(y - (2 * x + 1)), 1)

    def test_reject_invalid_dataset_size(self):
        for size in (0, 1, 2, True, 3.0, None):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    make_dataset(size=size)


if __name__ == "__main__":
    unittest.main()
