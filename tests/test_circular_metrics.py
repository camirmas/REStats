import numpy as np

from REStats.circular_metrics import circular_mae, circular_rmse, circular_residuals


def test_circular_rmse():
    true_values = np.array([0, 90, 180, 270])
    predicted_values = np.array([0, 90, 180, 270])
    expected_rmse = 0
    result = circular_rmse(true_values, predicted_values)
    assert np.isclose(result, expected_rmse, rtol=1e-5)


def test_circular_mae():
    true_values = np.array([0, 90, 180, 270])
    predicted_values = np.array([0, 90, 180, 270])
    expected_mae = 0
    result = circular_mae(true_values, predicted_values)
    assert np.isclose(result, expected_mae, rtol=1e-5)


def test_circular_residuals():
    true_degrees = np.array([10, 350, 5])
    predicted_degrees = np.array([5, 355, 10])
    expected_residuals = np.array([5, -5, -5])

    actual_residuals = circular_residuals(true_degrees, predicted_degrees)
    np.testing.assert_almost_equal(actual_residuals, expected_residuals, decimal=5)
