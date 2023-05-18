import numpy as np

from REStats.circular_metrics import (
    circular_err,
    circular_mae,
    circular_rmse,
    circular_residuals,
)


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


def test_circular_err():
    obs = np.array([0, 30, 60, 90])
    pred = np.array([10, 40, 70, 100])

    rmse, rmse_rel, mae = circular_err(obs, pred, unit="deg", verbose=False)

    expected_rmse = 10
    expected_rmse_rel = 22.22222222222223
    expected_mae = 10.0

    assert np.isclose(rmse, expected_rmse, atol=1e-8)
    assert np.isclose(rmse_rel, expected_rmse_rel, atol=1e-8)
    assert np.isclose(mae, expected_mae, atol=1e-8)
