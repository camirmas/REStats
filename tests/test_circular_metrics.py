import numpy as np
from REStats.circular_metrics import circular_mean, circular_std, circular_rmse, circular_mae


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
