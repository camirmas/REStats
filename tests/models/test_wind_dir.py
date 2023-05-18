from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults

from REStats.models.wind_dir import fit, predict, backtest, calc_persistence


def test_calc_persistence():
    wind_dir = np.array([10, 20, 30, 40, 50])
    steps = 2

    expected_per = np.array([10, 10, 30, 30])  # Expected persistence forecast

    per, _err = calc_persistence(wind_dir, steps)

    np.testing.assert_array_equal(per, expected_per)


@patch("statsmodels.tsa.arima.model.ARIMA.fit")
def test_fit(mock_fit):
    # Create a mock ARIMAResults object to be returned by fit()
    mock_results = MagicMock(spec=ARIMAResults)
    mock_fit.return_value = mock_results

    wind_dir = np.array([10, 20, 30, 40, 50])
    sin_model, cos_model = fit(wind_dir)

    assert isinstance(sin_model, ARIMAResults)
    assert isinstance(cos_model, ARIMAResults)
    assert mock_fit.call_count == 2  # Ensure that the ARIMA model is fitted twice


@patch(
    "statsmodels.tsa.arima.model.ARIMAResults.predict",
    return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
)
@patch(
    "statsmodels.tsa.arima.model.ARIMAResults.get_prediction",
    return_value=MagicMock(
        conf_int=lambda: np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    ),
)
def test_predict(mock_get_prediction, mock_predict):
    wind_dir = np.array([10, 20, 30, 40, 50])
    sin_model = cos_model = ARIMAResults

    result_df = predict(sin_model, cos_model, wind_dir)

    assert len(result_df) == len(wind_dir)
    assert set(result_df.columns) == {
        "wind_dir_pred",
        "wind_dir_obs",
        "lower_bound",
        "upper_bound",
    }
    assert (
        mock_predict.call_count == 2
    )  # Ensure that prediction is made for both models
    assert (
        mock_get_prediction.call_count == 2
    )  # Ensure that get_prediction is called for both models


@patch("statsmodels.tsa.arima.model.ARIMAResults.get_forecast")
@patch("statsmodels.tsa.arima.model.ARIMAResults.append")
def test_backtest(mock_append, mock_get_forecast):
    # Mocking ARIMAResults.append method to return the same instance
    mock_append.return_value = MagicMock(spec=ARIMAResults)

    # Mocking ARIMAResults.get_forecast method
    mock_forecast = MagicMock()
    mock_forecast.predicted_mean = np.array([0.1])
    mock_forecast.conf_int.return_value = {
        "lower wind_dir": np.array([0.1]),
        "upper wind_dir": np.array([0.1]),
    }
    mock_get_forecast.return_value = mock_forecast

    wd_train = pd.Series([10, 20, 30, 40, 50])
    wd_test = pd.Series([60, 70, 80, 90, 100])

    forecasts, err = backtest(wd_train, wd_test)

    assert isinstance(forecasts, pd.DataFrame)
    assert isinstance(err, tuple)
    assert len(err) == 3
    assert set(forecasts.columns) == {"mean", "lower_ci", "upper_ci"}
