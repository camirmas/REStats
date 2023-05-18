import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from REStats.circular_metrics import circular_err


def calc_persistence(wind_dir, steps=1):
    """
    Calculate a persistence forecast for wind direction. The persistence model assumes
    that the conditions at the time of forecasting will stay the same for the period of
    the forecast.

    Args:
        wind_dir (np.ndarray): An array of wind direction measurements.
        steps (int, optional): Number of steps forward for the forecast. Default is 1.

    Returns:
        tuple: A tuple where the first element is an array with the persistence forecast
               and the second element is the error calculated using circular_err
               function.
    """
    per = np.empty(len(wind_dir))

    t = 1
    while t <= len(wind_dir) - steps:
        per[t : t + steps] = wind_dir[t - 1]
        t += steps

    per = per[1 : len(wind_dir)]
    err = circular_err(wind_dir[1:], per)

    return per, err


def fit(wind_dir_data):
    """
    Fit ARMA models to the sine and cosine components of wind direction data.

    Args:
        wind_dir_data (array_like): A 1D array or list of observed wind direction values
            in degrees.

    Returns:
        tuple: A tuple containing two ARIMAResults objects, one for the sine model and
            one for the cosine model.
    """
    # Convert the wind direction data to radians
    wind_direction_radians = np.radians(wind_dir_data)

    # Decompose the wind direction data into two linear components
    sin_component = np.sin(wind_direction_radians)
    cos_component = np.cos(wind_direction_radians)

    # Fit ARMA models for each component
    sin_model = ARIMA(sin_component, order=(3, 0, 1)).fit()
    cos_model = ARIMA(cos_component, order=(3, 0, 1)).fit()

    return sin_model, cos_model


def predict(sin_model, cos_model, wind_dir_data):
    """
    Forecast wind direction from trained sine and cosine ARMA models, and compute the
    confidence intervals.

    Args:
        sin_model (ARIMAResults): The trained sine ARMA model.
        cos_model (ARIMAResults): The trained cosine ARMA model.
        wind_dir_data (array_like): A 1D array or list of observed wind direction
            values in degrees.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'wind_dir_pred' (predicted wind
            direction in degrees), 'wind_dir_obs' (observed wind direction in degrees),
            'lower_bound' (lower confidence interval in degrees), and 'upper_bound'
            (upper confidence interval in degrees).
    """
    # Predict the components
    sin_pred = sin_model.predict()
    cos_pred = cos_model.predict()

    # Convert the predictions back to radians and degrees
    dir_rad = np.arctan2(sin_pred, cos_pred)
    dir_deg = np.mod(np.degrees(dir_rad), 360)

    # Compute the CI intervals for the predictions in radians
    sin_pred_ci = sin_model.get_prediction().conf_int()
    cos_pred_ci = cos_model.get_prediction().conf_int()

    # Combine the CI intervals back to radians
    lower_bound_rad = np.arctan2(sin_pred_ci[:, 0], cos_pred_ci[:, 1])
    upper_bound_rad = np.arctan2(sin_pred_ci[:, 1], cos_pred_ci[:, 0])

    # Convert the CI intervals back to degrees
    lower_bound_deg = np.degrees(lower_bound_rad)
    lower_bound_deg = np.mod(lower_bound_deg, 360)

    upper_bound_deg = np.degrees(upper_bound_rad)
    upper_bound_deg = np.mod(upper_bound_deg, 360)

    res = pd.DataFrame(
        {
            "wind_dir_pred": dir_deg,
            "wind_dir_obs": wind_dir_data,
            "lower_bound": lower_bound_deg,
            "upper_bound": upper_bound_deg,
        }
    )

    return res


def backtest(wd_train, wd_test, steps=1):
    """
    Perform a backtest of wind direction forecasting, by training ARMA models on the
    training data and sequentially predicting the wind direction on the test data.

    Args:
        wd_train (pd.Series): The wind direction training data in degrees.
        wd_test (pd.Series): The wind direction test data in degrees.
        steps (int, optional): The number of steps to forecast at each iteration.
            Default is 1.

    Returns:
        tuple: A tuple containing two elements:
            - pd.DataFrame: A DataFrame with columns 'mean' (predicted wind direction),
                            'lower_ci' (lower bound of confidence interval), and
                            'upper_ci' (upper bound of confidence interval).
            - tuple: A tuple containing RMSE, relative RMSE, and MAE calculated by
                     comparing the forecasted values with the actual test data.
    """
    wd_test_radians = np.radians(wd_test)
    sin_model, cos_model = fit(wd_train)
    forecasts = []

    t = 0
    while t < len(wd_test):
        res = {}

        sin_fcast = sin_model.get_forecast(steps=steps)
        cos_fcast = cos_model.get_forecast(steps=steps)

        # Convert the forecast back to radians and degrees
        dir_rad = np.arctan2(sin_fcast.predicted_mean, cos_fcast.predicted_mean)
        dir_deg = np.mod(np.degrees(dir_rad), 360)

        res["mean"] = dir_deg

        # Compute the CI intervals for the forecast in radians
        sin_fcast_ci = sin_fcast.conf_int()
        cos_fcast_ci = cos_fcast.conf_int()

        # Combine the CI intervals back to radians
        lower_bound_rad = np.arctan2(
            sin_fcast_ci["lower wind_dir"], cos_fcast_ci["upper wind_dir"]
        )
        upper_bound_rad = np.arctan2(
            sin_fcast_ci["upper wind_dir"], cos_fcast_ci["lower wind_dir"]
        )

        # Convert the CI intervals back to degrees
        lower_bound_deg = np.degrees(lower_bound_rad)
        lower_bound_deg = np.mod(lower_bound_deg, 360)

        upper_bound_deg = np.degrees(upper_bound_rad)
        upper_bound_deg = np.mod(upper_bound_deg, 360)

        res["lower_ci"] = lower_bound_deg
        res["upper_ci"] = upper_bound_deg

        res_df = pd.DataFrame(res)

        forecasts.append(res_df)
        updated = wd_test_radians.iloc[t : t + steps]

        sin_model = sin_model.append(np.sin(updated), refit=False)
        cos_model = cos_model.append(np.cos(updated), refit=False)

        t += steps

    forecasts_full = pd.concat(forecasts)
    forecasts_full = forecasts_full.iloc[: len(wd_test)]

    print(f"\nResults for step size: {steps}")
    err = circular_err(wd_test, forecasts_full["mean"])

    return forecasts_full, err
