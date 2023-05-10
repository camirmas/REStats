import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def calc_persistence(df, steps=1):
    """
    Predicts wind direction using the persistence method.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex, containing "wind_dir" column.
        steps (int): Number of steps ahead to forecast.

    Returns:
        pd.Series: Predicted wind direction using the persistence method.
    """
    return df["wind_dir"].shift(steps)


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
    sin_forecast = sin_model.predict()
    cos_forecast = cos_model.predict()

    # Convert the predictions back to radians and degrees
    dir_rad = np.arctan2(sin_forecast, cos_forecast)
    dir_deg = np.mod(np.degrees(dir_rad), 360)

    # Compute the CI intervals for the predictions in radians
    sin_forecast_ci = sin_model.get_prediction().conf_int()
    cos_forecast_ci = cos_model.get_prediction().conf_int()

    # Combine the CI intervals back to radians
    lower_bound_rad = np.arctan2(sin_forecast_ci[:, 0], cos_forecast_ci[:, 1])
    upper_bound_rad = np.arctan2(sin_forecast_ci[:, 1], cos_forecast_ci[:, 0])

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


def backtest():
    pass
