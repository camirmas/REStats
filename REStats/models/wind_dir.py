import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from REStats.circular_metrics import circular_mae, circular_mean, circular_rmse


def calc_persistence(wind_dir, steps=1):
    per = np.empty(len(wind_dir))

    t = 1
    while t < len(wind_dir) - steps:
        per[t : t + steps] = wind_dir[t - 1]

        t += steps

    per = per[1 : len(wind_dir)]

    per_rmse = circular_rmse(wind_dir[1:], per)
    per_rmse_rel = per_rmse / circular_mean(wind_dir[1:]) * 100
    per_mae = circular_mae(wind_dir[1:], per)

    print(f"PER RMSE: {per_rmse} deg")
    print(f"PER RMSE (%): {per_rmse_rel}")
    print(f"PER MAE: {per_mae} deg")

    return per, (per_rmse, per_rmse_rel, per_mae)


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

    fcast_rmse = circular_rmse(wd_test, forecasts_full["mean"])
    fcast_rmse_rel = fcast_rmse / circular_mean(wd_test) * 100
    print(f"\nResults for step size: {steps}")
    print(f"Forecast RMSE: {fcast_rmse} deg")
    print(f"Forecast RMSE (%): {fcast_rmse_rel}")
    fcast_mae = circular_mae(wd_test, forecasts_full["mean"])
    print(f"Forecast MAE: {fcast_mae} deg")

    return forecasts_full, (fcast_rmse, fcast_rmse_rel, fcast_mae)
