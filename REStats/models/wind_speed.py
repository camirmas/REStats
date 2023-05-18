import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from REStats.utils import calc_err, transform, inv_transform
from REStats.models import weibull


def preprocess(ws):
    """
    Preprocess the wind speed data by fitting a Weibull distribution, extracting its
    parameters and transforming the data.

    The function returns the shape and scale parameters of the Weibull distribution,
    the modulus of the Weibull distribution, the transformed wind speed data, and the
    hour-wise statistics of the transformed wind speed data.

    Args:
        ws (pandas.Series or numpy.ndarray): The wind speed data to preprocess.

    Returns:
        dict: A dictionary containing the following keys:
            - 'wb_shape': The shape parameter of the fitted Weibull distribution.
            - 'wb_scale': The scale parameter of the fitted Weibull distribution.
            - 'wb_m': The modulus of the Weibull distribution.
            - 'ws_tf': The transformed wind speed data as a DataFrame.
            - 'hr_stats': The hour-wise statistics of the transformed wind speed data.
    """
    idata_wb = weibull.fit(ws)

    shape, scale = weibull.get_params(idata_wb)
    m = weibull.calc_m(shape)
    ws_tf, hr_stats = transform(ws, m)

    return {
        "wb_shape": shape,
        "wb_scale": scale,
        "wb_m": m,
        "ws_tf": ws_tf,
        "hr_stats": hr_stats,
    }


def fit(ws_tf, order=(2, 0, 2), trend="n", **kwargs):
    """
    Fit an ARIMA model to the preprocessed and transformed wind speed data.

    The order of the ARIMA model and the trend component can be customized. Additional
    keyword arguments can be passed to the ARIMA model instantiation.

    Args:
        ws_tf (pandas.DataFrame): Preprocessed and transformed wind speed data,
            obtained from the `preprocess` function.
        order (tuple, optional): A tuple of three integers specifying the order of the
            ARIMA model. Default is (2, 0, 2).
        trend (str, optional): The trend component to include in the model. Default is
            'n', which indicates no trend.
        **kwargs: Additional keyword arguments to pass to the ARIMA model instantiation.

    Returns:
        ARIMAResults: The fitted ARIMA model.
    """
    arma_mod = ARIMA(ws_tf.v_scaled_std, order=order, trend=trend, **kwargs)
    model = arma_mod.fit()

    return model


def backtest(v_train, v_test, idata_wb=None, steps=1):
    """
    Performs a backtest of wind speed forecasting using the Weibull distribution and
    ARIMA models.

    This function fits the Weibull distribution to the training data, scales the data
    using the shape parameter of the fitted distribution, fits an ARIMA model to the
    transformed training data, and then performs one-step ahead forecasting iteratively
    over the test data.

    Args:
        v_train (pandas.DataFrame): The training data containing the wind speed.
        v_test (pandas.DataFrame): The testing data containing the wind speed.
        idata_wb (arviz.InferenceData, optional): An InferenceData object from Arviz
            based on a Weibull distribution fit. If None, the Weibull distribution is
            fitted to the training data. Defaults to None.
        steps (int, optional): The number of steps ahead to forecast. Defaults to 1.

    Returns:
        tuple: A tuple containing two elements:
            - pandas.DataFrame: A DataFrame with the full forecasts for the testing
                data.
            - tuple: A tuple containing the RMSE, Relative RMSE, and MAE between the
                observed and predicted values in the test set.
    """
    if idata_wb is None:
        idata_wb = weibull.fit(v_train.wind_speed)

    shape, _scale = weibull.get_params(idata_wb)
    m = weibull.calc_m(shape)

    forecasts = []

    v_train, hr_stats = transform(v_train, m)
    v_test, _ = transform(v_test, m, hr_stats=hr_stats)

    arma_mod = ARIMA(v_train.v_scaled_std, order=(2, 0, 2), trend="n")
    arma_res = arma_mod.fit()

    t = 0
    while t < len(v_test):
        fcast = arma_res.get_forecast(steps=steps).summary_frame()
        inv_fcast = inv_transform(fcast, m, hr_stats)
        forecasts.append(inv_fcast)
        updated = v_test.v_scaled_std.iloc[t : t + steps]
        arma_res = arma_res.append(updated, refit=False)

        t += steps

    forecasts_full = pd.concat(forecasts)
    forecasts_full = forecasts_full.iloc[: len(v_test)]

    print(f"\nResults for step size: {steps}")
    err = calc_err(v_test.v, forecasts_full["mean"])

    return forecasts_full, err


def calc_persistence(v_test, steps=1):
    """
    Calculate a persistence forecast for the wind speed and evaluate the forecast
    accuracy.

    The persistence forecast method uses the wind speed at the current time step
    as the forecast for the next time step.

    Args:
        v_test (array_like): A 1D array or list of observed wind speed values.
        steps (int, optional): The forecast horizon in number of steps. Default is 1.

    Returns:
        tuple: A tuple containing the persistence forecast and a tuple of forecast
        accuracy metrics (RMSE, relative RMSE, MAE).
    """
    per = np.empty(len(v_test))

    t = 1
    while t <= len(v_test) - steps:
        per[t : t + steps] = v_test[t - 1]

        t += steps

    per = per[1 : len(v_test)]

    err = calc_err(v_test[1:], per)

    return per, err
