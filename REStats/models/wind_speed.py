import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from REStats.utils import transform, inv_transform

from .weibull import calc_m, get_params, fit_weibull


def backtest(v_train, v_test, idata_wb=None, steps=1):
    if idata_wb is None:
        idata_wb = fit_weibull(v_train.wind_speed)

    shape, _scale = get_params(idata_wb)
    m = calc_m(shape)

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

    fcast_rmse = mean_squared_error(v_test.v, forecasts_full["mean"], squared=False)
    fcast_rmse_rel = fcast_rmse / v_test.v.mean() * 100
    print(f"\nResults for step size: {steps}")
    print(f"Forecast RMSE: {fcast_rmse} m/s")
    print(f"Forecast RMSE (%): {fcast_rmse_rel}")
    fcast_mae = abs(v_test.v - forecasts_full["mean"]).mean()
    print(f"Forecast MAE: {fcast_mae} m/s")

    return forecasts_full, (fcast_rmse, fcast_rmse_rel, fcast_mae)


def persistence_wind_speed(v_test, steps=1):
    per = np.empty(len(v_test))

    t = 0
    while t < len(v_test) - steps:
        per[t : t + steps] = v_test[t]

        t += steps

    per = per[: len(v_test) - 1]

    per_rmse = mean_squared_error(v_test[1:], per, squared=False)
    per_rmse_rel = per_rmse / v_test[1:].mean() * 100
    per_mae = abs(v_test[1:] - per).mean()

    print(f"PER RMSE: {per_rmse} m/s")
    print(f"PER RMSE (%): {per_rmse_rel}")
    print(f"PER MAE: {per_mae} m/s")

    return per_rmse, per_rmse_rel, per_mae
