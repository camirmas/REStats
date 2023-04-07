from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from REStats.utils import transform, inv_transform
from .weibull import fit_weibull, get_params, calc_m
import pandas as pd


def backtest(v_train, v_test):
    idata_wb = fit_weibull(v_train.wind_speed)
    shape, _scale = get_params(idata_wb)
    m = calc_m(shape)

    forecasts = []

    v_train, hr_stats = transform(v_train, m)
    v_test, _ = transform(v_test, m, hr_stats=hr_stats)

    arma_mod = ARIMA(v_train.v_scaled_std, order=(2, 0, 2), trend="n")
    arma_res = arma_mod.fit()

    for t in range(len(v_test)-1):
        updated = v_test.v_scaled_std.iloc[t:t+1]
        arma_res = arma_res.append(updated, refit=False)
        fcast = arma_res.get_forecast().summary_frame()
        inv_fcast = inv_transform(fcast, m, hr_stats)
        forecasts.append(inv_fcast)
    
    forecasts_full = pd.concat(forecasts)

    fcast_rmse = mean_squared_error(v_test.v[1:], forecasts_full["mean"], squared=False)
    print("Forecast RMSE:", fcast_rmse)
    fcast_mae = abs(v_test.v[1:] - forecasts_full["mean"]).mean()
    print("Forecast MAE:", fcast_mae)

    return forecasts_full, (fcast_rmse, fcast_mae)


def persistence_wind_speed(v_test):
    per = v_test.shift(1)[1:]
    per_rmse = mean_squared_error(v_test[1:], per, squared=False)
    print("PER RMSE:", per_rmse)
    per_mae = abs(v_test[1:] - per).mean()
    print("PER MAE:", per_mae)

    return per_rmse, per_mae