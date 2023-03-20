import os

import numpy as np
import pandas as pd


def load_SCADA(year=2020):
    # TODO: Use full farm data in analysis (currently using one turbine).

    # DATA_DIRS = ['../data/Kelmarsh_SCADA_2019/', '../data/Kelmarsh/SCADA_2020']

    # FNAMES = [
    #     "Turbine_Data_Kelmarsh_1_2020-01-01_-_2021-01-01_228.csv",
        # "Turbine_Data_Kelmarsh_2_2020-01-01_-_2021-01-01_229.csv",
        # "Turbine_Data_Kelmarsh_3_2020-01-01_-_2021-01-01_230.csv",
        # "Turbine_Data_Kelmarsh_4_2020-01-01_-_2021-01-01_231.csv",
        # "Turbine_Data_Kelmarsh_5_2020-01-01_-_2021-01-01_232.csv",
        # "Turbine_Data_Kelmarsh_6_2020-01-01_-_2021-01-01_233.csv",
    # ]

    # turbines = []

    # for i, _ in enumerate(FNAMES):
    #     fname = DATA_DIRS[0] + FNAMES[i]
    #     print(f"Loading data: {FNAMES[i]}")
    #     wt = pd.read_csv(fname, header=9)
    #     turbines.append(wt)

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # wt_2019 = pd.read_csv("../data/Kelmarsh_SCADA_2019/Turbine_Data_Kelmarsh_1_2019-01-01_-_2020-01-01_228.csv", header=9)
    wt_raw = pd.read_csv(f"{curr_dir}/../data/Kelmarsh_SCADA_{year}/Turbine_Data_Kelmarsh_1_{2020}-01-01_-_{year+1}-01-01_228.csv", header=9)

    wt = wt_raw.loc[:, ["# Date and time", "Power (kW)", "Wind direction (°)", "Wind speed (m/s)"]]
    wt = wt.rename(columns={"# Date and time": "Date", "Power (kW)": "power", "Wind direction (°)": "wind_dir", "Wind speed (m/s)": "wind_speed"})
    wt["Date"] = pd.to_datetime(wt["Date"].astype("datetime64"))
    wt = wt.set_index("Date")
    wt = wt.asfreq("10min")
    wt = wt.sort_index()

    return wt


def filter_outliers(data, cut_out=None):
    wt = data.copy()

    if cut_out:
        wt = wt[wt.wind_speed <= cut_out]

    def filter_fn(group):
        q1 = group.power.quantile(.25)
        q3 = group.power.quantile(.75)
        iqr = q3 - q1
        filtered = group.query('(@q1 - 1.5 * @iqr) <= power <= (@q3 + 1.5 * @iqr)')
        return filtered
        
    wt_bins = np.arange(0, 18, .5)
    wt_groups = wt.groupby(pd.cut(wt.wind_speed, wt_bins))
    wt_filtered = wt_groups.apply(filter_fn)
    wt_filtered.index = wt_filtered.index.droplevel()
    wt_filtered = wt_filtered.sort_index()
    
    return wt_filtered


def transform(v_df, m, field="Wind speed", hr_stats=None):
    res_df = v_df.copy()

    v_scaled = res_df[field]**m

    if hr_stats:
        hr_mean, hr_std = hr_stats
    else:
        hr_group = v_scaled.groupby(v_scaled.index.hour)
        hr_mean, hr_std = hr_group.mean(), hr_group.std()
    
    res_df["v_scaled"] = v_scaled
    res_df["v"] = res_df[field]
    res_df["hr"] = res_df.index.hour
    res_df["v_scaled_std"] = res_df.apply(lambda x: (x.v_scaled - hr_mean[x.hr])/hr_std[x.hr], axis=1)
    
    return res_df, (hr_mean, hr_std)


def inv_transform(v_df, m, hr_stats):
    v_df_copy = v_df.copy()

    hr_mean, hr_std = hr_stats
    
    v_df_copy["hr"] = v_df_copy.index.hour

    inv_std =  v_df_copy.apply(lambda x: x * hr_std[x.hr] + hr_mean[x.hr], axis=1)
    inv_std = inv_std.drop(columns=["hr"])
    
    return inv_std**(1/m)