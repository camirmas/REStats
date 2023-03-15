import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


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


def filter_outliers(data, regions=(3, 12), n_neighbors=100, outlier_threshold=.0015, **kwargs):
    p_rated = 2050 # kW
    region_2, region_3 = regions
    wt = data.dropna()
    wt_r1 = wt[wt["Wind speed"] < region_2]
    wt_r2 = wt[(wt["Wind speed"] >= region_2) & (wt["Wind speed"] < region_3)]
    wt_r3 = wt[wt["Wind speed"] >= region_3]

    # region II filter
    p_r2 = wt_r2[["Power"]]
    X_r2 = (p_r2 - p_r2.mean())/p_r2.std()
    knn_r2 = NearestNeighbors(n_neighbors=100, **kwargs)
    knn_r2.fit(X_r2)
    distances_r2, _ = knn_r2.kneighbors(X_r2)

    avg_dist_r2 = distances_r2.mean(axis=1)
    [outliers_r2] = np.where(avg_dist_r2 > .05)
    keep_idxs_r2 = [i for i, _ in enumerate(wt_r2.index) if i not in outliers_r2]
    wt_filtered_r2 = wt_r2.iloc[keep_idxs_r2]
    
    # Manually remove outliers based on percentage of rated power
    wt_filtered_r2 = wt_filtered_r2[wt_filtered_r2["Power"] > (outlier_threshold*p_rated)]
    wt_filtered_r2 = wt_filtered_r2[wt_filtered_r2["Power"] > (outlier_threshold*p_rated)]

    wt_outliers_r2 = wt_r2.iloc[outliers_r2]

    print(f"R2 % removed: {(len(wt_r2) - len(wt_filtered_r2))/len(wt_r2)*100}") 

    ## region III filter
    p_r3 = wt_r3[["Power"]]
    X_r3 = (p_r3 - p_r3.mean())/p_r3.std()
    knn_r3 = NearestNeighbors(n_neighbors=n_neighbors, **kwargs)
    knn_r3.fit(X_r3)
    distances_r3, _ = knn_r3.kneighbors(X_r3)

    avg_dist_r3 = distances_r3.mean(axis=1)
    [outliers_r3] = np.where(avg_dist_r3 > .075)
    keep_idxs_r3 = [i for i, _ in enumerate(wt_r3.index) if i not in outliers_r3]
    wt_filtered_r3 = wt_r3.iloc[keep_idxs_r3]
    wt_outliers_r3 = wt_r3.iloc[outliers_r3]

    print(f"R3 % removed: {len(outliers_r3)/len(wt_r3)*100}") 

    wt_filtered_r3 = wt_r3.iloc[keep_idxs_r3]
    wt_outliers_r3 = wt_r3.iloc[outliers_r3]

    return pd.concat([wt_r1, wt_filtered_r2, wt_filtered_r3])


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