import os

import pandas as pd
from sklearn.ensemble import IsolationForest


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
    wt = wt.rename(columns={"# Date and time": "Date", "Power (kW)": "Power", "Wind direction (°)": "Wind direction", "Wind speed (m/s)": "Wind speed"})
    wt["Date"] = pd.to_datetime(wt["Date"].astype("datetime64"))
    wt = wt.set_index("Date")
    wt = wt.asfreq("10min")
    wt = wt.sort_index()

    return wt


def filter_outliers(data, contamination=.05):
    iforest = IsolationForest(contamination=contamination)
    pred = iforest.fit_predict(data)
    filtered = [p == 1 for p in pred]
    outliers = [p == -1 for p in pred]

    return data[filtered], data[outliers]

