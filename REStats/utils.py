import os

import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd


def load_SCADA(year=2020):
    # TODO: Use full farm data in analysis (currently using one turbine).

    # DATA_DIRS = ["../data/Kelmarsh_SCADA_2019/", "../data/Kelmarsh/SCADA_2020"]

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


def filter_outliers(
        df: pd.DataFrame,
        bin_col: str = "wind_speed",
        outlier_col: str = "power",
        bin_size: float = .5
) -> pd.DataFrame:
    """
    Filters rows by IQR outliers for `outlier_col` for each bin grouped by `bin_col`.

    Args:
        df (pd.DataFrame): The input DataFrame.
        bin_col (str): The column name to use for binning.
        bin_size (float): The size of each bin for binning by `bin_col`.
        outlier_col (str): The column name to use for filtering outliers.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        ValueError: If `bin_col` or `outlier_col` are not columns in `df`.
        ValueError: If `bin_size` is less than or equal to zero.
    """
    # Create bins based on bin_col
    df["bins"] = pd.cut(df[bin_col], bins=np.arange(df[bin_col].min(), df[bin_col].max() + bin_size, bin_size))
    
    # Group by bins and calculate IQR for outlier_col
    grouped = df.groupby("bins")[outlier_col]
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = q3 - q1
    
    # Filter outliers for each bin
    filtered_df = pd.DataFrame()
    for name, group in df.groupby("bins"):
        is_outlier = (group[outlier_col] < (q1[name] - 1.5 * iqr[name])) | (group[outlier_col] > (q3[name] + 1.5 * iqr[name]))
        filtered_df = pd.concat([filtered_df, group[~is_outlier]])
    
    # Drop the bins column and return the filtered dataframe
    filtered_df.drop("bins", axis=1, inplace=True)

    return filtered_df


def transform(v_df, m, field="wind_speed", hr_stats=None):
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


def circular_mean(data):
    """
    Calculates the circular mean of the given data.

    Args:
        data (array-like): Input data in degrees.

    Returns:
        float: Circular mean in degrees.
    """
    data_rad = np.deg2rad(data)
    return np.rad2deg(circmean(data_rad))


def circular_std(data):
    """
    Calculates the circular standard deviation of the given data.

    Args:
        data (array-like): Input data in degrees.

    Returns:
        float: Circular standard deviation in degrees.
    """
    data_rad = np.deg2rad(data)
    return np.rad2deg(circstd(data_rad))


def standardize(df, ref_df=None):
    """
    Standardizes a DataFrame containing wind_speed, wind_dir, and power columns.

    Args:
        df (pandas.DataFrame): DataFrame containing wind_speed, wind_dir, and power columns.

    Returns:
        pandas.DataFrame: Standardized DataFrame.
    """
    standardized_df = df.copy()

    # Standardize wind_speed and power
    for col in ["wind_speed", "power"]:
        if ref_df is None:
            mean = df[col].mean()
            std = df[col].std()
        else:
            mean = ref_df[col].mean()
            std = ref_df[col].std()

        standardized_df[col] = (df[col] - mean) / std

    # Standardize wind_dir (circular data)
    if ref_df is None:
        mean_wind_dir = circular_mean(df["wind_dir"])
        std_wind_dir = circular_std(df["wind_dir"])
    else:
        mean_wind_dir = circular_mean(ref_df["wind_dir"])
        std_wind_dir = circular_std(ref_df["wind_dir"])

    standardized_df["wind_dir"] = ((df["wind_dir"] - mean_wind_dir) / std_wind_dir)

    return standardized_df


def downsample(df):
    """
    Downsamples a pandas DataFrame containing a 10-minute time series of wind speed and wind direction data to 1 hour.
    
    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex, containing "wind_speed" and "wind_dir" columns.
        
    Returns:
        pd.DataFrame: Downsampled DataFrame with 1-hour resolution.
    """
    # Resample wind speed using mean
    wind_speed_h = df["wind_speed"].resample("1H").mean()

    # Resample wind direction using circular mean
    wind_dir_h = df["wind_dir"].resample("1H").apply(lambda x: circular_mean(x.values))

    # Resample power using mean
    power_h = df["power"].resample("1H").mean()

    # Combine resampled data into a new DataFrame
    downsampled_df = pd.DataFrame({
        "wind_speed": wind_speed_h,
        "wind_dir": wind_dir_h,
        "power": power_h
    })
    
    return downsampled_df