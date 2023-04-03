import pandas as pd
import numpy as np


def calc_iec_power_curve(data: pd.DataFrame, bin_size: float = 0.5) -> pd.DataFrame:
    """
    Calculates the IEC power curve for wind turbine data.

    Args:
        data (pd.DataFrame): The input DataFrame containing wind turbine data, with columns "wind_speed" and "power".
        bin_size (float, optional): The size of each bin. Defaults to 0.5.

    Returns:
        pd.DataFrame: The IEC power curve DataFrame, with columns "wind_speed" and "power".

    Raises:
        ValueError: If "wind_speed" or "power" columns are not in `data`.
    """
    if 'wind_speed' not in data.columns or 'power' not in data.columns:
        raise ValueError("Input DataFrame must contain 'wind_speed' and 'power' columns.")

    min_ws = np.floor(np.min(data.wind_speed))
    max_ws = np.ceil(np.max(data.wind_speed))
    wt_bins = np.arange(min_ws, max_ws, bin_size)
    wt_groups = data.groupby(pd.cut(data.wind_speed, wt_bins))
    wt_iec = wt_groups.mean()

    return wt_iec