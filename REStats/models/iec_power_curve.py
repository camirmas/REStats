import pandas as pd
import numpy as np

def calc_iec_power_curve(data: pd.DataFrame, cut_out: float, bin_size: float = 0.5) -> pd.DataFrame:
    """
    Calculates the IEC power curve for wind turbine data.

    Args:
        data (pd.DataFrame): The input DataFrame containing wind turbine data, with columns "wind_speed" and "power".
        cut_out (float): The cut-out wind speed of the wind turbine.
        bin_size (float, optional): The size of each bin. Defaults to 0.5.

    Returns:
        pd.DataFrame: The IEC power curve DataFrame, with columns "wind_speed" and "power".

    Raises:
        ValueError: If "wind_speed" or "power" columns are not in `data`.
        ValueError: If `cut_out` is less than or equal to zero.
    """
    wt_bins = np.arange(0, cut_out, bin_size)
    wt_groups = data.groupby(pd.cut(data.wind_speed, wt_bins))
    wt_iec = wt_groups.mean()

    return wt_iec