import pandas as pd
import numpy as np

def calc_iec_power_curve(data, cut_out, bin_size=.5):
    wt_bins = np.arange(0, cut_out, bin_size)
    wt_groups = data.groupby(pd.cut(data.wind_speed, wt_bins))
    wt_iec = wt_groups.mean()

    return wt_iec