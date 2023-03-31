import numpy as np
import pandas as pd
import pytest
from REStats.models import calc_iec_power_curve


def test_calc_iec_power_curve():
    # Prepare input data
    wind_speed = np.arange(1, 19)
    power = np.linspace(0, 7500, len(wind_speed))
    data = pd.DataFrame({'wind_speed': wind_speed, 'power': power})

    # Define parameters
    cut_out = 18
    bin_size = 0.5

    # Call the function
    output = calc_iec_power_curve(data, cut_out, bin_size)
    output.reset_index(drop=True, inplace=True)

    # Prepare the expected output
    wt_bins = np.arange(0, cut_out, bin_size)
    wt_groups = data.groupby(pd.cut(data.wind_speed, wt_bins))
    expected_output = wt_groups.mean().reset_index(drop=True)

    # Test the function
    pd.testing.assert_frame_equal(output, expected_output, check_exact=False, rtol=1e-5, atol=1e-5)

    # Test with missing columns
    with pytest.raises(ValueError):
        calc_iec_power_curve(pd.DataFrame({'wind_speed': [1, 2, 3]}), cut_out, bin_size)

    with pytest.raises(ValueError):
        calc_iec_power_curve(pd.DataFrame({'power': [100, 200, 300]}), cut_out, bin_size)

    # Test with invalid cut_out
    with pytest.raises(ValueError):
        calc_iec_power_curve(data, 0, bin_size)

    with pytest.raises(ValueError):
        calc_iec_power_curve(data, -1, bin_size)
