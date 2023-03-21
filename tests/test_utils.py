import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from REStats.utils import transform, inv_transform, filter_outliers


@pytest.fixture
def power_curve_data():
    # Generate wind speed data from 0 to 25 m/s
    wind_speed = np.linspace(0, 25, 500)

    # Generate power data using a simplified power curve formula
    cut_in_speed = 3  # m/s
    rated_speed = 12  # m/s
    cut_out_speed = 25  # m/s
    rated_power = 2000  # kW
    power = np.piecewise(wind_speed, [
        wind_speed < cut_in_speed,
        (wind_speed >= cut_in_speed) & (wind_speed < rated_speed),
        (wind_speed >= rated_speed) & (wind_speed <= cut_out_speed),
        wind_speed > cut_out_speed,
    ], [
        0,
        lambda x: rated_power * ((x - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3,
        rated_power,
        0
    ])

    # Create DataFrame from wind speed and power data
    df = pd.DataFrame({'Wind speed (m/s)': wind_speed, 'Power (kW)': power})

    return df


def test_transforms():
    m = .5
    index = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    v_df = pd.DataFrame({"Wind speed": np.linspace(0, 20, len(index))}, index=index)

    v_tf, hr_stats = transform(v_df, m)

    v_tf["y"] = v_tf.v_scaled_std
    v_inv = inv_transform(v_tf, m, hr_stats)

    assert_array_almost_equal(v_df["Wind speed"], v_inv.y)


def test_filter_outliers(power_curve_data):
    # Add outliers
    outliers = pd.DataFrame({"Wind speed (m/s)": [22], "Power (kW)": [150]})
    pc = pd.concat([power_curve_data, outliers])
    # Filter outliers
    filtered_df = filter_outliers(pc, 'Wind speed (m/s)', 'Power (kW)', 0.5)

    # Check that no outliers remain in the filtered data
    assert not ((filtered_df['Wind speed (m/s)'] > 15) & (filtered_df['Power (kW)'] < 1000)).any()