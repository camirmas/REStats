import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_approx_equal

from REStats.utils import (
    transform, inv_transform, filter_outliers, 
    circular_mean, circular_std, downsample, standardize)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'wind_speed': [5, 10, 15, 20],
        'wind_dir': [0, 90, 180, 270],
        'power': [100, 200, 300, 400]
    })


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
    df = pd.DataFrame({"Wind speed (m/s)": wind_speed, "Power (kW)": power})

    return df


def test_transforms():
    m = .5
    index = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    v_df = pd.DataFrame({"wind_speed": np.linspace(0, 20, len(index))}, index=index)

    v_tf, hr_stats = transform(v_df, m)

    v_tf["y"] = v_tf.v_scaled_std
    v_inv = inv_transform(v_tf, m, hr_stats)

    assert_array_almost_equal(v_df["wind_speed"], v_inv.y)


def test_filter_outliers(power_curve_data):
    # Add outliers
    outliers = pd.DataFrame({"Wind speed (m/s)": [22], "Power (kW)": [150]})
    pc = pd.concat([power_curve_data, outliers])
    # Filter outliers
    filtered_df = filter_outliers(pc, "Wind speed (m/s)", "Power (kW)", 0.5)

    # Check that no outliers remain in the filtered data
    assert not ((filtered_df["Wind speed (m/s)"] > 15) & (filtered_df["Power (kW)"] < 1000)).any()


def test_standardize(sample_dataframe):
    standardized_df = standardize(sample_dataframe)
    
    # Check if the standardized dataframe has the same columns
    assert standardized_df.columns.tolist() == sample_dataframe.columns.tolist()

    # Check if wind_speed and power columns are standardized
    for col in ["wind_speed", "power"]:
        mean = standardized_df[col].mean()
        std = standardized_df[col].std()
        np.testing.assert_almost_equal(mean, 0, decimal=6)
        np.testing.assert_almost_equal(std, 1, decimal=6)

    # Check if wind_dir column is standardized as circular data
    mean_wind_dir = circular_mean(sample_dataframe["wind_dir"])
    std_wind_dir = circular_std(sample_dataframe["wind_dir"])
    expected_wind_dir = (sample_dataframe["wind_dir"] - mean_wind_dir) / std_wind_dir
    np.testing.assert_almost_equal(standardized_df["wind_dir"].values, expected_wind_dir.values, decimal=6)


def test_downsample():
    # Create a sample DataFrame
    date_rng = pd.date_range(start="2020-01-01", end="2020-01-02", freq="10min", inclusive="left")
    wind_speed_data = np.random.uniform(0, 10, size=(len(date_rng),))
    wind_dir_data = np.random.uniform(0, 360, size=(len(date_rng),))
    power_data = np.random.uniform(0, 2000, size=(len(date_rng),))

    data = {
        "wind_speed": wind_speed_data,
        "wind_dir": wind_dir_data,
        "power": power_data
    }
    df = pd.DataFrame(data, index=date_rng)

    # Downsample the DataFrame
    downsampled_df = downsample(df)

    # Check if the downsampled DataFrame has the expected length
    expected_length = len(df.resample("1H"))
    assert len(downsampled_df) == expected_length

    # Check if the downsampled DataFrame has the correct columns
    assert "wind_speed" in downsampled_df.columns
    assert "wind_dir" in downsampled_df.columns
    assert "power" in downsampled_df.columns
    assert "turbulence_intensity" in downsampled_df.columns

    # Check if the downsampled wind_speed values are close to the mean of the original data
    for time, group in df.groupby(df.index.hour):
        expected_wind_speed = group["wind_speed"].mean()
        assert downsampled_df.loc[group.index[0].floor("H"), "wind_speed"] == pytest.approx(expected_wind_speed, abs=1e-6)

    # Check if the downsampled wind_dir values are close to the circular mean of the original data
    for time, group in df.groupby(df.index.hour):
        expected_wind_dir = circular_mean(group["wind_dir"].values)
        assert downsampled_df.loc[group.index[0].floor("H"), "wind_dir"] == pytest.approx(expected_wind_dir, abs=1e-6)

    # Check if the downsampled power values are close to the mean of the original data
    for time, group in df.groupby(df.index.hour):
        expected_power = group["power"].mean()
        assert downsampled_df.loc[group.index[0].floor("H"), "power"] == pytest.approx(expected_power, abs=1e-6)

    # Check if the downsampled turbulence_intensity values are close to the mean of the original data
    for time, group in df.groupby(df.index.hour):
        expected_turbulence_intensity = group["turbulence_intensity"].mean()
        assert downsampled_df.loc[group.index[0].floor("H"), "turbulence_intensity"] == pytest.approx(expected_turbulence_intensity, abs=1e-6)
