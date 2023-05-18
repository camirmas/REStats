import numpy as np
from scipy.stats import circstd, circmean


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


def circular_rmse(true_values, predicted_values):
    """
    Calculates the Root Mean Squared Error (RMSE) for circular variables.

    Args:
        true_values (np.array or pd.Series): True values of the circular variable.
        predicted_values (np.array or pd.Series): Predicted values of the circular
            variable.

    Returns:
        float: Circular RMSE value.
    """
    # Calculate the angular distance between true and predicted values
    angular_distance = np.radians(true_values) - np.radians(predicted_values)

    # Normalize the angular distance to the range [-pi, pi]
    angular_distance = np.arctan2(np.sin(angular_distance), np.cos(angular_distance))

    # Calculate the squared errors and take the mean
    squared_errors = np.degrees(angular_distance) ** 2
    mse = np.mean(squared_errors)

    # Calculate the square root of the mean squared error
    rmse = np.sqrt(mse)

    return rmse


def circular_mae(true_values, predicted_values):
    """
    Calculates the Mean Absolute Error (MAE) for circular variables.

    Args:
        true_values (np.array or pd.Series): True values of the circular variable.
        predicted_values (np.array or pd.Series): Predicted values of the circular
            variable.

    Returns:
        float: Circular MAE value.
    """
    # Calculate the angular distance between true and predicted values
    angular_distance = np.radians(true_values) - np.radians(predicted_values)

    # Normalize the angular distance to the range [-pi, pi]
    angular_distance = np.arctan2(np.sin(angular_distance), np.cos(angular_distance))

    # Calculate the absolute errors and take the mean
    absolute_errors = np.abs(np.degrees(angular_distance))
    mae = np.mean(absolute_errors)

    return mae


def circular_residuals(true_degrees, predicted_degrees):
    """
    Compute the circular residuals between true and predicted wind direction values in
    degrees.

    This function takes into account the circular nature of wind direction data when
    calculating residuals. The residuals are computed as the angular differences
    between true and predicted values, resulting in values between -180 and 180 degrees.

    Args:
        true_degrees (array_like): A 1D array or list of true wind direction values
            in degrees.
        predicted_degrees (array_like): A 1D array or list of predicted wind direction
            values in degrees, corresponding to the true_degrees.

    Returns:
        numpy.ndarray: A 1D numpy array of circular residuals in degrees.
    """
    true_radians = np.radians(true_degrees)
    predicted_radians = np.radians(predicted_degrees)

    residuals_radians = np.arctan2(
        np.sin(true_radians - predicted_radians),
        np.cos(true_radians - predicted_radians),
    )
    residuals_degrees = np.degrees(residuals_radians)
    return residuals_degrees


def circular_err(obs, pred, unit="deg", verbose=True):
    """
    Compute Root Mean Square Error (RMSE), relative RMSE, and Mean Absolute Error (MAE)
    in circular measurements (like angles).

    Args:
        obs (np.ndarray): An array of observed values.
        pred (np.ndarray): An array of predicted values.
        unit (str, optional): Unit of the error values to be printed, default is 'deg'
            (degrees).
        verbose (bool, optional): If True, print error values; default is True.

    Returns:
        tuple: A tuple containing RMSE, relative RMSE (in percentage) and MAE.
    """
    rmse = circular_rmse(obs, pred)
    rmse_rel = rmse / circular_mean(obs) * 100
    mae = circular_mae(obs, pred)

    if verbose:
        print(f"RMSE: {rmse} {unit}")
        print(f"RMSE (%): {rmse_rel}")
        print(f"MAE: {mae} {unit}")

    return rmse, rmse_rel, mae
