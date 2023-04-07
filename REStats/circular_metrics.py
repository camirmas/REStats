import numpy as np
from scipy.stats import circmean, circstd


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
        predicted_values (np.array or pd.Series): Predicted values of the circular variable.
        
    Returns:
        float: Circular RMSE value.
    """
    # Calculate the angular distance between true and predicted values
    angular_distance = np.radians(true_values) - np.radians(predicted_values)
    
    # Normalize the angular distance to the range [-pi, pi]
    angular_distance = np.arctan2(np.sin(angular_distance), np.cos(angular_distance))
    
    # Calculate the squared errors and take the mean
    squared_errors = (np.degrees(angular_distance) ** 2)
    mse = np.mean(squared_errors)
    
    # Calculate the square root of the mean squared error
    rmse = np.sqrt(mse)
    
    return rmse


def circular_mae(true_values, predicted_values):
    """
    Calculates the Mean Absolute Error (MAE) for circular variables.
    
    Args:
        true_values (np.array or pd.Series): True values of the circular variable.
        predicted_values (np.array or pd.Series): Predicted values of the circular variable.
        
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