def persistence_wind_dir(df, steps=1):
    """
    Predicts wind direction using the persistence method.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex, containing "wind_dir" column.
        steps (int): Number of steps ahead to forecast.

    Returns:
        pd.Series: Predicted wind direction using the persistence method.
    """
    return df["wind_dir"].shift(steps)
