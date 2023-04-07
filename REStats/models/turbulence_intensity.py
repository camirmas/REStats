def persistence_turbulence_intensity(df, steps=1):
    """
    Predicts turbulence intensity using the persistence method.
    
    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex, containing "turbulence_intensity" column.
        steps (int): Number of steps ahead to forecast.
        
    Returns:
        pd.Series: Predicted turbulence intensity using the persistence method.
    """
    return df["turbulence_intensity"].shift(steps)
