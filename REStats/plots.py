import os

import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes


def save_figs(figs, format="pdf", output_dir=None):
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    output_dir = output_dir or (curr_dir + os.sep + "../figs")

    for name in figs:
        outfile = f"{output_dir}/{name}.{format}"
        figs[name].savefig(outfile, bbox_inches="tight")


def plot_wind_rose(
    df,
    speed_col="wind_speed",
    direction_col="wind_dir",
    bins=None,
    cmap=None,
    legend=True,
    **kwargs,
):
    """
    Plot a wind rose using a pandas DataFrame and return the figure object.

    Parameters:
    df (pandas.DataFrame): DataFrame containing wind speed and direction data.
    speed_col (str): Column name for wind speed data.
    direction_col (str): Column name for wind direction data.
    bins (list or int, optional): Number of bins or a list of bins for wind speed data.
        Defaults to None.
    cmap (list, optional): Colormap for the plot.
    legend (bool, optional): Whether to display the legend. Defaults to True.

    Returns:
    matplotlib.figure.Figure: The generated figure object.
    """
    # Create a new figure and windrose axes
    fig = plt.figure(**kwargs)
    ax = WindroseAxes.from_ax(fig=fig)

    # Create the wind rose plot
    ax.bar(df[direction_col], df[speed_col], normed=True, bins=bins, cmap=cmap)

    # Show legend
    if legend:
        ax.set_legend()

    return fig


def plot_circular_histogram(true_directions, predicted_directions, num_bins=36):
    """
    Plot a circular histogram of true and predicted wind direction values in degrees.

    Args:
        true_directions (array_like): A 1D array or list of true wind direction values
            in degrees.
        predicted_directions (array_like): A 1D array or list of predicted wind
            direction values in degrees.
        num_bins (int, optional): Number of bins for the histogram. Default is 36.

    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object containing the circular
            histogram plot.
    """
    true_radians = np.radians(true_directions)
    predicted_radians = np.radians(predicted_directions)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    bins = np.linspace(0, 2 * np.pi, num_bins + 1)
    true_hist, _, _ = ax.hist(
        true_radians, bins=bins, alpha=0.5, color="blue", label="True Directions"
    )
    predicted_hist, _, _ = ax.hist(
        predicted_radians,
        bins=bins,
        alpha=0.5,
        color="red",
        label="Predicted Directions",
    )

    ax.legend(loc="upper right")

    return fig
