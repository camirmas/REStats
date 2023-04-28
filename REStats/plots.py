import os

from windrose import WindroseAxes
import matplotlib.pyplot as plt


def save_figs(figs, format="pdf", output_dir=None):
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    output_dir = output_dir or (curr_dir + os.sep + "../figs")
    
    for name in figs:
        outfile = f"{output_dir}/{name}.{format}"
        figs[name].savefig(outfile, bbox_inches="tight")


def plot_wind_rose(df, speed_col="wind_speed", direction_col="wind_dir", bins=None, cmap=None, legend=True, **kwargs):
    """
    Plot a wind rose using a pandas DataFrame and return the figure object.

    Parameters:
    df (pandas.DataFrame): DataFrame containing wind speed and direction data.
    speed_col (str): Column name for wind speed data.
    direction_col (str): Column name for wind direction data.
    bins (list or int, optional): Number of bins or a list of bins for wind speed data. Defaults to None.
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