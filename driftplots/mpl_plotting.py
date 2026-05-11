from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

if TYPE_CHECKING:
    from driftplots.data_model import DataModel


def plot_matplotlib(
    processed_data: DataModel,
    amplitude_cmap_scaling: str | tuple[float, float],
    n_color_bins: int,
    point_size: float,
    add_histogram_plot: bool,
    weight_histogram_by_amplitude: bool,
    title: bool | str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Render a static matplotlib drift-map figure from pre-processed data.

    Parameters
    ----------
    processed_data :
    amplitude_cmap_scaling :

    n_color_bins :

    point_size :

    add_histogram_plot :

    weight_histogram_by_amplitude :

    title :

    ax :
        Existing axis to draw the drift map on.  When ``add_histogram_plot``
        is ``True``, a histogram axis is added beside this axis.

    Returns
    -------
    matplotlib.figure.Figure
        The drift-map figure.
    """
    fig, hist_axis, raster_axis = _setup_axes(add_histogram_plot, ax)

    # Plot the raster plot
    spike_times, spike_depths, _ = processed_data.get_scatter_data()

    rgba_colors = processed_data.compute_amplitude_colors(
        amplitude_cmap_scaling, n_color_bins, unit_normalise=True
    )

    raster_axis.scatter(
        spike_times,
        spike_depths,
        c=rgba_colors,
        s=point_size,
        antialiased=True,
    )

    if not add_histogram_plot:
        raster_axis.set_xlabel("Time (s)")
        raster_axis.set_ylabel("Depth (μm)")

    else:
        # Plot the histogram on the left-hand subplot
        hist_axis.set_xlabel("Count")
        raster_axis.set_xlabel("Time (s)")
        hist_axis.set_ylabel("Depth (μm)")

        bin_centers, counts = processed_data.compute_activity_histogram(
            weight_histogram_by_amplitude
        )
        hist_axis.plot(counts, bin_centers, color="black", linewidth=1)

    if title:
        fig.suptitle(title if isinstance(title, str) else "Drift Map")

    return fig


def _setup_axes(
    add_histogram_plot: bool,
    ax: Axes | None,
) -> tuple[Figure, Axes | None, Axes]:
    """Return the figure, optional histogram axis, and raster axis.

    When ``ax`` is ``None``, create a new figure and axes matching the
    package's default Matplotlib drift-map layout.  When ``ax`` is provided,
    use it as the raster axis and, if requested, append a left-side histogram
    axis that shares its y-axis.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10 * (6 / 8)))
        if add_histogram_plot:
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 5])
            hist_axis = fig.add_subplot(gs[0])
            raster_axis = fig.add_subplot(gs[1], sharey=hist_axis)
            return fig, hist_axis, raster_axis
        return fig, None, fig.add_subplot()

    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes or None")

    if not add_histogram_plot:
        return ax.figure, None, ax

    divider = make_axes_locatable(ax)
    hist_axis = divider.append_axes("left", size="20%", pad=0.05, sharey=ax)
    return ax.figure, hist_axis, ax
