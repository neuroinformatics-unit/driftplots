import matplotlib.pyplot as plt


def _plot_matplotlib(
    processed_data,
    amplitude_scaling,
    n_color_bins,
    point_size,
    add_histogram_plot,
    weight_histogram_by_amplitude,
) -> None:

    # Setup axis and plot the raster drift map
    fig = plt.figure(figsize=(10, 10 * (6 / 8)))

    if add_histogram_plot:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 5])
        hist_axis = fig.add_subplot(gs[0])
        raster_axis = fig.add_subplot(gs[1], sharey=hist_axis)
    else:
        raster_axis = fig.add_subplot()

    spike_times, spike_depths, _ = processed_data.get_scatter_data()

    # set amplitude colors
    rgba_colors = processed_data.compute_amplitude_colors(
        amplitude_scaling, n_color_bins, unit_normalise=True
    )

    raster_axis.scatter(
        spike_times,
        spike_depths,
        c=rgba_colors,
        s=point_size,
        antialiased=True,
    )

    if not add_histogram_plot:
        raster_axis.set_xlabel("time")
        raster_axis.set_ylabel("y position")
        return fig

    # If the histogram plot is requested, plot it alongside
    # it's peak colouring, bounds display and drift point display.
    hist_axis.set_xlabel("count")
    raster_axis.set_xlabel("time")
    hist_axis.set_ylabel("y position")

    bin_centers, counts = processed_data.compute_activity_histogram(
        weight_histogram_by_amplitude
    )
    hist_axis.plot(counts, bin_centers, color="black", linewidth=1)

    return fig
