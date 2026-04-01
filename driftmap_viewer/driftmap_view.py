from driftmap_viewer.data_loader import DataLoader
from driftmap_viewer.interactive.driftmap_plot_widget import DriftmapPlotWidget
from driftmap_viewer import mpl_plotting

# test ideas:
# check signatures match default args between itneractive and matplotlib


class DriftMapView:
    """Load Kilosort sorter output and provide interactive or static drift map plots.

    On construction, spike data is loaded from a Kilosort output directory
    and stored as read-only arrays. Plotting methods apply optional
    filtering (noise exclusion, amplitude filtering, decimation) before
    handing the data to a plot backend.

    Parameters
    ----------
    sorter_path : str | Path
        Path to a Kilosort sorter output directory. Must contain
        exactly one ``kilosort*.log`` file used to detect the KS version.

    Attributes
    ----------
    spike_times : np.ndarray
        (num_spikes,) spike times (seconds for KS 1-3, samples for KS4).
    spike_amplitudes : np.ndarray
        (num_spikes,) spike amplitudes.
    spike_depths : np.ndarray
        (num_spikes,) spike depths along the probe (µm).
    spike_clusters : np.ndarray
        (num_spikes,) template index assigned to each spike.
    templates : np.ndarray
        (num_templates, num_samples, num_channels) template waveforms.
    channel_locations : np.ndarray
        (num_channels, 2) x/y positions of each channel on the probe.
    """

    def __init__(self, sorter_path):
        """Load spike data from a Kilosort output directory.

        Parameters
        ----------
        sorter_path : str | Path
            Path to the Kilosort sorter output.

        Raises
        ------
        AssertionError
            If the directory does not contain exactly one ``kilosort*.log``
            file, or if the loaded spike arrays have mismatched sizes.
        """
        self.data_loader = DataLoader(sorter_path) # TODO: rename

    def drift_map_plot_interactive(
        self,
        decimate=False,
        exclude_noise=False,
        amplitude_scaling="linear",
        n_color_bins=20,
        point_size=7.5,
        filter_amplitude_mode=None,
        filter_amplitude_values=(),
    ):
        """Create an interactive pyqtgraph-based drift map widget.

        Parameters
        ----------
        decimate :
        exclude_noise :
        amplitude_scaling :
        n_color_bins :
        point_size :
        filter_amplitude_mode :
        filter_amplitude_values : tuple of float

        Returns
        -------
        DriftmapPlotWidget
            The pyqtgraph widget. This is already populated but not yet
            shown, use app.exec() to display.
        """
        processed_data = self.data_loader.get_processed_data(
            exclude_noise, decimate, filter_amplitude_mode, filter_amplitude_values
        )

        self.plot = DriftmapPlotWidget(
            processed_data,
            amplitude_scaling=amplitude_scaling,
            n_color_bins=n_color_bins,
            point_size=point_size,
        )

        return self.plot

    def drift_map_plot_matplotlib(
        self,
        decimate=False,
        exclude_noise=False,
        amplitude_scaling="linear",
        n_color_bins=20,
        point_size=7.5,
        filter_amplitude_mode=None,
        filter_amplitude_values=(),
        add_histogram_plot=False,
        weight_histogram_by_amplitude=False,
    ):
        """"""
        processed_data = self.data_loader.get_processed_data(
            exclude_noise, decimate, filter_amplitude_mode, filter_amplitude_values
        )

        fig = mpl_plotting.plot_matplotlib(
            processed_data,
            amplitude_scaling,
            n_color_bins,
            point_size,
            add_histogram_plot,
            weight_histogram_by_amplitude,
        )

        return fig
