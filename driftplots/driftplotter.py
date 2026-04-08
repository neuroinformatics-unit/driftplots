from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from driftplots import mpl_plotting
from driftplots.data_loader import DataLoader
from driftplots.interactive.driftmap_plot_widget import DriftmapPlotWidget
from pyqtgraph.Qt import QtWidgets

# test ideas:
# check signatures match default args between interactive and matplotlib


class DriftPlotter:
    """Load Kilosort sorter output and provide interactive or static drift map plots.

    On construction, spike data is loaded from a Kilosort output directory
    and stored as read-only arrays. Plotting methods apply optional
    filtering (noise exclusion, amplitude filtering, decimation) before
    handing the data to a plot backend.

    Parameters
    ----------
    sorter_path
        Path to a Kilosort sorter output directory. Must contain
        exactly one ``kilosort*.log`` file used to detect the KS version.

    Attributes
    ----------
    spike_times
        (num_spikes,) spike times (seconds for KS 1-3, samples for KS4).
    spike_amplitudes
        (num_spikes,) spike amplitudes.
    spike_depths
        (num_spikes,) spike depths along the probe (µm).
    spike_clusters
        (num_spikes,) template index assigned to each spike.
    templates
        (num_templates, num_samples, num_channels) template waveforms.
    channel_locations
        (num_channels, 2) x/y positions of each channel on the probe.
    """

    def __init__(self, sorter_path: str | Path) -> None:
        """Load spike data from a Kilosort output directory.

        Parameters
        ----------
        sorter_path
            Path to the Kilosort sorter output.

        Raises
        ------
        AssertionError
            If the directory does not contain exactly one ``kilosort*.log``
            file, or if the loaded spike arrays have mismatched sizes.
        """
        self.data_loader = DataLoader(sorter_path)  # TODO: rename

    def drift_map_plot_interactive(
        self,
        decimate: int | bool | None | Literal["estimate"] = "estimate",
        exclude_noise: bool = False,
        amplitude_cmap_scaling: str | tuple[float, float] = "linear",
        n_color_bins: int = 20,
        point_size: float = 5.0,
        filter_amplitude_mode: str | None = None,
        filter_amplitude_values: tuple[float, ...] = (),
    ) -> DriftmapPlotWidget:
        """Create an interactive pyqtgraph-based drift map widget.

        Parameters
        ----------
        decimate
            Keep every *n*-th spike. Too many spikes will slow down the plot.
            if ``"estimate"` the number of spikes will be decimated to a reasonable
            range (e.g. 50,000). ``False``, ``None`` or ``0`` disables decimation.`
            Otherwise pass an integer e.g. 2 to keep every 2nd spike.
        exclude_noise
            Remove spikes labelled as noise.
        amplitude_cmap_scaling
            Colour-scaling mode or explicit ``(min, max)`` range.
        n_color_bins
            Number of grey-scale colour bins for amplitude.
        point_size
            Scatter-point diameter in pixels.
        filter_amplitude_mode
            Amplitude filtering mode.
        filter_amplitude_values
            Bounds for amplitude filtering.

        Returns
        -------
        DriftmapPlotWidget
            The pyqtgraph widget. This is already populated but not yet
            shown, use app.exec() to display.
        """
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        processed_data = self.data_loader.get_processed_data(
            exclude_noise, decimate, filter_amplitude_mode, filter_amplitude_values
        )

        self.plot = DriftmapPlotWidget(
            processed_data,
            app,
            amplitude_cmap_scaling=amplitude_cmap_scaling,
            n_color_bins=n_color_bins,
            point_size=point_size,
        )

        return self.plot

    def drift_map_plot_matplotlib(
        self,
        decimate: int | bool | None | Literal["estimate"] = "estimate",
        exclude_noise: bool | str = False,
        amplitude_cmap_scaling: str | tuple[float, float] = "linear",
        n_color_bins: int = 20,
        point_size: float = 5.0,
        filter_amplitude_mode: str | None = None,
        filter_amplitude_values: tuple[float, ...] = (),
        add_histogram_plot: bool = False,
        weight_histogram_by_amplitude: bool = False,
    ) -> Figure:
        """"""
        processed_data = self.data_loader.get_processed_data(
            exclude_noise, decimate, filter_amplitude_mode, filter_amplitude_values
        )

        fig = mpl_plotting.plot_matplotlib(
            processed_data,
            amplitude_cmap_scaling,
            n_color_bins,
            point_size,
            add_histogram_plot,
            weight_histogram_by_amplitude,
        )

        return fig
