from driftmap_viewer.mpl_plotting.driftmapviewer_new import get_drift_map_plot, _plot_kilosort_drift_map_raster
import matplotlib.pyplot as plt
from driftmap_viewer.ks_extractors import kilosort1_3
from driftmap_viewer.ks_extractors import kilosort_4
from driftmap_viewer.ks_extractors import helpers
from pathlib import Path
import numpy as np
from .driftmap_plot_widget import DriftmapPlotWidget
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import matplotlib.pyplot as plt
from driftmap_viewer.interactive.data_model import DataLoader


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
    spike_templates : np.ndarray
        (num_spikes,) template index assigned to each spike.
    templates : np.ndarray
        (num_templates, num_samples, num_channels) template waveforms.
    channel_locations : np.ndarray
        (num_channels, 2) x/y positions of each channel on the probe.

    TODO
    ----
    - Evaluate memory cost of holding all arrays; consider lazy / mmap loading.
    - Harmonise spike_times units (seconds everywhere).
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
        self.data_loader = DataLoader(sorter_path)

    def drift_map_plot_interactive(
        self,
        decimate=False,
        exclude_noise=True,
        amplitude_scaling="linear",
        n_color_bins=20,
        point_size=7.5,
        filter_amplitude_mode=None,
        filter_amplitude_values=(),
    ):
        """Create an interactive pyqtgraph-based drift map widget.

        Parameters
        ----------
        decimate : int | False
            Keep every *n*-th spike. ``False`` disables decimation.
        exclude_noise : bool
            Remove spikes labelled as noise.
        amplitude_scaling : {"linear", "log2", "log10"} | tuple
            Colour-scaling mode. A 2-tuple ``(min, max)`` fixes the
            colour range explicitly.
        n_color_bins : int
            Number of grey-scale colour bins for amplitude.
        point_size : float
            Scatter-point diameter in pixels.
        filter_amplitude_mode : {"percentile", "absolute"} | None
            Amplitude filtering mode (see ``_process_data``).
        filter_amplitude_values : tuple of float
            Bounds for amplitude filtering.

        Returns
        -------
        DriftmapPlotWidget
            The pyqtgraph widget (already populated but not yet shown).
        """
        processed_data = self.data_loader.get_processed_data(
            exclude_noise,
            decimate,
            filter_amplitude_mode,
            filter_amplitude_values
        )

        self.plot = DriftmapPlotWidget(
            processed_data,
            amplitude_scaling=amplitude_scaling,
            n_color_bins=n_color_bins,
            point_size=point_size,
        )

        return self.plot

    # ----------------------------------------------------------------------------------
    # TODO MATPLOTLIB
    # ----------------------------------------------------------------------------------

    def _drift_map_plot_matplotlib(self,
        decimate=False,
        exclude_noise=True,
        log_transform_amplitudes=True,
        filter_amplitude_mode=None,
        filter_amplitude_values=(),
     ):
        (
            spike_times,
            spike_amplitudes,
            spike_depths,
            spike_templates,
        ) = self._process_data(
            exclude_noise,
            decimate,
            filter_amplitude_mode,
            filter_amplitude_values
        )

        fig = plt.figure(figsize=(10, 10 * (6 / 8)))
        raster_axis = fig.add_subplot()

        _plot_kilosort_drift_map_raster(
            spike_times,
            spike_amplitudes,
            spike_depths,
            axis=raster_axis,
        )

        # histogram
        if False:
            hist_axis.set_xlabel("count")
            raster_axis.set_xlabel("time")
            hist_axis.set_ylabel("y position")

            bin_centers, counts = _compute_activity_histogram(
                spike_amplitudes, spike_depths, weight_histogram_by_amplitude
            )
            hist_axis.plot(counts, bin_centers, color="black", linewidth=1)

        return fig
