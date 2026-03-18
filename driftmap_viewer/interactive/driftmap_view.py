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


class DriftMapView():
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
    channel_positions : np.ndarray
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
        self.sorter_path = Path(sorter_path)

        log_file = list(self.sorter_path.glob("kilosort*.log"))
        assert len(log_file) == 1
        self.ks_version = Path(log_file[0]).name.split(".")[0]

        func = kilosort_4.get_spikes_info_ks4 if self.ks_version == "kilosort4" else  kilosort1_3.get_spikes_info_ks1_3

        (
            self.spike_times,
            self.spike_amplitudes,
            self.spike_depths,
            self.spike_templates,
            self.templates,
            self.channel_positions
        ) = func(
            self.sorter_path
        )

        assert self.spike_times.size == self.spike_amplitudes.size == self.spike_depths.size == self.spike_templates.size

        self.spike_times.flags.writeable = False
        self.spike_amplitudes.flags.writeable = False
        self.spike_depths.flags.writeable = False
        self.spike_templates.flags.writeable = False
        self.templates.flags.writeable = False
        self.channel_positions.flags.writeable = False

    def _process_data(
        self,
        exclude_noise,
        decimate,
        filter_amplitude_mode,
        filter_amplitude_values
     ):
        """Filter and subsample the loaded spike data.

        Operations are applied in order: decimation → noise exclusion →
        amplitude filtering → masking. Decimation is applied first as a
        performance knob to thin the full dataset before further filtering.

        Parameters
        ----------
        exclude_noise : bool
            If ``True``, spikes belonging to clusters labelled "noise" in
            the Kilosort cluster groups file are removed.
        decimate : int | False
            Keep every *n*-th spike. Applied first to reduce the dataset
            before noise/amplitude filters. ``False`` disables decimation.
        filter_amplitude_mode : {"percentile", "absolute"} | None
            How ``filter_amplitude_values`` is interpreted.
            ``None`` disables amplitude filtering.
        filter_amplitude_values : tuple of float
            (low, high) bounds. Interpreted as percentile ranks or
            absolute amplitude values depending on ``filter_amplitude_mode``.

        Returns
        -------
        spike_times : np.ndarray
        spike_amplitudes : np.ndarray
        spike_depths : np.ndarray
        spike_templates : np.ndarray
            Filtered copies (views when no filtering is needed) of the
            corresponding instance arrays.
        """
        # Select a view for now, this may be copied depending on options (e.g. decimate)
        spike_times = self.spike_times
        spike_amplitudes = self.spike_amplitudes
        spike_depths = self.spike_depths
        spike_templates = self.spike_templates

        keep_bool_mask = None

        if exclude_noise:
            keep_bool_mask = ~helpers.get_noise_mask(
                self.sorter_path
            )

        if filter_amplitude_mode is not None:
            assert filter_amplitude_mode in ["percentile", "absolute"]

            if filter_amplitude_mode == "percentile":
                min_val, max_val = np.percentile(
                    spike_amplitudes, filter_amplitude_values
                )
            else:
                min_val, max_val = filter_amplitude_values

            if keep_bool_mask is None:
                keep_bool_mask = np.ones(spike_amplitudes.size, dtype=bool)

            keep_bool_mask[spike_amplitudes < min_val] = False
            keep_bool_mask[spike_amplitudes > max_val] = False

        if keep_bool_mask is not None:
            spike_times = spike_times[keep_bool_mask]
            spike_amplitudes = spike_amplitudes[keep_bool_mask]
            spike_depths = spike_depths[keep_bool_mask]
            spike_templates = spike_templates[keep_bool_mask]

        if decimate:
            spike_times = spike_times[:: decimate]
            spike_amplitudes = spike_amplitudes[:: decimate]
            spike_depths = spike_depths[:: decimate]
            spike_templates = spike_templates[:: decimate]

        return spike_times, spike_amplitudes, spike_depths, spike_templates

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

        self.plot = DriftmapPlotWidget(
            spike_times,
            spike_amplitudes,
            spike_depths,
            spike_templates,
            self.templates,
            self.channel_positions,
            amplitude_scaling=amplitude_scaling,
            n_color_bins=n_color_bins,
            point_size=point_size,
            sorter_path=self.sorter_path
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
