from driftmapviewer_new import get_drift_map_plot, _plot_kilosort_drift_map_raster
import matplotlib.pyplot as plt
import kilosort1_3
import kilosort_4
import helpers
from pathlib import Path
import numpy as np
from driftmap_plot_widget import DriftmapPlotWidget
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import matplotlib.pyplot as plt


class DriftMapView():
    """
    # TODO: pay the cost once, then can plot a lot
    # TOOD: compute cost of holding all in memory
    """
    def __init__(self, sorter_path):
        """
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

    def _process_data(
        self,
        exclude_noise,
        decimate,
        filter_amplitude_mode,
        filter_amplitude_values
     ):
        """

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

        if decimate:
            spike_times = spike_times[:: decimate]
            spike_amplitudes = spike_amplitudes[:: decimate]
            spike_depths = spike_depths[:: decimate]
            spike_templates = spike_templates[:: decimate]

            if keep_bool_mask is not None:
                keep_bool_mask = keep_bool_mask[:: decimate]

        if keep_bool_mask is not None:
            spike_times = spike_times[keep_bool_mask]
            spike_amplitudes = spike_amplitudes[keep_bool_mask]
            spike_depths = spike_depths[keep_bool_mask]
            spike_templates = spike_templates[keep_bool_mask]

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

        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

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
