from driftmapviewer_new import get_drift_map_plot, _plot_kilosort_drift_map_raster, _filter_large_amplitude_spikes
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


# TODO: dont use gain, instead set clim
# TODO: removed localised peaks
# TODO: removed drift event and boundaries

# TODO: it would be really cool and useful to hover over
# the plot and see the template waveform

class DriftMapView():
    def __init__(self, sorter_path):
        self.sorter_path = Path(sorter_path)

        log_file = list(self.sorter_path.glob("kilosort*.log"))
        assert len(log_file) == 1
        self.ks_version = Path(log_file[0]).name.split(".")[0]

        # TODO: pay the cost once, then can plot a lot
        # TOOD: compute cost of holding all in memory

        func = kilosort_4.get_spikes_info_ks4 if self.ks_version == "kilosort4" else  kilosort1_3.get_spikes_info_ks1_3

        (
            self.spike_times,
            self.spike_amplitudes,
            self.spike_depths,
            self.spike_templates,
            self.templates
        ) = func(
            self.sorter_path
        )

        self.spike_times.flags.writeable = False
        self.spike_amplitudes.flags.writeable = False
        self.spike_depths.flags.writeable = False
        self.spike_templates.flags.writeable = False
        self.templates.flags.writeable = False

    # TODO: this isn't great, it is convenient but wasteful. But much of this is necessary
    # e.g. compute amplitude with all spikes etc...
    # still refactor for return a spike bool, index on the fly, and have a func compute
    # min/max that takes the spike bool and uses to compute spike_amplitudes min/max
    # and compute log on the fly at the end...
    def _process_data(
        self,
        exclude_noise,
        log_transform_amplitudes,
        decimate,
        only_include_large_amplitude_spikes,
        large_amplitude_only_segment_size
     ):
        # start with a view, but we may end up with a copy depending on the settings
        spike_times = self.spike_times
        spike_amplitudes = self.spike_amplitudes
        spike_depths = self.spike_depths
        spike_templates = self.spike_templates

        #  min, max = np.percentile(spike_amplitudes, (90, 98))
        # spike_amplitudes = np.clip(spike_amplitudes, min, max)
        # This makes the assumption that there will never be different .csv and .tsv files
        # in the same sorter output (this should never happen, there will never even be two).
        # Though can be saved as .tsv, it seems the .csv is also tab formatted as far as pandas is concerned.

        # TODO: this is super weird, can be improved?
        if log_transform_amplitudes:
            spike_amplitudes = np.log(spike_amplitudes)  # TODO: give optional (None, 2 or 10)

        # Calculate the amplitude range for plotting first, so the scale is always the
        # same across all options (e.g. decimation) which helps with interpretability.
        amplitude_range_all_spikes = (
            spike_amplitudes.min(),
            spike_amplitudes.max(),
        )

        # TODO: this is horrible, just create a bool array and index it later
        if exclude_noise:
            spike_times, spike_amplitudes, spike_depths, spike_templates = helpers.exclude_noise(
                self.sorter_path, spike_times, spike_amplitudes, spike_depths, spike_templates
            )

        if decimate:
            spike_times = spike_times[:: decimate]
            spike_amplitudes = spike_amplitudes[:: decimate]
            spike_depths = spike_depths[:: decimate]
            spike_templates = spike_templates[:: decimate]

        if only_include_large_amplitude_spikes:

            spike_times, spike_amplitudes, spike_depths, spike_templates = _filter_large_amplitude_spikes(
                spike_times, spike_amplitudes, spike_depths, spike_templates,
                large_amplitude_only_segment_size
            )

        return spike_times, spike_amplitudes, spike_depths, amplitude_range_all_spikes, spike_templates

    def get_drift_map_plot_interactive(
        self,
        only_include_large_amplitude_spikes=True,
        decimate=False,
        exclude_noise=True,
        log_transform_amplitudes=True,
        large_amplitude_only_segment_size=800,
    ):
        (
            spike_times,
            spike_amplitudes,
            spike_depths,
            amplitude_range_all_spikes,
            spike_templates,
        ) = self._process_data(
            exclude_noise,
            log_transform_amplitudes,
            decimate,
            only_include_large_amplitude_spikes,
            large_amplitude_only_segment_size
        )

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        self.plot = DriftmapPlotWidget(
            spike_times,
            spike_amplitudes,
            spike_depths,
            amplitude_range_all_spikes,
            spike_templates,
            self.templates
        )

        app.exec()


        # histogram
        if False:
            hist_axis.set_xlabel("count")
            raster_axis.set_xlabel("time")
            hist_axis.set_ylabel("y position")

            bin_centers, counts = _compute_activity_histogram(
                spike_amplitudes, spike_depths, weight_histogram_by_amplitude
            )
            hist_axis.plot(counts, bin_centers, color="black", linewidth=1)






    def get_drift_map_plot(self,
        only_include_large_amplitude_spikes=True,
        decimate=False,
        exclude_noise=True,
        log_transform_amplitudes=True,
        large_amplitude_only_segment_size=800,
     ):
        (
            spike_times,
            spike_amplitudes,
            spike_depths,
            amplitude_range_all_spikes,
            _
        ) = self._process_data(
            exclude_noise,
            log_transform_amplitudes,
            decimate,
            only_include_large_amplitude_spikes,
            large_amplitude_only_segment_size
        )

        fig = plt.figure(figsize=(10, 10 * (6 / 8)))
        raster_axis = fig.add_subplot()

        _plot_kilosort_drift_map_raster(
            spike_times,
            spike_amplitudes,
            spike_depths,
            amplitude_range_all_spikes,
            axis=raster_axis,
        )

        return fig

    def get_1d_histogram_plot(
        self,
    ):
        pass

    def get_2d_histogram_plot(self):
        pass

    def get_combined_drift_map_plot(
        self
    ):
        pass




for file in [
    r"C:\Users\Jzimi\Desktop\derivatives\1119617_LSE1_shank12_g0\0\sorter_output",
]:
    plotter = DriftMapView(
        file
    )

    fig = plotter.get_drift_map_plot_interactive(
        only_include_large_amplitude_spikes=True,  # exclude amplitude outliers? maybe just do this instead of doing this segmented way.
        decimate=False,
        exclude_noise=False,
        log_transform_amplitudes=False
    )




if False:
    plot = get_drift_map_plot(
        r"C:\Users\Joe\PycharmProjects\viewephys3\kilosort4_output\sorter_output",
        only_include_large_amplitude_spikes=True,
        add_histogram_plot=True,
        weight_histogram_by_amplitude=True,
        decimate=False,
        exclude_noise=True
    )

    plt.show()
