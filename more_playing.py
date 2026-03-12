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

    @staticmethod
    def _apply_boolean_mask(spike_times, spike_amplitudes, spike_depths, spike_templates, mask):
        return (
            spike_times[mask],
            spike_amplitudes[mask],
            spike_depths[mask],
            spike_templates[mask],
        )

    @staticmethod
    def _compute_threshold_mask(
        spike_amplitudes,
        amplitude_percentile_min=None,
        amplitude_raw_min=None,
    ):
        keep = np.ones(spike_amplitudes.shape[0], dtype=bool)

        if amplitude_percentile_min is not None:
            if not 0 <= amplitude_percentile_min <= 100:
                raise ValueError("`amplitude_percentile_min` must be in [0, 100].")
            percentile_threshold = np.percentile(spike_amplitudes, amplitude_percentile_min)
            keep &= spike_amplitudes >= percentile_threshold

        if amplitude_raw_min is not None:
            keep &= spike_amplitudes >= amplitude_raw_min

        return keep

    @staticmethod
    def _compute_global_std_mask(spike_amplitudes, std_multiplier=1.5):
        threshold = np.mean(spike_amplitudes) + std_multiplier * np.std(spike_amplitudes, ddof=1)
        return spike_amplitudes >= threshold

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
        large_amplitude_only_segment_size,
        amplitude_filter_mode="segment_std",
        amplitude_percentile_min=None,
        amplitude_raw_min=None,
        amplitude_global_std_multiplier=1.5,
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

        if only_include_large_amplitude_spikes and amplitude_filter_mode != "none":
            if amplitude_filter_mode == "segment_std":
                spike_times, spike_amplitudes, spike_depths, spike_templates = _filter_large_amplitude_spikes(
                    spike_times, spike_amplitudes, spike_depths, spike_templates,
                    large_amplitude_only_segment_size,
                )
            elif amplitude_filter_mode == "global_std":
                keep = self._compute_global_std_mask(spike_amplitudes, amplitude_global_std_multiplier)
                spike_times, spike_amplitudes, spike_depths, spike_templates = self._apply_boolean_mask(
                    spike_times, spike_amplitudes, spike_depths, spike_templates, keep
                )
            elif amplitude_filter_mode == "percentile":
                if amplitude_percentile_min is None:
                    raise ValueError(
                        "`amplitude_filter_mode='percentile'` requires `amplitude_percentile_min`."
                    )
                keep = self._compute_threshold_mask(
                    spike_amplitudes,
                    amplitude_percentile_min=amplitude_percentile_min,
                    amplitude_raw_min=None,
                )
                spike_times, spike_amplitudes, spike_depths, spike_templates = self._apply_boolean_mask(
                    spike_times, spike_amplitudes, spike_depths, spike_templates, keep
                )
            elif amplitude_filter_mode == "raw":
                if amplitude_raw_min is None:
                    raise ValueError("`amplitude_filter_mode='raw'` requires `amplitude_raw_min`.")
                keep = self._compute_threshold_mask(
                    spike_amplitudes,
                    amplitude_percentile_min=None,
                    amplitude_raw_min=amplitude_raw_min,
                )
                spike_times, spike_amplitudes, spike_depths, spike_templates = self._apply_boolean_mask(
                    spike_times, spike_amplitudes, spike_depths, spike_templates, keep
                )
            else:
                raise ValueError(
                    "Unknown `amplitude_filter_mode`. Use one of: "
                    "'none', 'segment_std', 'global_std', 'percentile', 'raw'."
                )

        # Hard thresholds can be applied on top of any filtering mode.
        if amplitude_percentile_min is not None or amplitude_raw_min is not None:
            keep = self._compute_threshold_mask(
                spike_amplitudes,
                amplitude_percentile_min=amplitude_percentile_min,
                amplitude_raw_min=amplitude_raw_min,
            )
            spike_times, spike_amplitudes, spike_depths, spike_templates = self._apply_boolean_mask(
                spike_times, spike_amplitudes, spike_depths, spike_templates, keep
            )

        return spike_times, spike_amplitudes, spike_depths, amplitude_range_all_spikes, spike_templates

    def get_drift_map_plot_interactive(
        self,
        only_include_large_amplitude_spikes=True,
        decimate=False,
        exclude_noise=True,
        log_transform_amplitudes=True,
        large_amplitude_only_segment_size=800,
        amplitude_filter_mode="segment_std",
        amplitude_percentile_min=None,
        amplitude_raw_min=None,
        amplitude_global_std_multiplier=1.5,
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
            large_amplitude_only_segment_size,
            amplitude_filter_mode,
            amplitude_percentile_min,
            amplitude_raw_min,
            amplitude_global_std_multiplier,
        )

        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        self.plot = DriftmapPlotWidget(
            spike_times,
            spike_amplitudes,
            spike_depths,
            amplitude_range_all_spikes,
            spike_templates,
            self.templates
        )

        return self.plot



        # histogram
        if False:
            hist_axis.set_xlabel("count")
            raster_axis.set_xlabel("time")
            hist_axis.set_ylabel("y position")

            bin_centers, counts = _compute_activity_histogram(
                spike_amplitudes, spike_depths, weight_histogram_by_amplitude
            )
            hist_axis.plot(counts, bin_centers, color="black", linewidth=1)






    def get_drift_map_plot_matplotlib(self,
        only_include_large_amplitude_spikes=True,
        decimate=False,
        exclude_noise=True,
        log_transform_amplitudes=True,
        large_amplitude_only_segment_size=800,
        amplitude_filter_mode="segment_std",
        amplitude_percentile_min=None,
        amplitude_raw_min=None,
        amplitude_global_std_multiplier=1.5,

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
            large_amplitude_only_segment_size,
            amplitude_filter_mode,
            amplitude_percentile_min,
            amplitude_raw_min,
            amplitude_global_std_multiplier,
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

from PySide6 import QtWidgets
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

panels = []
for file in [
    # r"Y:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\joe\scratch\derivatives\BG_046_23062025\shank_0\sorting\motion\sorter_output",
    # r"Y:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\joe\scratch\derivatives\BG_046_24062025\shank_0\sorting\no_motion\sorter_output",
    # r"Y:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\joe\scratch\derivatives\BG_046_25062025\shank_0\sorting\no_motion\sorter_output"
    r"X:\aeon\dj_store\ephys-processed\social-ephys0.1-aeon3\ephys_blocks\2024-06-04T11-00-00_2024-06-04T14-00-00\0-95\kilosort4_400\spike_sorting\sorter_output",
    r"X:\aeon\dj_store\ephys-processed\social-ephys0.1-aeon3\ephys_blocks\2024-06-04T13-00-00_2024-06-04T16-00-00\0-95\kilosort4_400\spike_sorting\sorter_output"
]:
    plotter = DriftMapView(
        file
    )

    # TODO: plot num spikes loaded...
    fig = plotter.get_drift_map_plot_interactive(
        only_include_large_amplitude_spikes=True,
        decimate=False,
        exclude_noise=False,
        log_transform_amplitudes=False,
        amplitude_filter_mode="percentile",      # options: none, segment_std, global_std, percentile, raw
        amplitude_percentile_min=75.0,           # hard cutoff (drop bottom 30% amplitudes)
        amplitude_raw_min=4.0,                   # hard absolute cutoff (applied in addition)
    )

    panels.append(fig)

from multi_session_drift_map import MultiSessionDriftmapWidget
multi = MultiSessionDriftmapWidget(panels)

app.exec()


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
