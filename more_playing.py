from driftmapviewer_new import get_drift_map_plot, _plot_kilosort_drift_map_raster, _filter_large_amplitude_spikes
import matplotlib.pyplot as plt
import kilosort1_3
import kilosort_4
import helpers
from pathlib import Path
import numpy as np
from PySide6 import QtWidgets


class DriftMapView():
    def __init__(self, sorter_path):
        self.sorter_path = Path(sorter_path)

        log_file = list(self.sorter_path.glob("kilosort*.log"))
        assert len(log_file) == 1
        self.ks_version = Path(log_file[0]).name.split(".")[0]

        func = kilosort_4.get_spikes_info_ks4 if self.ks_version == "kilosort4" else kilosort1_3.get_spikes_info_ks1_3

        (
            self.spike_times,
            self.spike_amplitudes,
            self.spike_depths,
            self.spike_templates,
            self.templates
        ) = func(self.sorter_path)

        self.spike_times.flags.writeable = False
        self.spike_amplitudes.flags.writeable = False
        self.spike_depths.flags.writeable = False
        self.spike_templates.flags.writeable = False
        self.templates.flags.writeable = False

    def _process_data(
        self,
        exclude_noise,
        log_transform_amplitudes,
        decimate,
        only_include_large_amplitude_spikes,
        large_amplitude_only_segment_size,
    ):
        spike_times = self.spike_times
        spike_amplitudes = self.spike_amplitudes
        spike_depths = self.spike_depths
        spike_templates = self.spike_templates

        if log_transform_amplitudes:
            spike_amplitudes = np.log(spike_amplitudes)

        amplitude_range_all_spikes = (
            spike_amplitudes.min(),
            spike_amplitudes.max(),
        )

        if exclude_noise:
            spike_times, spike_amplitudes, spike_depths, spike_templates = helpers.exclude_noise(
                self.sorter_path, spike_times, spike_amplitudes, spike_depths, spike_templates
            )

        if decimate:
            spike_times = spike_times[::decimate]
            spike_amplitudes = spike_amplitudes[::decimate]
            spike_depths = spike_depths[::decimate]
            spike_templates = spike_templates[::decimate]

        if only_include_large_amplitude_spikes:
            spike_times, spike_amplitudes, spike_depths, spike_templates = _filter_large_amplitude_spikes(
                spike_times, spike_amplitudes, spike_depths, spike_templates,
                large_amplitude_only_segment_size,
            )

        return spike_times, spike_amplitudes, spike_depths, amplitude_range_all_spikes, spike_templates

    def get_session_data(
        self,
        only_include_large_amplitude_spikes=True,
        decimate=False,
        exclude_noise=True,
        log_transform_amplitudes=True,
        large_amplitude_only_segment_size=800,
    ) -> dict:
        """Return a data dict suitable for passing to MultiSessionDriftmapWidget."""
        spike_times, spike_amplitudes, spike_depths, amplitude_range_all_spikes, spike_templates = self._process_data(
            exclude_noise,
            log_transform_amplitudes,
            decimate,
            only_include_large_amplitude_spikes,
            large_amplitude_only_segment_size,
        )
        return dict(
            spike_times=spike_times,
            spike_amplitudes=spike_amplitudes,
            spike_depths=spike_depths,
            amplitude_range_all_spikes=amplitude_range_all_spikes,
            spike_templates=spike_templates,
            templates=self.templates,
        )

    def get_drift_map_plot(
        self,
        only_include_large_amplitude_spikes=True,
        decimate=False,
        exclude_noise=True,
        log_transform_amplitudes=True,
        large_amplitude_only_segment_size=800,
    ):
        spike_times, spike_amplitudes, spike_depths, amplitude_range_all_spikes, _ = self._process_data(
            exclude_noise,
            log_transform_amplitudes,
            decimate,
            only_include_large_amplitude_spikes,
            large_amplitude_only_segment_size,
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


from multi_session_driftmap_widget import MultiSessionDriftmapWidget

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    session_data = []
    for file in [
        r"Y:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\joe\scratch\derivatives\BG_046_24062025\shank_0\sorting\no_motion\sorter_output",
    ]:
        sorter_output = Path(file)
        total_spikes = np.load(sorter_output / "spike_times.npy").size
        kept_spikes = int(np.load(sorter_output / "kept_spikes.npy").sum())
        print(f"[{sorter_output.name}] total spikes: {total_spikes:,}")
        print(f"[{sorter_output.name}] kept spikes:  {kept_spikes:,}")

        plotter = DriftMapView(file)
        session_data.append(plotter.get_session_data(
            only_include_large_amplitude_spikes=True,
            decimate=False,
            exclude_noise=False,
            log_transform_amplitudes=False,
        ))

    multi = MultiSessionDriftmapWidget(session_data)
    app.exec()


if __name__ == "__main__":
    main()
