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
import spikeinterface as si


class DataModel:
    def __init__(
            self,
            sorter,
            spike_times,
            spike_amplitudes,
            spike_depths,
            spike_templates,
            templates,
            channel_locations
    ):
        self.sorter = sorter
        self.spike_times = spike_times
        self.spike_depths = spike_depths
        self.spike_amplitudes = spike_amplitudes
        self.spike_templates = spike_templates
        self.templates = templates
        self.channel_locations = channel_locations

    def get_scatter_data(self):
        return self.spike_times, self.spike_depths, self.spike_amplitudes

    def get_template_id(self, spike_idx):
        return self.spike_templates[spike_idx]

    def get_template_heatmap(self, spike_index, view_mode):
        """"""
        template_idx = self.spike_templates[spike_index]
        template = self.templates[template_idx, :, :]

        # find the shank with signal, in case multi-shank probe was sorted
        mid_idx = int(template.shape[0] / 2)
        chan_with_signal = np.where(template[mid_idx, :] != 0)  # arbitrary cutoff, should check more
        positions_with_signal = self.channel_locations[chan_with_signal]
        shank_contacts_with_signal = np.unique(positions_with_signal[:, 0])

        if len(shank_contacts_with_signal) != 2:
            warnings.warn("This spikes template has signal on more than one shank.")

        shank_select = np.zeros(self.channel_locations.shape[0], dtype=bool)
        for pos in shank_contacts_with_signal:
            shank_select = np.logical_or(shank_select, self.channel_locations[:, 0] == pos)

        sort_idx = np.argsort(self.channel_locations[shank_select, 1], axis=0)

        # TODO: could include shank it is on, will need to actually
        template = template[:, shank_select]
        template = template[:, sort_idx]

        if view_mode == "heatmap_all_channels":
            template = template.copy()  # TODO: check if this is necessary
            template[:, template[mid_idx, :] == 0] = np.nan
        else:
            contains_data_idx = np.where(template[mid_idx, :] != 0)[0]
            template = template[:, contains_data_idx]

        return template


def get_sorting_analyzer(analyzer):

    random_spike_indices = analyzer.get_extension("random_spikes").data["random_spikes_indices"]
    spike_vector = analyzer.sorting.to_spike_vector()
    spike_times = spike_vector["sample_index"][random_spike_indices] / analyzer.sorting.get_sampling_frequency()
    spike_amplitudes = np.abs(analyzer.get_extension("spike_amplitudes").data["amplitudes"])  # TODO: THIS!
    spike_depths = analyzer.get_extension("spike_locations").data["spike_locations"]["y"]
    spike_templates = spike_vector["unit_index"][random_spike_indices]

    templates_dict = analyzer.get_extension("templates").data
    all_template_keys = templates_dict.keys()
    template_key = list(all_template_keys)[0]

    if len(all_template_keys) != 1:
        warnings.warn(f"Multiple template calculation methods detected. Using {template_key}")

    templates = analyzer.get_extension("templates").data[template_key]
    channel_locations = analyzer.get_channel_locations()

    return spike_times, spike_amplitudes, spike_depths, spike_templates, templates, channel_locations

class DataLoader:
    """"""
    def __init__(self, folder_path: Path) -> None:
        """
        """
        if isinstance(folder_path, si.SortingAnalyzer):
            func = get_sorting_analyzer

            self.sorter = "TODO"
        else:
            folder_path = Path(folder_path)
            ks_version = self._get_ks_version(folder_path)

            func = kilosort_4.get_spikes_info_ks4 if ks_version == "kilosort4" else  kilosort1_3.get_spikes_info_ks1_3

            self.sorter = ks_version

        (
            self._spike_times,
            self._spike_amplitudes,
            self._spike_depths,
            self._spike_templates,
            self.templates,
            self.channel_locations,
        ) = func(
            folder_path
        )

        assert self._spike_times.size == self._spike_amplitudes.size == self._spike_depths.size == self._spike_templates.size
        assert self.channel_locations.shape[0] > self.channel_locations.shape[1]

        self._spike_times.flags.writeable = False
        self._spike_amplitudes.flags.writeable = False
        self._spike_depths.flags.writeable = False
        self._spike_templates.flags.writeable = False
        self.templates.flags.writeable = False
        self.channel_locations.flags.writeable = False

    def _get_ks_version(self, sorter_path):
        """
        """
        log_file = list(sorter_path.glob("kilosort*.log"))
        assert len(log_file) == 1

        return Path(log_file[0]).name.split(".")[0]

    def get_processed_data(
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
        spike_times = self._spike_times
        spike_amplitudes = self._spike_amplitudes
        spike_depths = self._spike_depths
        spike_templates = self._spike_templates

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

        return DataModel(
            self.sorter,
            spike_times,
            spike_amplitudes,
            spike_depths,
            spike_templates,
            self.templates,
            self.channel_locations
        )
