from pathlib import Path

import numpy as np
import spikeinterface as si

from driftmap_viewer.data_model import DataModel
from driftmap_viewer.extractors import kilosort1_3, kilosort_4, kilosort_helpers
from driftmap_viewer.extractors import analyzer_helpers


class DataLoader:
    """"""

    def __init__(self, path_or_analyzer: Path) -> None:
        """ """
        self.path_or_analyzer = path_or_analyzer

        # Get the data loading function depending on if we are analyzer or kilosort output
        if isinstance(path_or_analyzer, si.SortingAnalyzer):
            func = analyzer_helpers.get_sorting_analyzer
        else:
            ks_version = kilosort_helpers.get_ks_version(
                Path(path_or_analyzer)
            )
            func = (
                kilosort_4.get_spikes_info_ks4
                if ks_version == "kilosort4"
                else kilosort1_3.get_spikes_info_ks1_3
            )

        # Load the required data and check sizes match (one entry per spike)
        (
            self._spike_times,
            self._spike_amplitudes,
            self._spike_depths,
            self._spike_clusters,
            self.templates,
            self.channel_locations,
        ) = func(path_or_analyzer)

        assert (
            self._spike_times.size
            == self._spike_amplitudes.size
            == self._spike_depths.size
            == self._spike_clusters.size
        )
        assert self.channel_locations.shape[0] > self.channel_locations.shape[1]

        self._spike_times.flags.writeable = False
        self._spike_amplitudes.flags.writeable = False
        self._spike_depths.flags.writeable = False
        self._spike_clusters.flags.writeable = False
        self.templates.flags.writeable = False
        self.channel_locations.flags.writeable = False

    def get_processed_data(
        self, exclude_noise, decimate, filter_amplitude_mode, filter_amplitude_values
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
        spike_clusters : np.ndarray
            Filtered copies (views when no filtering is needed) of the
            corresponding instance arrays.
        """
        # Select a view for now, this may be copied depending on options (e.g. decimate)
        spike_times = self._spike_times
        spike_amplitudes = self._spike_amplitudes
        spike_depths = self._spike_depths
        spike_clusters = self._spike_clusters

        keep_bool_mask = None

        # First, exclude spikes from units labeled as "noise"
        if exclude_noise:
            if isinstance(self.path_or_analyzer, si.SortingAnalyzer):
                keep_bool_mask = ~analyzer_helpers.get_noise_mask(self.path_or_analyzer)
            else:
                keep_bool_mask = ~kilosort_helpers.get_noise_mask(spike_clusters, self.path_or_analyzer)

        # Next, filter spikes based on amplitude
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

        # mask exclude_noise / filtered amplitudes
        if keep_bool_mask is not None:
            spike_times = spike_times[keep_bool_mask]
            spike_amplitudes = spike_amplitudes[keep_bool_mask]
            spike_depths = spike_depths[keep_bool_mask]
            spike_clusters = spike_clusters[keep_bool_mask]

        if decimate:
            spike_times = spike_times[::decimate]
            spike_amplitudes = spike_amplitudes[::decimate]
            spike_depths = spike_depths[::decimate]
            spike_clusters = spike_clusters[::decimate]

        return DataModel(
            spike_times,
            spike_amplitudes,
            spike_depths,
            spike_clusters,
            self.templates,
            self.channel_locations,
        )
