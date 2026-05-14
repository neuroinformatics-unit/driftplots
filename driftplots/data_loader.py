from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import spikeinterface as si

from driftplots.data_model import DataModel
from driftplots.extractors import (
    analyzer_helpers,
    kilosort1_3,
    kilosort4,
    kilosort_helpers,
)


class DataLoader:
    """"""

    def __init__(
        self, path_or_analyzer: Path | si.SortingAnalyzer, verbose: bool
    ) -> None:
        """ """

        self._print(f"Loading data from {path_or_analyzer}...", verbose)

        # Get the data loading function depending on whether
        # the input is a SortingAnalyzer or Kilosort output.
        func: Callable
        if isinstance(path_or_analyzer, si.SortingAnalyzer):
            self.path_or_analyzer = path_or_analyzer
            func = analyzer_helpers.get_sorting_analyzer
        else:
            self.path_or_analyzer = Path(path_or_analyzer)
            ks_version = kilosort_helpers.get_ks_version(self.path_or_analyzer)
            func = (
                kilosort4.get_spikes_info_ks4
                if ks_version == "kilosort4"
                else kilosort1_3.get_spikes_info_ks1_3
            )

        # Load the required data and check sizes match (one entry per spike)
        (
            self._spike_times,
            self._spike_amplitudes,
            self._spike_depths,
            self._spike_templates,
            self.templates,
            self.channel_locations,
        ) = func(self.path_or_analyzer)

        assert (
            self._spike_times.size
            == self._spike_amplitudes.size
            == self._spike_depths.size
            == self._spike_templates.size
        )
        assert self.channel_locations.shape[0] > self.channel_locations.shape[1]

        self._spike_times.flags.writeable = False
        self._spike_amplitudes.flags.writeable = False
        self._spike_depths.flags.writeable = False
        self._spike_templates.flags.writeable = False
        self.templates.flags.writeable = False
        self.channel_locations.flags.writeable = False

        self._print(f"Loaded {self._spike_times.size} spikes.", verbose)

    def get_processed_data(
        self,
        good_units_only,
        decimate,
        filter_amplitude_mode,
        filter_amplitude_values,
        verbose: bool,
    ):
        """Filter and subsample the loaded spike data.

        Parameters
        ----------
        good_units_only
            If ``True``, only spikes belonging to "good" units are kept.
        decimate :
            Keep every *n*-th spike. Applied first to reduce the dataset
            before noise/amplitude filters. ``False`` disables decimation.
        filter_amplitude_mode :
            How ``filter_amplitude_values`` is interpreted.
            ``None`` disables amplitude filtering.
        filter_amplitude_values :
            (low, high) bounds. Interpreted as percentile ranks or
            absolute amplitude values depending on ``filter_amplitude_mode``.
        verbose :
            If `True`, messages are printed.

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

        # First, exclude spikes from units that are not labelled "good"
        if good_units_only:
            if isinstance(self.path_or_analyzer, si.SortingAnalyzer):
                keep_bool_mask = analyzer_helpers.get_good_unit_mask(
                    good_units_only, spike_templates, self.path_or_analyzer
                )
            else:
                keep_bool_mask = kilosort_helpers.get_good_unit_mask(
                    spike_templates, self.path_or_analyzer
                )

            self._print(
                f"Keeping good spikes only. {keep_bool_mask.sum()} spikes remaining.",
                verbose,
            )

        # Next, filter spikes based on amplitude
        if filter_amplitude_mode is not None:
            assert filter_amplitude_mode in ["percentile", "absolute"]

            if filter_amplitude_mode == "percentile":
                amps_for_percentile = (
                    spike_amplitudes
                    if keep_bool_mask is None
                    else spike_amplitudes[keep_bool_mask]
                )
                min_val, max_val = np.percentile(
                    amps_for_percentile, filter_amplitude_values
                )
            else:
                min_val, max_val = filter_amplitude_values

            if keep_bool_mask is None:
                keep_bool_mask = np.ones(spike_amplitudes.size, dtype=bool)

            keep_bool_mask[spike_amplitudes < min_val] = False
            keep_bool_mask[spike_amplitudes > max_val] = False

            self._print(
                "Excluded spikes based on amplitude. "
                f"{keep_bool_mask.sum()} spikes remaining.",
                verbose,
            )

        # mask good_units_only / filtered amplitudes
        if keep_bool_mask is not None:
            spike_times = spike_times[keep_bool_mask]
            spike_amplitudes = spike_amplitudes[keep_bool_mask]
            spike_depths = spike_depths[keep_bool_mask]
            spike_templates = spike_templates[keep_bool_mask]

        if decimate:
            num_spikes = spike_times.size
            if decimate == "estimate":
                ideal_num_spikes = 100_000
                decimation_factor = int(num_spikes // ideal_num_spikes)
            else:
                decimation_factor = int(decimate)

            if decimation_factor > 1:
                spike_times = spike_times[::decimation_factor]
                spike_amplitudes = spike_amplitudes[::decimation_factor]
                spike_depths = spike_depths[::decimation_factor]
                spike_templates = spike_templates[::decimation_factor]

                self._print(
                    f"Decimated by factor {decimation_factor}. "
                    f"{spike_times.size} spikes remaining.",
                    verbose,
                )

        return DataModel(
            spike_times,
            spike_amplitudes,
            spike_depths,
            spike_templates,
            self.templates,
            self.channel_locations,
        )

    def _print(self, message, verbose):
        if verbose:
            print(message)
