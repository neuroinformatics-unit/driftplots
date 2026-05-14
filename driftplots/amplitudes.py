from __future__ import annotations

from pathlib import Path

import numpy as np
import spikeinterface as si

from driftplots.data_loader import DataLoader


def get_amplitudes(
    list_of_path_or_analyzer: list[Path | si.SortingAnalyzer],
    exclude_noise: bool = False,
    concatenate: bool = False,
    verbose: bool = True,
) -> np.ndarray | list[np.ndarray]:
    """Return spike amplitudes from one or more sorter outputs.

    Parameters
    ----------
    list_of_path_or_analyzer
        Kilosort output directory paths or SpikeInterface ``SortingAnalyzer``
        objects to load amplitudes from. Can be a mix of both.
    exclude_noise
        If ``True``, remove spikes belonging to clusters labelled "noise"
        before returning amplitudes.
    concatenate
        If ``True``, concatenate all per-session amplitude arrays into a
        single 1-D array.  If ``False`` (default), return a list with one
        array per session.
    verbose :
            If `True`, messages are printed.

    Returns
    -------
    np.ndarray or list of np.ndarray
        When ``concatenate=True``, a single 1-D array of all spike
        amplitudes.  When ``concatenate=False``, a list of 1-D arrays,
        one per entry in ``list_of_path_or_analyzer``.
    """
    all_spike_amplitudes = []

    for path_or_analyzer in list_of_path_or_analyzer:
        loader = DataLoader(path_or_analyzer)

        processed_data = loader.get_processed_data(
            exclude_noise,
            decimate=False,
            filter_amplitude_mode=None,
            filter_amplitude_values=None,
            verbose=verbose,
        )

        all_spike_amplitudes.append(processed_data.spike_amplitudes)

    if concatenate:
        all_spike_amplitudes = np.concatenate(all_spike_amplitudes)

    return all_spike_amplitudes
