from __future__ import annotations

from pathlib import Path

import numpy as np
import spikeinterface as si

from driftmap_viewer import DataLoader


def get_amplitudes(
    list_of_path_or_analyzer: list[Path | si.SortingAnalyzer],
    exclude_noise: bool = False,
    concatenate: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """Load and concatenate amplitudes.npy from multiple sorter output paths.

    Parameters
    ----------
    list_of_path_or_analyzer
        List of sorter output directories, each containing amplitudes.npy.
    concatenate
        If ``True``, concatenate all amplitudes into a single array.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Concatenated amplitudes from all paths.
    """
    all_spike_amplitudes = []

    for path_or_analyzer in list_of_path_or_analyzer:
        loader = DataLoader(path_or_analyzer)

        processed_data = loader.get_processed_data(exclude_noise)

        all_spike_amplitudes.append(processed_data.spike_amplitudes)

    if concatenate:
        all_spike_amplitudes = np.concatenate(all_spike_amplitudes)

    return all_spike_amplitudes
