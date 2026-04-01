from pathlib import Path

import numpy as np
import spikeinterface as si

from driftmap_viewer.extractors import (  # TODO: remove underscorw?
    kilosort_4,
    kilosort_helpers,
)

def get_amplitudes(
    list_of_path_or_analyzer: list[Path | si.SortingAnalyzer], concatenate=False
) -> np.ndarray:
    """Load and concatenate amplitudes.npy from multiple sorter output paths.

    Parameters
    ----------
    paths : list of Path
        List of sorter output directories, each containing amplitudes.npy.

    Returns
    -------
    np.ndarray
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
