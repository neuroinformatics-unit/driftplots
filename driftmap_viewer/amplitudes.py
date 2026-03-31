from pathlib import Path

import numpy as np
import spikeinterface as si

from driftmap_viewer.extractors import analyzer as dv_analyzer  # TODO: RENAME
from driftmap_viewer.extractors import (  # TODO: remove underscorw?
    kilosort_4,
    kilosort_helpers,
)


# TODO: should probably let exclude noise, kept spikes! check noise etc.
# TODO: Expose absolute
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
        if isinstance(path_or_analyzer, si.SortingAnalyzer):
            analyzer = path_or_analyzer
            spike_amplitudes = dv_analyzer.get_amplitudes(analyzer)
        else:
            path_ = path_or_analyzer
            ks_version = kilosort_helpers.get_ks_version(path_)

            if ks_version == "kilosort4":
                amplitudes = np.load(path_ / "amplitudes.npy")
                spike_templates = np.load(
                    path_ / "spike_templates.npy"
                )  # rename spike_tempaltes_idx?
                templates = np.load(path_ / "templates.npy")  # rename unwhiten?

                spike_amplitudes = kilosort_4.compute_spike_amplitudes(
                    templates, spike_templates, amplitudes
                )
            else:
                templates = np.load(path_ / "templates.npy")
                whitening_matrix_inv = np.load(path_ / "whitening_mat_inv.npy")
                spike_templates = np.load(path_ / "spike_templates.npy").squeeze()
                temp_scaling_amplitudes = np.load(path_ / "amplitudes.npy").squeeze()

                # Compute amplitudes, scale if required and drop un-localised spikes before returning.
                spike_amplitudes, _ = _template_positions_amplitudes(
                    templates,
                    whitening_matrix_inv,
                    spike_templates,
                    temp_scaling_amplitudes,
                )

        all_spike_amplitudes.append(spike_amplitudes)

    if concatenate:
        all_spike_amplitudes = np.concatenate(all_spike_amplitudes)

    return all_spike_amplitudes
