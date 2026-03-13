from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import helpers


def get_spikes_info_ks4(
    sorter_output: Path,
): # -> tuple[np.ndarray, ...]:

    # TODO: kept_spikes is not always the same size as other spike data
    # Currently not used for loading
    # kept_spikes = np.load(sorter_output / "kept_spikes.npy")

    spike_times = np.load(sorter_output / "spike_times.npy")
    spike_amplitudes = np.load(sorter_output / "amplitudes.npy")
    spike_depths = np.load(sorter_output / "spike_positions.npy")[:, 1]
    spike_templates = np.load(sorter_output / "spike_templates.npy")  # rename spike_tempaltes_idx?

    templates = np.load(sorter_output / "templates.npy") # rename unwihten?

    inv_white_mat = np.load(sorter_output / "whitening_mat_inv.npy")

    for i in range(templates.shape[0]):
        templates[i, :, :] = templates[i, :, :] @ inv_white_mat

    return spike_times, spike_amplitudes, spike_depths, spike_templates, templates
