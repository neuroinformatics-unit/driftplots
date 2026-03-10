from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import helpers


def get_spikes_info_ks4(
    sorter_output: Path,
): # -> tuple[np.ndarray, ...]:

    spike_times = np.load(sorter_output / "spike_times.npy")
    spike_amplitudes = np.load(sorter_output / "amplitudes.npy")
    spike_depths = np.load(sorter_output / "spike_positions.npy")[:, 1]

    spike_templates = np.load(sorter_output / "spike_templates.npy")
    templates = np.load(sorter_output / "templates.npy")

    # MASSIVE TODO: It is still not clear if these templates are white or not...

    return spike_times, spike_amplitudes, spike_depths, spike_templates, templates
