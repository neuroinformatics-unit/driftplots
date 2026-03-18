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

    return spike_times, spike_amplitudes, spike_depths
