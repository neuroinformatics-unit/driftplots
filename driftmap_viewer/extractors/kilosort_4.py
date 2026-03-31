from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np


def compute_spike_amplitudes(templates, spike_templates, amplitudes):
    # This is based on https://github.com/MouseLand/Kilosort/issues/804, need to double check it
    # TODO: these amplitudes are not scaled by gain / offset, but this doesn't matter for our purposes
    template_ptp = np.max(templates, axis=1) - np.min(templates, axis=1)
    template_max_peaks = np.max(template_ptp, axis=1)
    spike_amplitudes = template_max_peaks[spike_templates] * amplitudes
    return spike_amplitudes


def get_spikes_info_ks4(
    sorter_output: Path,
) -> tuple[np.ndarray, ...]:

    # TODO: kept_spikes is not always the same size as other spike data
    # Currently not used for loading
    # kept_spikes = np.load(sorter_output / "kept_spikes.npy")

    spike_times = np.load(sorter_output / "spike_times.npy")
    amplitudes = np.load(sorter_output / "amplitudes.npy")
    spike_depths = np.load(sorter_output / "spike_positions.npy")[:, 1]
    spike_templates = np.load(
        sorter_output / "spike_templates.npy"
    )  # rename spike_tempaltes_idx?

    templates = np.load(sorter_output / "templates.npy")  # rename unwhiten?
    channel_positions = np.load(sorter_output / "channel_positions.npy")
    spike_amplitudes = compute_spike_amplitudes(templates, spike_templates, amplitudes)

    return (
        spike_times,
        spike_amplitudes,
        spike_depths,
        spike_templates,
        templates,
        channel_positions,
    )
