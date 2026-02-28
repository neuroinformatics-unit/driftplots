"""Utilities to load spike positions/amplitudes and template waveforms for Kilosort outputs.

This module is meant to be *standalone* (no package-relative imports).
It works with the sibling modules:
  - kilosort_4.py
  - kilosort1_3.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

import kilosort_4
import kilosort1_3


def load_spikes(sorter_output: str | Path, ks_version: str = "kilosort4"):
    """Return (spike_times_s, spike_amplitudes, spike_depths_um)."""
    sorter_output = Path(sorter_output)

    if ks_version.lower() in ("ks4", "kilosort4", "kilosort_4"):
        spike_times, spike_amplitudes, spike_depths = kilosort_4.get_spikes_info_ks4(sorter_output)
        # KS4 spike_times are usually in samples; convert to seconds if we can.
        # If "params.py" exists, use it; otherwise return raw.
        try:
            from spikeinterface.core import read_python
            params = read_python(sorter_output / "params.py")
            sr = float(params.get("sample_rate", 1.0))
            spike_times = spike_times.squeeze() / sr
        except Exception:
            spike_times = spike_times.squeeze()

        return spike_times.squeeze(), spike_amplitudes.squeeze(), spike_depths.squeeze()

    if ks_version.lower() in ("ks1", "ks2", "ks3", "kilosort1", "kilosort2", "kilosort3", "kilosort1_3"):
        spike_times, spike_amplitudes, spike_depths = kilosort1_3.get_spikes_info_ks1_3(sorter_output)
        return spike_times.squeeze(), spike_amplitudes.squeeze(), spike_depths.squeeze()

    raise ValueError(f"Unknown ks_version={ks_version!r}")


def load_templates(sorter_output: str | Path, ks_version: str = "kilosort4"):
    """Return (spike_templates, template_waveforms_mainchan).

    - spike_templates: (n_spikes,) template id per spike
    - template_waveforms_mainchan: (n_templates, n_samples) waveform on each template's max-energy channel
    """
    sorter_output = Path(sorter_output)

    if ks_version.lower() in ("ks4", "kilosort4", "kilosort_4"):
        return _load_templates_ks4(sorter_output)

    if ks_version.lower() in ("ks1", "ks2", "ks3", "kilosort1", "kilosort2", "kilosort3", "kilosort1_3"):
        return _load_templates_ks1_3(sorter_output)

    raise ValueError(f"Unknown ks_version={ks_version!r}")


def _load_templates_ks4(sorter_output: Path):
    spike_templates = np.load(sorter_output / "spike_templates.npy").squeeze()
    templates = np.load(sorter_output / "templates.npy")  # (n_templates, n_samples, n_channels)

    # main channel per template = channel with max absolute deflection
    # abs max over time -> per channel; then argmax over channels
    per_ch_peak = np.max(np.abs(templates), axis=1)  # (n_templates, n_channels)
    max_site = np.argmax(per_ch_peak, axis=1)        # (n_templates,)

    waveforms = np.empty(templates.shape[:2], dtype=templates.dtype)  # (n_templates, n_samples)
    for i in range(templates.shape[0]):
        waveforms[i, :] = templates[i, :, max_site[i]]

    return spike_templates.astype(np.int64, copy=False), waveforms


def _load_templates_ks1_3(sorter_output: Path):
    # Reuse internals already present in kilosort1_3.py
    params = kilosort1_3._load_ks_dir(sorter_output, load_pcs=False)

    # This existing helper already unwhitens and computes a main-channel waveform per template.
    # It returns waveforms as (n_templates, n_samples)
    *_, waveforms = kilosort1_3._template_positions_amplitudes(
        params["templates"],
        params["whitening_matrix_inv"],
        params["channel_positions"][:, 1],
        params["spike_templates"],
        params["temp_scaling_amplitudes"],
    )

    return params["spike_templates"].astype(np.int64, copy=False), waveforms
