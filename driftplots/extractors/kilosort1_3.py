from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
from spikeinterface.core import read_python


def get_spikes_info_ks1_3(
    sorter_output: str | Path,
) -> tuple[np.ndarray, ...]:
    """
    Compute the amplitude and depth of all detected spikes from the Kilosort output.

    This function was ported from Nick Steinmetz's `spikes` repository
    MATLAB code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    sorter_output : str | Path
        Path to the Kilosort run sorting output.

    Returns
    -------
    spike_times : np.ndarray
        (num_spikes,) array of spike times.
    spike_amplitudes : np.ndarray
        (num_spikes,) array of corresponding spike amplitudes.
    spike_depths : np.ndarray
        (num_spikes,) array of corresponding depths (probe y-axis location).

    Notes
    -----
    In `_template_positions_amplitudes` spike depths are calculated as simply
    the template depth for each spike (so it is the same for all spikes in a
    cluster). Here we need to find the depth of each individual spike, using
    its low-dimensional projection.
    `pc_features` (num_spikes, num_PC, num_channels) holds the PC values for each spike.
    Taking the first component, the subset of 32 channels associated with this
    spike are indexed to get the actual channel locations (in um). Then, the channel
    locations are weighted by their PC values.
    """
    if isinstance(sorter_output, str):
        sorter_output = Path(sorter_output)

    params = _load_ks_dir(sorter_output, load_pcs=True)

    # Compute spike depths
    pc_features = params["pc_features"][:, 0, :]
    pc_features[pc_features < 0] = 0
    pc_features = pc_features**2

    # Get the channel indexes corresponding to the 32 channels from the PC.
    spike_features_indices = params["pc_features_indices"][params["spike_templates"], :]

    ycoords = params["channel_positions"][:, 1]
    spike_feature_ycoords = ycoords[spike_features_indices]

    # TODO: document this, it's from Nick Steinmetz, or phy
    spike_depths = np.sum(spike_feature_ycoords * pc_features, axis=1) / np.sum(
        pc_features, axis=1
    )

    spike_amplitudes, white_templates = _template_positions_amplitudes(
        params["templates"],
        params["whitening_matrix_inv"],
        params["spike_templates"],
        params["temp_scaling_amplitudes"],
    )

    return (
        params["spike_times"],
        spike_amplitudes,
        spike_depths,
        params["spike_templates"],
        white_templates,
        params["channel_positions"],
    )


def _template_positions_amplitudes(
    templates: np.ndarray,
    inverse_whitening_matrix: np.ndarray,
    spike_templates: np.ndarray,
    template_scaling_amplitudes: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Calculate the amplitude and depths of (unwhitened) templates and spikes.

    This function was ported from Nick Steinmetz's `spikes` repository
    MATLAB code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    templates : np.ndarray
        (num_clusters, num_samples, num_channels) array of templates.
    inverse_whitening_matrix: np.ndarray
        Inverse of the whitening matrix used in KS preprocessing, used to
        unwhiten templates.
    ycoords : np.ndarray
        (num_channels,) array of the y-axis (depth) channel positions.
    spike_templates : np.ndarray
        (num_spikes,) array indicating the template associated with each spike.
    template_scaling_amplitudes : np.ndarray
        (num_spikes,) array holding the scaling amplitudes, by which the
        template was scaled to match each spike.

    Returns
    -------
    spike_amplitudes : np.ndarray
        (num_spikes,) array of the amplitude of each spike.
    spike_depths : np.ndarray
        (num_spikes,) array of the depth (probe y-axis) of each spike. Note
        this is just the template depth for each spike (i.e. depth of all spikes
        from the same cluster are identical).
    white_templates : np.ndarray
        Whitened templates (num_clusters, num_samples, num_channels).
    """
    # Unwhiten the template waveforms
    unwhite_templates = templates @ inverse_whitening_matrix

    # First, calculate the depth of each template from the amplitude
    # on each channel by the center of mass method.

    # Take the max amplitude for each channel, then use the channel
    # with most signal as template amplitude. Zero any small channel amplitudes.
    template_amplitudes_per_channel = np.max(unwhite_templates, axis=1) - np.min(
        unwhite_templates, axis=1
    )

    template_amplitudes_unscaled = np.max(template_amplitudes_per_channel, axis=1)

    threshold_values = 0.3 * template_amplitudes_unscaled
    template_amplitudes_per_channel[
        template_amplitudes_per_channel < threshold_values[:, np.newaxis]
    ] = 0

    # Next, find the depth of each spike based on its template. Recompute the template
    # amplitudes as the average of the spike amplitudes ('since
    # tempScalingAmps are equal mean for all templates')
    spike_amplitudes = (
        template_amplitudes_unscaled[spike_templates] * template_scaling_amplitudes
    )

    return (
        spike_amplitudes,
        templates,
    )


def _load_ks_dir(sorter_output: Path, load_pcs: bool = False) -> dict:
    """
    Loads the output of Kilosort into a `params` dict.

    This function was ported from Nick Steinmetz's `spikes` repository MATLAB
    code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    sorter_output : Path
        Path to the kilosort run sorting output.
    load_pcs : bool
        If `True`, principal component (PC) features are loaded.

    Parameters
    ----------
    params : dict
        A dictionary of parameters combining both the kilosort `params.py`
        file as data loaded from `npy` files. The contents of the `npy`
        files can be found in the Phy documentation.

    Notes
    -----
    Template-backed quantities in driftplots are taken from
    `spike_templates.npy`, which preserves Kilosort's original template
    assignment and therefore does not reflect later reassignment in Phy.
    """
    params = read_python(sorter_output / "params.py")

    spike_times = np.load(sorter_output / "spike_times.npy") / params["sample_rate"]
    spike_templates = np.load(sorter_output / "spike_templates.npy")

    temp_scaling_amplitudes = np.load(sorter_output / "amplitudes.npy")

    if load_pcs:
        pc_features = np.load(sorter_output / "pc_features.npy")
        pc_features_indices = np.load(sorter_output / "pc_feature_ind.npy")
    else:
        pc_features = pc_features_indices = None

    new_params = {
        "spike_times": np.asarray(spike_times).reshape(-1),
        "spike_templates": np.asarray(spike_templates).reshape(-1),
        "pc_features": pc_features,
        "pc_features_indices": pc_features_indices,
        "temp_scaling_amplitudes": np.asarray(temp_scaling_amplitudes).reshape(-1),
        "channel_positions": np.load(sorter_output / "channel_positions.npy"),
        "templates": np.load(sorter_output / "templates.npy"),
        "whitening_matrix_inv": np.load(sorter_output / "whitening_mat_inv.npy"),
    }
    params.update(new_params)

    return params
