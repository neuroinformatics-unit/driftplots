from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    import numpy as np
import numpy as np
from spikeinterface.core import read_python


def get_spikes_info_ks1_3(
    sorter_output: str | Path,
) -> tuple[np.ndarray, ...]:
    """
    Compute the amplitude and depth of all detected spikes from the kilosort output.

    This function was ported from Nick Steinmetz's `spikes` repository
    MATLAB code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    sorter_output : str | Path
        Path to the kilosort run sorting output.

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
    In `_template_positions_amplitudes` spike depths is calculated as simply the template
    depth, for each spike (so it is the same for all spikes in a cluster). Here we need
    to find the depth of each individual spike, using its low-dimensional projection.
    `pc_features` (num_spikes, num_PC, num_channels) holds the PC values for each spike.
    Taking the first component, the subset of 32 channels associated with this
    spike  are indexed to get the actual channel locations (in um). Then, the channel
    locations are weighted by their PC values.
    """
    if isinstance(sorter_output, str):
        sorter_output = Path(sorter_output)

    params = _load_ks_dir(sorter_output, load_pcs=True)

    # Compute spike depths
    pc_features = params["pc_features"][:, 0, :]
    pc_features[pc_features < 0] = 0

    # Get the channel indexes corresponding to the 32 channels from the PC.
    spike_features_indices = params["pc_features_indices"][params["spike_templates"], :]

    ycoords = params["channel_positions"][:, 1]
    spike_feature_ycoords = ycoords[spike_features_indices]

    spike_depths = np.sum(spike_feature_ycoords * pc_features ** 2, axis=1) / np.sum(
        pc_features ** 2, axis=1)

    # Compute amplitudes, scale if required and drop un-localised spikes before returning.
    spike_amplitudes, _, _, _, unwhite_templates, *_ = _template_positions_amplitudes(
        params["templates"],
        params["whitening_matrix_inv"],
        ycoords,
        params["spike_templates"],
        params["temp_scaling_amplitudes"],
    )

    return params["spike_times"], spike_amplitudes, spike_depths


def _template_positions_amplitudes(
    templates: np.ndarray,
    inverse_whitening_matrix: np.ndarray,
    ycoords: np.ndarray,
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
    template_amplitudes : np.ndarray
        (num_templates,) Amplitude of each template, calculated as average of spike amplitudes.
    template_depths : np.ndarray
        (num_templates,) array of the depth of each template.
    unwhite_templates : np.ndarray
        Unwhitened templates (num_clusters, num_samples, num_channels).
    trough_peak_durations : np.ndarray
        (num_templates, ) array of durations from trough to peak for each template waveform
    waveforms : np.ndarray
        (num_templates, num_samples) Waveform of each template, taken as the signal on the maximum loading channel.
    """
    # Unwhiten the template waveforms
    unwhite_templates = np.zeros_like(templates)
    for idx, template in enumerate(templates):
        unwhite_templates[idx, :, :] = templates[idx, :, :] @ inverse_whitening_matrix

    # First, calculate the depth of each template from the amplitude
    # on each channel by the center of mass method.

    # Take the max amplitude for each channel, then use the channel
    # with most signal as template amplitude. Zero any small channel amplitudes.
    template_amplitudes_per_channel = np.max(unwhite_templates, axis=1) - np.min(unwhite_templates, axis=1)

    template_amplitudes_unscaled = np.max(template_amplitudes_per_channel, axis=1)

    threshold_values = 0.3 * template_amplitudes_unscaled
    template_amplitudes_per_channel[template_amplitudes_per_channel < threshold_values[:, np.newaxis]] = 0

    # Calculate the template depth as the center of mass based on channel amplitudes
    template_depths = np.sum(template_amplitudes_per_channel * ycoords[np.newaxis, :], axis=1) / np.sum(
        template_amplitudes_per_channel, axis=1
    )

    # Next, find the depth of each spike based on its template. Recompute the template
    # amplitudes as the average of the spike amplitudes ('since
    # tempScalingAmps are equal mean for all templates')
    spike_amplitudes = template_amplitudes_unscaled[spike_templates] * template_scaling_amplitudes

    # Take the average of all spike amplitudes to get actual template amplitudes
    # (since tempScalingAmps are equal mean for all templates)
    num_indices = templates.shape[0]
    sum_per_index = np.zeros(num_indices, dtype=np.float64)
    np.add.at(sum_per_index, spike_templates, spike_amplitudes)
    counts = np.bincount(spike_templates, minlength=num_indices)
    template_amplitudes = np.divide(sum_per_index, counts, out=np.zeros_like(sum_per_index), where=counts != 0)

    # Each spike's depth is the depth of its template
    spike_depths = template_depths[spike_templates]

    # Get channel with the largest amplitude (take that as the waveform)
    max_site = np.argmax(np.max(np.abs(templates), axis=1), axis=1)

    # Use template channel with max signal as waveform
    waveforms = np.empty(templates.shape[:2])
    for idx, template in enumerate(templates):
        waveforms[idx, :] = templates[idx, :, max_site[idx]]

    # Get trough-to-peak time for each template. Find the trough as the
    # minimum signal for the template waveform. The duration (in
    # samples) is the num samples from trough to the largest value
    # following the trough.
    waveform_trough = np.argmin(waveforms, axis=1)

    trough_peak_durations = np.zeros(waveforms.shape[0])
    for idx, tmp_max in enumerate(waveforms):
        trough_peak_durations[idx] = np.argmax(tmp_max[waveform_trough[idx] :])

    return (
        spike_amplitudes,
        spike_depths,
        template_depths,
        template_amplitudes,
        unwhite_templates,
        trough_peak_durations,
        waveforms,
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
    exclude_noise : bool
        If `True`, units labelled as "noise` are removed from all
        returned arrays (i.e. both units and associated spikes are dropped).
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
    When merging and splitting in `Phy`, all changes are made to the
    `spike_clusters.npy` (cluster assignment per spike) and `cluster_groups`
    csv/tsv which contains the quality assignment (e.g. "noise") for each cluster.
    As this function strips the spikes and units based on only these two
    data structures, they will work following manual reassignment in Phy.
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
        "spike_times": spike_times.squeeze(),
        "spike_templates": spike_templates.squeeze(),
        "pc_features": pc_features,
        "pc_features_indices": pc_features_indices,
        "temp_scaling_amplitudes": temp_scaling_amplitudes.squeeze(),
        "channel_positions": np.load(sorter_output / "channel_positions.npy"),
        "templates": np.load(sorter_output / "templates.npy"),
        "whitening_matrix_inv": np.load(sorter_output / "whitening_mat_inv.npy"),
    }
    params.update(new_params)

    return params
