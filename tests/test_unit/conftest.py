"""Shared synthetic dataset fixtures for unit tests. All features
are tested with mock KS4. SortingAnalzyer and KS1-3 loaders are tested separately.

Templates are 2-D Gaussians (time × channel) placed on a synthetic
NP2-like channel layout with two shanks, two columns per shank, and
non-monotonic depth ordering — matching how real SpikeGLX channel
maps are organised on disk.
"""

import numpy as np
import pytest

NUM_SPIKES = 150
NUM_CLUSTERS = 3
NOISE_CLUSTER_IDS = [0]
TEMPLATE_SAMPLES = 61
SAMPLE_RATE = 30_000.0


def _generate_positions_and_templates(rng):
    """Build synthetic channel positions and templates that mimic real KS output.

    Returns
    -------
    channel_positions : (40, 2) array
    disk_templates : (NUM_CLUSTERS, TEMPLATE_SAMPLES, 40) array
        Templates as they would be stored on disk — signal lives only on
        one shank's channels, embedded in the full 40-channel layout.
    expected_heatmaps : dict[int, (TEMPLATE_SAMPLES, n_signal_chans) array]
        Ground-truth heatmap for each cluster: the active-shank channels
        sorted by depth, signal-only (matching ``get_template_heatmap``
        with mode "heatmap").

    Layout
    ------
    40 channels across 2 shanks, each with 2 columns × 10 depth rows.

    Shank 0: x = {0, 10},  y = 0..9
    Shank 1: x = {50, 60}, y = 0..9

    Channel ordering on disk is **split by shank** (shank 0 first,
    then shank 1), so channel index does NOT correspond to a simple
    depth sweep across the whole probe::

        ch  0: (0,  0)   ─┐
        ch  1: (10, 0)    │ shank 0
        ch  2: (0,  1)    │ depth ↑
        ...               │
        ch 19: (10, 9)   ─┘
        ch 20: (50, 0)   ─┐
        ch 21: (60, 0)    │ shank 1
        ...               │
        ch 39: (60, 9)   ─┘

    Templates are built as clean Gaussians in depth on a single shank,
    then embedded into the full 40-channel layout at the correct
    (scrambled) channel indices.  This is exactly how real Kilosort
    outputs are organised: the template channels follow the channel map,
    not depth order.
    """
    # -- channel positions (interleaved layout) --
    # At each depth row, channels cycle through all 4 columns across both shanks:
    #   (0, 0), (10, 0), (250, 0), (260, 0), (0, 1), (10, 1), (250, 1), (260, 1), ...
    positions = np.empty((40, 2), dtype=np.float64)
    idx = 0
    for depth in range(10):
        for shank_x in (0.0, 250.0):
            positions[idx] = [shank_x, depth]
            positions[idx + 1] = [shank_x + 10.0, depth]
            idx += 2

    # -- build templates --
    mid_t = TEMPLATE_SAMPLES // 2
    time_axis = np.arange(TEMPLATE_SAMPLES, dtype=np.float64)
    time_profile = np.exp(-((time_axis - mid_t) ** 2) / (2 * 6.0**2))

    # Which channel indices belong to each shank?
    # Shank 0: x in {0, 10}, Shank 1: x in {250, 260}
    shank0_idx = np.where(positions[:, 0] < 125)[0]
    shank1_idx = np.where(positions[:, 0] >= 125)[0]

    # Depth (y) for each shank's channels, in channel-index order
    shank0_y = positions[shank0_idx, 1]
    shank1_y = positions[shank1_idx, 1]

    # Cluster definitions: (shank_indices, shank_y, peak_depth, amplitude)
    cluster_defs = [
        (shank0_idx, shank0_y, 2.0, 5.0),   # cluster 0 — noise, moderate amp
        (shank0_idx, shank0_y, 5.0, 10.0),   # cluster 1 — shank 0
        (shank1_idx, shank1_y, 7.0, 15.0),   # cluster 2 — shank 1
    ]

    templates = rng.standard_normal(
        (NUM_CLUSTERS, TEMPLATE_SAMPLES, 40)
    ).astype(np.float32) * 0.01
    expected_heatmaps = {}

    for cluster_id, (shank_idx, shank_y, peak_y, amplitude) in enumerate(
        cluster_defs
    ):
        # Spatial profile: Gaussian decay in depth from peak
        spatial = amplitude * np.exp(-((shank_y - peak_y) ** 2) / (2 * 2.0**2))

        # Clean template on shank channels (TEMPLATE_SAMPLES, n_shank_chans)
        clean_template = np.outer(time_profile, spatial).astype(np.float32)

        # Embed into full 40-channel layout (what goes on disk)
        templates[cluster_id][:, shank_idx] = clean_template

    # Zero the outer 2 depth rows (depths 0, 1, 8, 9) so that "heatmap"
    # mode (which filters out channels where template[mid_t] == 0) returns
    # fewer channels than "heatmap_all_channels".  This ensures the two
    # view modes produce genuinely different outputs in tests.
    outer_depths = {0, 1, 8, 9}
    outer_chan_idx = np.where(np.isin(positions[:, 1], list(outer_depths)))[0]
    templates[:, :, outer_chan_idx] = 0.0

    return positions, templates


@pytest.fixture(scope="session")
def synthetic_data():
    """A deterministic synthetic dataset with Gaussian templates.

    Returns a dict with all arrays and pre-computed reference values
    so tests can assert exact results.
    """
    rng = np.random.default_rng(0)

    channel_locations, templates = _generate_positions_and_templates(rng)
    n_channels = channel_locations.shape[0]

    num_clusters = templates.shape[0]

    # Derive peak channels from templates (channel with max abs signal)
    mid_t = templates.shape[1] // 2
    peak_channels = [
        int(np.argmax(np.abs(templates[c, mid_t, :])))
        for c in range(num_clusters)
    ]

    # Whiten the templates
    whitening_mat, whitening_mat_inv = _make_whitening_matrix(rng, n_channels)
    whitened_templates = np.array(
        [template @ whitening_mat for template in templates], dtype=np.float32
    )

    # Build expected heatmaps from whitened templates (what KS4 stores on
    # disk and what DataModel.get_template_heatmap operates on).
    expected_heatmaps = _build_expected_heatmaps(
        whitened_templates, channel_locations
    )

    spike_templates = np.array(
        [i % num_clusters for i in range(NUM_SPIKES)], dtype=np.int32
    )

    spike_times = np.sort(rng.uniform(1.0, 100.0, NUM_SPIKES))
    spike_times_samples = np.rint(spike_times * SAMPLE_RATE).astype(np.int64)
    spike_times = spike_times_samples / SAMPLE_RATE

    spike_depths = (
        channel_locations[np.array(peak_channels)[spike_templates], 1]
        + rng.uniform(-0.1, 0.1, NUM_SPIKES)
    )

    # Per-spike template-scaling factors (what KS4 saves in amplitudes.npy).
    # Jittered around 1.0 so each spike differs from its template.
    scaling_factors_first_session = 1.0 + 0.1 * rng.standard_normal(NUM_SPIKES)
    scaling_factors_second_session = 1.0 + 0.1 * rng.standard_normal(NUM_SPIKES)

    # Expected spike amplitudes after the loader applies
    # template_ptp_max[cluster] * scaling_factor.
    template_ptp = np.max(whitened_templates, axis=1) - np.min(
        whitened_templates, axis=1
    )
    template_max_peaks = np.max(template_ptp, axis=1)

    spike_amplitudes = template_max_peaks[spike_templates] * scaling_factors_first_session
    spike_amplitudes_second = template_max_peaks[spike_templates] * scaling_factors_second_session

    return {
        "spike_times": spike_times,
        "spike_times_samples": spike_times_samples,
        "spike_amplitudes": spike_amplitudes,
        "spike_amplitudes_second": spike_amplitudes_second,
        "scaling_factors_first_session": scaling_factors_first_session,
        "scaling_factors_second_session": scaling_factors_second_session,
        "spike_depths": spike_depths,
        "spike_templates": spike_templates,
        "templates": templates,
        "whitened_templates": whitened_templates,
        "whitening_mat": whitening_mat,
        "whitening_mat_inv": whitening_mat_inv,
        "channel_locations": channel_locations,
        "peak_channels": peak_channels,
        "expected_heatmaps": expected_heatmaps,
    }


def _write_ks4_output(out, data, scaling_factors_first_session_key):
    """Write synthetic data as KS4-format .npy files to *out*.

    ``spike_times.npy`` is saved in samples and ``amplitudes.npy`` stores
    per-spike template-scaling factors.
    """
    np.save(out / "spike_times.npy", data["spike_times_samples"])
    np.save(out / "spike_templates.npy", data["spike_templates"])
    np.save(out / "templates.npy", data["whitened_templates"])
    np.save(out / "channel_positions.npy", data["channel_locations"])
    np.save(out / "whitening_mat_inv.npy", data["whitening_mat_inv"])

    (out / "params.py").write_text(
        f"sample_rate = {SAMPLE_RATE}\n"
        "dtype = 'int16'\n"
        f"n_channels_dat = {data['channel_locations'].shape[0]}\n"
    )

    spike_positions = np.column_stack([
        np.zeros(data["spike_depths"].size),
        data["spike_depths"],
    ])
    np.save(out / "spike_positions.npy", spike_positions)
    np.save(out / "amplitudes.npy", data[scaling_factors_first_session_key])

    n_clusters = data["whitened_templates"].shape[0]
    lines = ["cluster_id\tKSLabel\n"]
    for i in range(n_clusters):
        label = "noise" if i in NOISE_CLUSTER_IDS else "good"
        lines.append(f"{i}\t{label}\n")
    (out / "cluster_group.tsv").write_text("".join(lines))
    (out / "kilosort4.log").write_text("")

    return out


@pytest.fixture()
def synthetic_ks4_output(tmp_path, synthetic_data):
    """Write synthetic data as KS4-format .npy files to a temp directory."""
    return _write_ks4_output(tmp_path, synthetic_data, "scaling_factors_first_session")


@pytest.fixture()
def synthetic_ks4_output_second(tmp_path, synthetic_data):
    """A second KS4 output directory with different scaling factors.

    Same spike structure as ``synthetic_ks4_output`` but distinct
    per-spike scaling factors so multi-session tests can verify
    concatenation of genuinely different amplitude arrays.
    """
    out = tmp_path / "session_2"
    out.mkdir()
    return _write_ks4_output(out, synthetic_data, "scaling_factors_second_session")


def _build_expected_heatmaps(templates, channel_locations):
    """Build expected heatmaps for "heatmap" mode (i.e. they have
    the zero'd channels cut-off).

    Mirrors ``DataModel.get_template_heatmap``: for each cluster, find
    the shank with max signal, sort channels by depth, keep only those
    with nonzero signal at mid_t.
    """
    mid_t = templates.shape[1] // 2
    COL_CUTOFF_UM = 125

    expected = {}
    for cluster_id in range(templates.shape[0]):
        template = templates[cluster_id]
        max_chan = int(np.argmax(np.max(np.abs(template), axis=0)))
        max_x = channel_locations[max_chan, 0]

        chan_x = channel_locations[:, 0]
        unique_x = np.unique(chan_x)
        valid_x = unique_x[np.abs(unique_x - max_x) < COL_CUTOFF_UM]

        shank_mask = np.zeros(channel_locations.shape[0], dtype=bool)
        for x in valid_x:
            shank_mask |= chan_x == x

        sort_idx = np.argsort(channel_locations[shank_mask, 1])
        sorted_template = template[:, shank_mask][:, sort_idx]

        nonzero_mask = sorted_template[mid_t, :] != 0
        expected[cluster_id] = sorted_template[:, nonzero_mask]

    return expected


def _make_whitening_matrix(rng, n_channels):
    """Create a realistic invertible whitening matrix with local correlations.

    Assumes channels are arranged along a 1D line, so nearby channels are
    more correlated than distant ones.
    """
    positions = np.arange(n_channels, dtype=np.float64)
    dists = np.abs(positions[:, None] - positions[None, :])

    # Covariance: nearby channels are more correlated (regularise with eps)
    length_scale = 5.0
    eps = 1e-5
    covariance = np.exp(-dists / length_scale)
    covariance += eps * np.eye(n_channels, dtype=np.float64)

    # Cholesky: C = L @ L.T
    chol = np.linalg.cholesky(covariance)

    # Right-multiply convention (template @ W) requires W = inv(L).T
    whitening_mat = np.linalg.inv(chol).T
    whitening_mat_inv = chol.T

    return whitening_mat, whitening_mat_inv

