"""Shared synthetic dataset fixtures for unit tests.

Templates are 2-D Gaussians (time × channel) with distinct scalings
per cluster, placed on the real NP2 channel layout from the example
data.  Because the channel y-positions are not monotonically sorted,
we also provide a depth-reordered copy of each template for easy
assertion.
"""
from pathlib import Path

import numpy as np
import pytest

from driftplots.data_model import DataModel

EXAMPLE_CHANNEL_POSITIONS = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "example_data"
    / "sorting"
    / "sorter_output"
    / "channel_positions.npy"
)

NUM_SPIKES = 50
NUM_CLUSTERS = 3
TEMPLATE_SAMPLES = 61
MID_SAMPLE = TEMPLATE_SAMPLES // 2


def _make_gaussian_template(
    n_samples: int,
    n_channels: int,
    peak_channel: int,
    amplitude: float,
    sigma_t: float = 6.0,
    sigma_c: float = 3.0,
) -> np.ndarray:
    """Create a (n_samples, n_channels) 2-D Gaussian centred at (mid, peak_channel)."""
    time_axis = np.arange(n_samples, dtype=np.float64)
    chan_axis = np.arange(n_channels, dtype=np.float64)
    time_grid, chan_grid = np.meshgrid(time_axis, chan_axis, indexing="ij")
    template = amplitude * np.exp(
        -((time_grid - n_samples / 2) ** 2) / (2 * sigma_t**2)
        - ((chan_grid - peak_channel) ** 2) / (2 * sigma_c**2)
    )
    return template.astype(np.float32)


@pytest.fixture(scope="session")
def channel_locations():
    """Real NP2 channel positions (384, 2)."""
    return np.load(EXAMPLE_CHANNEL_POSITIONS)


def _make_whitening_matrix(rng, n_channels):
    """Create a realistic invertible whitening matrix.

    Constructs a positive-definite matrix from a random orthogonal
    basis with decaying eigenvalues, mimicking a real whitening
    transform.  Returns both the matrix and its inverse.
    """
    # Random orthogonal matrix via QR decomposition
    random_mat = rng.standard_normal((n_channels, n_channels))
    orthogonal, _ = np.linalg.qr(random_mat)

    # Decaying eigenvalues (largest channels decorrelate most)
    eigenvalues = np.exp(-0.01 * np.arange(n_channels))

    whitening_mat = (orthogonal * eigenvalues) @ orthogonal.T
    whitening_mat = whitening_mat.astype(np.float32)

    whitening_mat_inv = np.linalg.inv(whitening_mat.astype(np.float64)).astype(
        np.float32
    )
    return whitening_mat, whitening_mat_inv


def _make_templates(n_channels, peak_channels, template_amplitudes):
    """Build (NUM_CLUSTERS, TEMPLATE_SAMPLES, n_channels) Gaussian templates."""
    templates = np.zeros(
        (NUM_CLUSTERS, TEMPLATE_SAMPLES, n_channels), dtype=np.float32
    )
    for cluster_idx, (peak_chan, amplitude) in enumerate(zip(peak_channels, template_amplitudes)):
        templates[cluster_idx] = _make_gaussian_template(
            TEMPLATE_SAMPLES, n_channels, peak_chan, amplitude
        )
    return templates


def _make_spikes(rng, channel_locations, peak_channels, template_amplitudes, spike_clusters):
    """Generate deterministic spike times, depths and amplitudes.

    Note: in real KS4 data, spike depths come from spike_positions.npy.
    Here we derive them from the peak channel's y-position plus jitter.
    """
    spike_times = np.linspace(1.0, 100.0, NUM_SPIKES).astype(np.float64)

    spike_depths = (
        channel_locations[np.array(peak_channels)[spike_clusters], 1]
        + rng.uniform(-5, 5, NUM_SPIKES)  # small jitter across spikes of the sample template
    ).astype(np.float64)

    spike_amplitudes = np.array(
        [template_amplitudes[cluster] * (1 + 0.1 * rng.standard_normal()) for cluster in spike_clusters],
        dtype=np.float64,
    )

    return spike_times, spike_depths, spike_amplitudes


def _make_reordered_templates(templates, channel_locations, peak_channels):
    """Pre-compute depth-reordered templates matching DataModel.get_template_heatmap.

    For each cluster, selects same-shank channels (|x - peak_x| < 125),
    sorts by y, then produces "signal_only" and "all_channels" variants.
    """
    reordered = {}
    for cluster_id in range(NUM_CLUSTERS):
        peak_chan = peak_channels[cluster_id]
        template = templates[cluster_id]
        peak_x = channel_locations[peak_chan, 0]

        x_locs = channel_locations[:, 0]
        shank_mask = np.abs(x_locs - peak_x) < 125
        depth_order = np.argsort(channel_locations[shank_mask, 1])

        sorted_template = template[:, shank_mask][:, depth_order]

        has_signal = sorted_template[MID_SAMPLE, :] != 0
        signal_only = sorted_template[:, has_signal]

        all_channels = sorted_template.copy()
        all_channels[:, all_channels[MID_SAMPLE, :] == 0] = np.nan

        reordered[cluster_id] = {
            "signal_only": signal_only,
            "all_channels": all_channels,
        }
    return reordered


@pytest.fixture(scope="session")
def synthetic_data(channel_locations):
    """A deterministic synthetic dataset with Gaussian templates.

    Returns a dict with all arrays and pre-computed reference values
    so tests can assert exact results.
    """
    rng = np.random.default_rng(0)
    n_channels = channel_locations.shape[0]

    peak_channels = [10, 25, 40]
    template_amplitudes = [5.0, 10.0, 20.0]

    templates = _make_templates(n_channels, peak_channels, template_amplitudes)

    whitening_mat, whitening_mat_inv = _make_whitening_matrix(rng, n_channels)
    whitened_templates = np.array(
        [template @ whitening_mat for template in templates], dtype=np.float32
    )

    spike_clusters = np.array(
        [i % NUM_CLUSTERS for i in range(NUM_SPIKES)], dtype=np.int32
    )
    spike_times, spike_depths, spike_amplitudes = _make_spikes(
        rng, channel_locations, peak_channels, template_amplitudes, spike_clusters
    )

    reordered_templates = _make_reordered_templates(
        templates, channel_locations, peak_channels
    )

    return {
        "spike_times": spike_times,
        "spike_amplitudes": spike_amplitudes,
        "spike_depths": spike_depths,
        "spike_clusters": spike_clusters,
        "templates": templates,
        "whitened_templates": whitened_templates,
        "whitening_mat": whitening_mat,
        "whitening_mat_inv": whitening_mat_inv,
        "channel_locations": channel_locations,
        "peak_channels": peak_channels,
        "template_amplitudes": template_amplitudes,
        "reordered_templates": reordered_templates,
    }


@pytest.fixture(scope="session")
def synthetic_model(synthetic_data):
    """A DataModel built from the synthetic dataset."""
    data = synthetic_data
    return DataModel(
        data["spike_times"],
        data["spike_amplitudes"],
        data["spike_depths"],
        data["spike_clusters"],
        data["templates"],
        data["channel_locations"],
    )


@pytest.fixture()
def synthetic_ks4_output(tmp_path, synthetic_data):
    """Write synthetic data as KS4-format .npy files to a temp directory.

    KS outputs whitened templates as ``templates.npy``, so that is what
    gets saved here.  The unwhitened templates live only in
    ``synthetic_data["templates"]``.
    """
    data = synthetic_data

    np.save(tmp_path / "spike_times.npy", data["spike_times"])
    np.save(tmp_path / "spike_clusters.npy", data["spike_clusters"])
    np.save(tmp_path / "templates.npy", data["whitened_templates"])
    np.save(tmp_path / "channel_positions.npy", data["channel_locations"])
    np.save(tmp_path / "whitening_mat_inv.npy", data["whitening_mat_inv"])

    # KS4 spike_positions: (n_spikes, 2) with depth in column 1
    spike_positions = np.column_stack([
        np.zeros(data["spike_depths"].size),
        data["spike_depths"],
    ])
    np.save(tmp_path / "spike_positions.npy", spike_positions)

    # KS4 amplitudes: per-spike template-scaling factors.
    # Use varied values so amplitude filtering tests can reduce the count.
    rng = np.random.default_rng(99)
    ks4_amplitudes = rng.uniform(0.5, 2.0, size=data["spike_times"].size).astype(np.float32)
    np.save(tmp_path / "amplitudes.npy", ks4_amplitudes)

    # cluster_group.tsv: all clusters labelled "good" (no noise)
    n_clusters = data["templates"].shape[0]
    lines = ["cluster_id\tKSLabel\n"]
    for i in range(n_clusters):
        lines.append(f"{i}\tgood\n")
    (tmp_path / "cluster_group.tsv").write_text("".join(lines))

    # Version detection: KS4 is identified by a kilosort4.log file
    (tmp_path / "kilosort4.log").write_text("")

    return tmp_path


NOISE_CLUSTER_IDS = [0]


@pytest.fixture()
def synthetic_ks4_output_with_noise(tmp_path, synthetic_data):
    """Same as synthetic_ks4_output but cluster 0 is labelled noise.

    Reuses the shared synthetic data so noise-exclusion tests don't
    need a separate, independently generated dataset.
    """
    data = synthetic_data

    np.save(tmp_path / "spike_times.npy", data["spike_times"])
    np.save(tmp_path / "spike_clusters.npy", data["spike_clusters"])
    np.save(tmp_path / "templates.npy", data["whitened_templates"])
    np.save(tmp_path / "channel_positions.npy", data["channel_locations"])
    np.save(tmp_path / "whitening_mat_inv.npy", data["whitening_mat_inv"])

    spike_positions = np.column_stack([
        np.zeros(data["spike_depths"].size),
        data["spike_depths"],
    ])
    np.save(tmp_path / "spike_positions.npy", spike_positions)

    rng = np.random.default_rng(99)
    ks4_amplitudes = rng.uniform(0.5, 2.0, size=data["spike_times"].size).astype(np.float32)
    np.save(tmp_path / "amplitudes.npy", ks4_amplitudes)

    n_clusters = data["templates"].shape[0]
    lines = ["cluster_id\tKSLabel\n"]
    for i in range(n_clusters):
        label = "noise" if i in NOISE_CLUSTER_IDS else "good"
        lines.append(f"{i}\t{label}\n")
    (tmp_path / "cluster_group.tsv").write_text("".join(lines))

    (tmp_path / "kilosort4.log").write_text("")

    return tmp_path
