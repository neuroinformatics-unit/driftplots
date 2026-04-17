"""Tests for the KS1–3 extractor and its integration with DataModel.

Two identical spikes from a single cluster are written to disk in KS1–3
format. The public ``get_spikes_info_ks1_3`` loader is called once
and the results are wrapped in a ``DataModel`` for all assertions.
Hand-computed expected values make failures easy to diagnose.
"""

import numpy as np
import pytest

from driftplots.data_model import DataModel
from driftplots.extractors.kilosort1_3 import get_spikes_info_ks1_3

# ---------------------------------------------------------------------------
# Constants — tweak here to change the synthetic scenario
# ---------------------------------------------------------------------------
NUM_CHANNELS = 4
TEMPLATE_SAMPLES = 61
SAMPLE_RATE = 30_000.0
SPIKE_SAMPLE = 15_000  # spike at 0.5 s
SCALING_FACTOR = 1.5  # per-spike template-scaling amplitude
NUM_SPIKES = 2


def _build_whitening_mat_inv():
    """A simple non-trivial invertible 4×4 matrix.

    Using a matrix that is clearly not the identity so unwhitening
    is genuinely exercised.
    """
    return np.array(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.1, 1.0, 0.1, 0.0],
            [0.0, 0.1, 1.0, 0.1],
            [0.0, 0.0, 0.1, 1.0],
        ],
        dtype=np.float64,
    )


def _build_whitened_template():
    """(1, TEMPLATE_SAMPLES, NUM_CHANNELS) whitened template.

    A simple pulse at the middle time-step on channels 1 and 2 only.
    """
    template = np.zeros((1, TEMPLATE_SAMPLES, NUM_CHANNELS), dtype=np.float64)
    mid = TEMPLATE_SAMPLES // 2
    template[0, mid, 1] = 10.0  # channel 1
    template[0, mid, 2] = 6.0  # channel 2
    return template


def _expected_unwhitened_template():
    """What the template should look like after ``@ whitening_mat_inv``."""
    whitened = _build_whitened_template()[0]  # (T, C)
    W_inv = _build_whitening_mat_inv()
    return whitened @ W_inv  # (T, C)


def _expected_template_amplitude_unscaled():
    """Max channel (max-min) of the unwhitened template."""
    uw = _expected_unwhitened_template()  # (T, C)
    per_chan = np.max(uw, axis=0) - np.min(uw, axis=0)
    return float(np.max(per_chan))


def _expected_spike_amplitude():
    return _expected_template_amplitude_unscaled() * SCALING_FACTOR


def _expected_spike_depth():
    """PC-weighted centre-of-mass depth for each synthetic spike.

    pc_features[:, 0, :] for each spike = [0.0, 3.0, 1.0, 0.0].
    Negative values clipped to 0, then squared → [0, 9, 1, 0].
    Channel y-coords (via pc_feature_ind [0,1,2,3]) → [100, 200, 300, 400].
    depth = (0*100 + 9*200 + 1*300 + 0*400) / (0+9+1+0) = 2100/10 = 210.
    """
    return 210.0


# ---------------------------------------------------------------------------
# Fixture — write KS1–3 output to tmp_path, load via public function
# ---------------------------------------------------------------------------
@pytest.fixture()
def ks13_data_model(tmp_path):
    """Write a minimal KS1–3 output dir and return a DataModel."""
    channel_positions = np.array(
        [[0.0, 100.0], [0.0, 200.0], [0.0, 300.0], [0.0, 400.0]]
    )
    templates = _build_whitened_template()
    W_inv = _build_whitening_mat_inv()

    spike_times = np.full((NUM_SPIKES, 1), SPIKE_SAMPLE, dtype=np.int64)
    spike_templates = np.zeros((NUM_SPIKES, 1), dtype=np.int64)
    amplitudes = np.full((NUM_SPIKES, 1), SCALING_FACTOR, dtype=np.float64)

    # PC features: (n_spikes, n_pcs=3, n_pc_channels=4)
    pc_features = np.zeros((NUM_SPIKES, 3, NUM_CHANNELS), dtype=np.float64)
    pc_features[:, 0, :] = [0.0, 3.0, 1.0, 0.0]  # only PC0 matters

    # pc_feature_ind: (n_clusters=1, n_pc_channels=4) — maps to real channels
    pc_feature_ind = np.array([[0, 1, 2, 3]], dtype=np.int64)

    # params.py — spikeinterface's read_python just exec's the file
    (tmp_path / "params.py").write_text(
        f"sample_rate = {SAMPLE_RATE}\n"
        f"dtype = 'int16'\n"
        f"n_channels_dat = {NUM_CHANNELS}\n"
    )

    np.save(tmp_path / "spike_times.npy", spike_times)
    np.save(tmp_path / "spike_templates.npy", spike_templates)
    np.save(tmp_path / "amplitudes.npy", amplitudes)
    np.save(tmp_path / "templates.npy", templates)
    np.save(tmp_path / "channel_positions.npy", channel_positions)
    np.save(tmp_path / "whitening_mat_inv.npy", W_inv)
    np.save(tmp_path / "pc_features.npy", pc_features)
    np.save(tmp_path / "pc_feature_ind.npy", pc_feature_ind)

    # cluster_group.tsv — single cluster, labelled "good"
    (tmp_path / "cluster_group.tsv").write_text(
        "cluster_id\tKSLabel\n0\tgood\n"
    )

    # KS version detection — any kilosort*.log that isn't kilosort4.log
    (tmp_path / "kilosort3.log").write_text("")

    # Load through the public API
    (
        spike_times_out,
        spike_amplitudes,
        spike_depths,
        spike_templates_out,
        white_templates,
        channel_positions_out,
    ) = get_spikes_info_ks1_3(tmp_path)

    return DataModel(
        spike_times=spike_times_out,
        spike_amplitudes=spike_amplitudes,
        spike_depths=spike_depths,
        spike_templates=spike_templates_out,
        templates=white_templates,
        channel_locations=channel_positions_out,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestKS13SpikeDepth:
    def test_spike_depth_from_pc_centre_of_mass(self, ks13_data_model):
        """Depth should be the PC-weighted centre-of-mass of channel y-coords."""
        np.testing.assert_allclose(
            ks13_data_model.spike_depths[0],
            _expected_spike_depth(),
        )


class TestKS13TemplateUnwhitening:
    def test_unwhitened_amplitude_matches_hand_calculation(self, ks13_data_model):
        """Spike amplitude must reflect unwhitened template × scaling factor."""
        np.testing.assert_allclose(
            ks13_data_model.spike_amplitudes[0],
            _expected_spike_amplitude(),
        )


class TestKS13SpikeAmplitude:
    def test_amplitude_is_template_amp_times_scaling(self, ks13_data_model):
        """amplitude = max-channel(max-min) of unwhitened template × scaling."""
        expected_unscaled = _expected_template_amplitude_unscaled()
        expected = expected_unscaled * SCALING_FACTOR
        np.testing.assert_allclose(
            ks13_data_model.spike_amplitudes[0],
            expected,
        )


class TestKS13SpikeTime:
    def test_spike_time_in_seconds(self, ks13_data_model):
        """Spike time should be spike_sample / sample_rate."""
        np.testing.assert_allclose(
            ks13_data_model.spike_times[0],
            SPIKE_SAMPLE / SAMPLE_RATE,
        )


class TestKS13TemplateHeatmap:
    def test_heatmap_from_data_model(self, ks13_data_model):
        """DataModel.get_template_heatmap should work on the loaded templates."""
        heatmap = ks13_data_model.get_template_heatmap(0, "heatmap")
        assert heatmap.ndim == 2
        assert heatmap.shape[0] == TEMPLATE_SAMPLES
