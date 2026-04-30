"""Tests for the KS1–3 extractor and its integration with DataModel.

These checks intentionally stay separate from the KS4 tests. KS1–3 computes
spike depths from PC features and converts ``amplitudes.npy`` into an
approximate spike amplitude by scaling the unwhitened template peak-to-trough.
KS4 instead reads spike positions directly and uses a different amplitude
proxy, so the expected-value math here is version-specific.

Two spikes from different templates are written to disk with distinct times,
scaling amplitudes, and PC features so each loaded output array varies across
spikes.
"""

import numpy as np
import pytest

from driftplots.data_loader import DataLoader
from driftplots.data_model import DataModel


# ---------------------------------------------------------------------------
# Fixture — write KS1–3 output to tmp_path, load via public function
# ---------------------------------------------------------------------------
@pytest.fixture()
def ks13_data(tmp_path):
    """Build minimal KS1–3 data and write it to disk."""
    num_channels = 4
    template_samples = 61
    sample_rate = 30_000.0
    spike_samples = np.array([15_000, 21_000], dtype=np.int64)
    spike_template_ids = np.array([0, 1], dtype=np.int64)
    spike_scaling_amplitudes = np.array([1.5, 0.75], dtype=np.float64)
    spike_pc1_features = np.array(
        [
            [0.0, 3.0, 1.0, 0.0],
            [1.0, -2.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    channel_positions = np.array(
        [[0.0, 100.0], [0.0, 200.0], [0.0, 300.0], [0.0, 400.0]],
        dtype=np.float64,
    )
    pc_feature_channel_indices = np.tile(
        np.arange(num_channels, dtype=np.int64),
        (2, 1),
    )
    whitening_mat_inv = _build_whitening_mat_inv()
    templates = _build_whitened_templates(template_samples, num_channels)

    data = {
        "sample_rate": sample_rate,
        "template_samples": template_samples,
        "spike_samples": spike_samples,
        "spike_template_ids": spike_template_ids,
        "spike_scaling_amplitudes": spike_scaling_amplitudes,
        "spike_pc1_features": spike_pc1_features,
        "channel_positions": channel_positions,
        "pc_feature_channel_indices": pc_feature_channel_indices,
        "whitening_mat_inv": whitening_mat_inv,
        "templates": templates,
    }

    _write_ks13_sorter_output(tmp_path, data)
    data["sorter_output"] = tmp_path
    return data


def _load_data_model(sorter_output) -> DataModel:
    return DataLoader(sorter_output).get_processed_data(
        exclude_noise=False, decimate=False,
        filter_amplitude_mode=None, filter_amplitude_values=None,
    )

def _write_ks13_sorter_output(sorter_output, data) -> None:
    """Write the synthetic KS1-3 data to disk in sorter-output format."""
    num_spikes = data["spike_samples"].size
    num_channels = data["channel_positions"].shape[0]

    spike_times = data["spike_samples"][:, np.newaxis]
    spike_templates = data["spike_template_ids"][:, np.newaxis]
    amplitudes = data["spike_scaling_amplitudes"][:, np.newaxis]

    pc_features = np.zeros((num_spikes, 3, num_channels), dtype=np.float64)
    pc_features[:, 0, :] = data["spike_pc1_features"]

    (sorter_output / "params.py").write_text(
        f"sample_rate = {data['sample_rate']}\n"
        f"dtype = 'int16'\n"
        f"n_channels_dat = {num_channels}\n"
    )

    np.save(sorter_output / "spike_times.npy", spike_times)
    np.save(sorter_output / "spike_templates.npy", spike_templates)
    np.save(sorter_output / "amplitudes.npy", amplitudes)
    np.save(sorter_output / "templates.npy", data["templates"])
    np.save(sorter_output / "channel_positions.npy", data["channel_positions"])
    np.save(sorter_output / "whitening_mat_inv.npy", data["whitening_mat_inv"])
    np.save(sorter_output / "pc_features.npy", pc_features)
    np.save(
        sorter_output / "pc_feature_ind.npy",
        data["pc_feature_channel_indices"],
    )

    (sorter_output / "cluster_group.tsv").write_text("cluster_id\tKSLabel\n0\tgood\n")
    (sorter_output / "kilosort3.log").write_text("")


def _build_whitened_templates(template_samples: int, num_channels: int) -> np.ndarray:
    """Build two distinct whitened templates for the synthetic KS1-3 data.

    The two templates have distinct shapes and dominant channels so the loaded
    amplitudes and heatmaps differ across spikes.
    """
    templates = np.zeros((2, template_samples, num_channels), dtype=np.float64)
    mid = template_samples // 2

    waveform_0 = np.array(
        [-2.0, -6.0, -3.0, 1.0, 5.0, 11.0, 4.0], dtype=np.float64
    )
    templates[0, mid - 3: mid + 4, 1] = waveform_0
    templates[0, mid - 3: mid + 4, 2] = 0.6 * waveform_0

    waveform_1 = np.array(
        [4.0, 9.0, 3.0, -2.0, -7.0, -10.0, -5.0], dtype=np.float64
    )
    templates[1, mid - 3: mid + 4, 0] = waveform_1
    templates[1, mid - 3: mid + 4, 3] = 0.4 * waveform_1

    return templates


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

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKilosort1_3:

    def test_spike_depth_from_pc_centre_of_mass(self, ks13_data):
        """Spike depths should match the fixture's weighted depth centres of mass."""
        data_model = _load_data_model(ks13_data["sorter_output"])
        channel_depths = ks13_data["channel_positions"][:, 1]
        positive_energy = np.square(np.clip(ks13_data["spike_pc1_features"], 0.0, None))

        expected = np.array(
            [np.average(channel_depths, weights=w) for w in positive_energy]
        )

        np.testing.assert_allclose(data_model.spike_depths, expected)

    def test_spike_templates_preserved(self, ks13_data):
        data_model = _load_data_model(ks13_data["sorter_output"])

        np.testing.assert_array_equal(
            data_model.spike_templates, ks13_data["spike_template_ids"]
        )

    def test_unwhitened_amplitude(self, ks13_data):
        """Spike amplitudes must reflect template identity and spike scaling."""
        data_model = _load_data_model(ks13_data["sorter_output"])
        unwhitened = ks13_data["templates"] @ ks13_data["whitening_mat_inv"]

        per_channel = np.max(unwhitened, axis=1) - np.min(unwhitened, axis=1)
        template_amplitudes = np.max(per_channel, axis=1)
        expected = template_amplitudes[ks13_data["spike_template_ids"]] * ks13_data["spike_scaling_amplitudes"]

        np.testing.assert_allclose(data_model.spike_amplitudes, expected)

    def test_spike_times_in_seconds(self, ks13_data):
        """Spike times should be converted from samples to seconds."""
        data_model = _load_data_model(ks13_data["sorter_output"])
        expected = ks13_data["spike_samples"] / ks13_data["sample_rate"]

        np.testing.assert_allclose(data_model.spike_times, expected)

    def test_heatmap_from_data_model(self, ks13_data):
        """Heatmap should contain the whitened template values for channels with signal."""
        data_model = _load_data_model(ks13_data["sorter_output"])

        for spike_index in range(data_model.spike_times.size):
            heatmap = data_model.get_template_heatmap(spike_index, "heatmap")
            template_id = ks13_data["spike_template_ids"][spike_index]

            full_template = ks13_data["templates"][template_id]
            mid = full_template.shape[0] // 2

            signal_channels = np.where(full_template[mid, :] != 0)[0]
            expected = full_template[:, signal_channels]
            np.testing.assert_array_equal(heatmap, expected)
