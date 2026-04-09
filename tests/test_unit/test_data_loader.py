from pathlib import Path

import numpy as np
import pytest

from driftplots.data_loader import DataLoader
from driftplots.extractors.kilosort4 import compute_spike_amplitudes
from driftplots.extractors.kilosort_helpers import load_spike_clusters

SORTER_OUTPUT = Path(__file__).parent.parent.parent / "examples" / "example_data" / "sorting" / "sorter_output"

NUM_SPIKES = 100
NUM_CLUSTERS = 5
NUM_CHANNELS = 10
TEMPLATE_SAMPLES = 61
NOISE_CLUSTER_IDS = [0, 2]


@pytest.fixture()
def loader():
    return DataLoader(SORTER_OUTPUT)


@pytest.fixture()
def synthetic_sorter_output(tmp_path):
    """Create a minimal KS4-style sorter output with noise clusters."""
    rng = np.random.default_rng(42)

    spike_times = np.sort(rng.integers(0, 30000, size=NUM_SPIKES)).astype(np.int64)
    amplitudes = rng.uniform(0.5, 5.0, size=NUM_SPIKES).astype(np.float32)
    spike_positions = rng.uniform(0, 3000, size=(NUM_SPIKES, 2)).astype(np.float32)
    spike_clusters = rng.integers(0, NUM_CLUSTERS, size=NUM_SPIKES).astype(np.int32)
    templates = rng.standard_normal((NUM_CLUSTERS, TEMPLATE_SAMPLES, NUM_CHANNELS)).astype(np.float32)
    channel_positions = np.column_stack([
        np.zeros(NUM_CHANNELS),
        np.arange(NUM_CHANNELS) * 20.0,
    ]).astype(np.float32)

    np.save(tmp_path / "spike_times.npy", spike_times)
    np.save(tmp_path / "amplitudes.npy", amplitudes)
    np.save(tmp_path / "spike_positions.npy", spike_positions)
    np.save(tmp_path / "spike_clusters.npy", spike_clusters)
    np.save(tmp_path / "templates.npy", templates)
    np.save(tmp_path / "channel_positions.npy", channel_positions)

    # cluster_group.tsv with noise labels
    lines = ["cluster_id\tKSLabel\n"]
    for i in range(NUM_CLUSTERS):
        label = "noise" if i in NOISE_CLUSTER_IDS else "good"
        lines.append(f"{i}\t{label}\n")
    (tmp_path / "cluster_group.tsv").write_text("".join(lines))

    # KS4 log file (triggers KS4 version detection)
    (tmp_path / "kilosort4.log").write_text("")

    return tmp_path


# ---------------------------------------------------------------------------
# Raw loading – check values match the .npy files on disk
# ---------------------------------------------------------------------------
class TestRawLoading:
    """Loaded arrays should exactly match the raw .npy files."""

    def test_spike_times(self, loader):
        expected = np.load(SORTER_OUTPUT / "spike_times.npy")
        np.testing.assert_array_equal(loader._spike_times, expected)

    def test_spike_depths(self, loader):
        expected = np.load(SORTER_OUTPUT / "spike_positions.npy")[:, 1]
        np.testing.assert_array_equal(loader._spike_depths, expected)

    def test_spike_clusters(self, loader):
        expected = load_spike_clusters(SORTER_OUTPUT)
        np.testing.assert_array_equal(loader._spike_clusters, expected)

    def test_spike_amplitudes(self, loader):
        templates = np.load(SORTER_OUTPUT / "templates.npy")
        amplitudes = np.load(SORTER_OUTPUT / "amplitudes.npy")
        spike_clusters = load_spike_clusters(SORTER_OUTPUT)
        expected = compute_spike_amplitudes(templates, spike_clusters, amplitudes)
        np.testing.assert_array_equal(loader._spike_amplitudes, expected)

    def test_arrays_are_read_only(self, loader):
        for arr in (loader._spike_times, loader._spike_amplitudes,
                    loader._spike_depths, loader._spike_clusters,
                    loader.templates, loader.channel_locations):
            assert not arr.flags.writeable


# ---------------------------------------------------------------------------
# get_processed_data – verify each processing option
# ---------------------------------------------------------------------------
class TestProcessedData:
    """Each processing option should visibly change the output."""

    def test_no_processing(self, loader):
        """With all processing off, output matches raw arrays."""
        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        np.testing.assert_array_equal(result.spike_times, loader._spike_times)
        np.testing.assert_array_equal(result.spike_amplitudes, loader._spike_amplitudes)
        np.testing.assert_array_equal(result.spike_depths, loader._spike_depths)
        np.testing.assert_array_equal(result.spike_clusters, loader._spike_clusters)

    def test_decimate_int(self, loader):
        """Decimation by factor keeps every n-th spike."""
        factor = 3
        result = loader.get_processed_data(
            exclude_noise=False, decimate=factor,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size == loader._spike_times[::factor].size
        np.testing.assert_array_equal(result.spike_times, loader._spike_times[::factor])

    def test_decimate_estimate(self, loader):
        """'estimate' decimation targets ~100k spikes."""
        result = loader.get_processed_data(
            exclude_noise=False, decimate="estimate",
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        # Test data has 2684 spikes (< 100k), so no decimation should occur
        assert result.spike_times.size == loader._spike_times.size

    def test_filter_amplitude_absolute(self, loader):
        """Absolute amplitude filter removes spikes outside bounds."""
        amps = loader._spike_amplitudes
        low, high = np.percentile(amps, (25, 75))

        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode="absolute",
            filter_amplitude_values=(low, high),
        )
        assert result.spike_times.size < loader._spike_times.size
        assert np.all(result.spike_amplitudes >= low)
        assert np.all(result.spike_amplitudes <= high)

    def test_filter_amplitude_percentile(self, loader):
        """Percentile amplitude filter removes spikes outside percentile bounds."""
        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode="percentile",
            filter_amplitude_values=(10, 90),
        )
        amps = loader._spike_amplitudes
        low, high = np.percentile(amps, (10, 90))

        assert result.spike_times.size < loader._spike_times.size
        assert np.all(result.spike_amplitudes >= low)
        assert np.all(result.spike_amplitudes <= high)

    def test_exclude_noise(self, loader):
        """Test data has no noise clusters, so count should be unchanged."""
        result = loader.get_processed_data(
            exclude_noise=True, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size == loader._spike_times.size

    def test_combined_decimate_and_filter(self, loader):
        """Combining decimate + amplitude filter reduces spikes further."""
        amps = loader._spike_amplitudes
        low, high = np.percentile(amps, (10, 90))

        result = loader.get_processed_data(
            exclude_noise=False, decimate=2,
            filter_amplitude_mode="absolute",
            filter_amplitude_values=(low, high),
        )
        # Should be fewer than either alone
        only_decimated = loader.get_processed_data(
            exclude_noise=False, decimate=2,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        only_filtered = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode="absolute",
            filter_amplitude_values=(low, high),
        )
        assert result.spike_times.size <= only_decimated.spike_times.size
        assert result.spike_times.size <= only_filtered.spike_times.size

    def test_all_arrays_same_length(self, loader):
        """All output arrays must have the same length after processing."""
        result = loader.get_processed_data(
            exclude_noise=False, decimate=2,
            filter_amplitude_mode="percentile",
            filter_amplitude_values=(5, 95),
        )
        n = result.spike_times.size
        assert result.spike_amplitudes.size == n
        assert result.spike_depths.size == n
        assert result.spike_clusters.size == n


# ---------------------------------------------------------------------------
# Noise exclusion – synthetic data with known noise clusters
# ---------------------------------------------------------------------------
class TestExcludeNoise:
    """Test noise exclusion using synthetic data with noise-labelled clusters."""

    def test_noise_spikes_removed(self, synthetic_sorter_output):
        """Spikes from noise clusters should be excluded."""
        loader = DataLoader(synthetic_sorter_output)

        result = loader.get_processed_data(
            exclude_noise=True, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size < loader._spike_times.size
        assert not np.any(np.isin(result.spike_clusters, NOISE_CLUSTER_IDS))

    def test_noise_off_keeps_all(self, synthetic_sorter_output):
        """With exclude_noise=False, noise spikes are kept."""
        loader = DataLoader(synthetic_sorter_output)

        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size == loader._spike_times.size

    def test_noise_spike_count_matches(self, synthetic_sorter_output):
        """Number of removed spikes should equal the noise spike count."""
        loader = DataLoader(synthetic_sorter_output)

        noise_count = np.isin(loader._spike_clusters.ravel(), NOISE_CLUSTER_IDS).sum()

        result = loader.get_processed_data(
            exclude_noise=True, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size == loader._spike_times.size - noise_count
