from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si

from driftplots.data_loader import DataLoader

ANALYZER_PATH = Path(__file__).parent.parent.parent / "examples" / "example_data" / "analyzer.zarr"

pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning",
)


# ---------------------------------------------------------------------------
# Loading from sorter output – values match known synthetic data
# ---------------------------------------------------------------------------
class TestFromSorterOutput:
    """DataLoader should correctly load and compute arrays from KS4 on-disk output."""

    def test_loaded_arrays_match_known_values_and_are_read_only(self, synthetic_ks4_output, synthetic_data):
        loader = DataLoader(synthetic_ks4_output)

        # Spike times and depths should match what was written
        np.testing.assert_array_equal(loader._spike_times, synthetic_data["spike_times"])
        np.testing.assert_array_almost_equal(loader._spike_depths, synthetic_data["spike_depths"])
        np.testing.assert_array_equal(loader._spike_clusters, synthetic_data["spike_clusters"])

        # Templates are loaded as-is (whitened) from templates.npy
        np.testing.assert_array_equal(loader.templates, synthetic_data["whitened_templates"])

        # Channel locations should match exactly
        np.testing.assert_array_equal(loader.channel_locations, synthetic_data["channel_locations"])

        # All arrays should be read-only
        for arr in (loader._spike_times, loader._spike_amplitudes,
                    loader._spike_depths, loader._spike_clusters,
                    loader.templates, loader.channel_locations):
            assert not arr.flags.writeable


# ---------------------------------------------------------------------------
# Loading from SortingAnalyzer – values match analyzer extensions
# ---------------------------------------------------------------------------
class TestFromSortingAnalyzer:
    """DataLoader should correctly load arrays from a SortingAnalyzer."""

    def test_loaded_arrays_match_analyzer_extensions_and_are_read_only(self):
        analyzer = si.load_sorting_analyzer(ANALYZER_PATH)
        loader = DataLoader(analyzer)

        # Expected values from the analyzer extensions
        random_spike_indices = analyzer.get_extension("random_spikes").data["random_spikes_indices"]
        spike_vector = analyzer.sorting.to_spike_vector()

        expected_times = spike_vector["sample_index"][random_spike_indices] / analyzer.sorting.get_sampling_frequency()
        np.testing.assert_array_equal(loader._spike_times, expected_times)

        expected_amplitudes = np.abs(analyzer.get_extension("spike_amplitudes").data["amplitudes"])
        np.testing.assert_array_equal(loader._spike_amplitudes, expected_amplitudes)

        expected_depths = analyzer.get_extension("spike_locations").data["spike_locations"]["y"]
        np.testing.assert_array_equal(loader._spike_depths, expected_depths)

        expected_clusters = spike_vector["unit_index"][random_spike_indices]
        np.testing.assert_array_equal(loader._spike_clusters, expected_clusters)

        expected_templates = analyzer.get_extension("templates").data["average"]
        np.testing.assert_array_equal(loader.templates, expected_templates)

        np.testing.assert_array_equal(loader.channel_locations, analyzer.get_channel_locations())

        # All arrays should be read-only
        for arr in (loader._spike_times, loader._spike_amplitudes,
                    loader._spike_depths, loader._spike_clusters,
                    loader.templates, loader.channel_locations):
            assert not arr.flags.writeable


# ---------------------------------------------------------------------------
# get_processed_data – verify each processing option
# ---------------------------------------------------------------------------
class TestProcessedData:
    """Each processing option should visibly change the output."""

    def test_no_processing(self, synthetic_ks4_output):
        """With all processing off, output matches raw arrays."""
        loader = DataLoader(synthetic_ks4_output)
        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        np.testing.assert_array_equal(result.spike_times, loader._spike_times)
        np.testing.assert_array_equal(result.spike_amplitudes, loader._spike_amplitudes)
        np.testing.assert_array_equal(result.spike_depths, loader._spike_depths)
        np.testing.assert_array_equal(result.spike_clusters, loader._spike_clusters)

    def test_decimate_int(self, synthetic_ks4_output):
        """Decimation by factor keeps every n-th spike."""
        loader = DataLoader(synthetic_ks4_output)
        factor = 3
        result = loader.get_processed_data(
            exclude_noise=False, decimate=factor,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size == loader._spike_times[::factor].size
        np.testing.assert_array_equal(result.spike_times, loader._spike_times[::factor])

    def test_decimate_estimate(self, synthetic_ks4_output):
        """'estimate' decimation targets ~100k spikes."""
        loader = DataLoader(synthetic_ks4_output)
        result = loader.get_processed_data(
            exclude_noise=False, decimate="estimate",
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        # Synthetic data has 50 spikes (< 100k), so no decimation should occur
        assert result.spike_times.size == loader._spike_times.size

    def test_decimate_estimate_over_100k(self, synthetic_ks4_output):
        """'estimate' decimation should downsample when spikes exceed 100k."""
        loader = DataLoader(synthetic_ks4_output)
        n = 300_000
        rng = np.random.default_rng(0)
        loader._spike_times = np.linspace(0, 1000, n)
        loader._spike_amplitudes = rng.uniform(1, 10, n)
        loader._spike_depths = rng.uniform(0, 3840, n)
        loader._spike_clusters = rng.integers(0, 3, n)

        result = loader.get_processed_data(
            exclude_noise=False, decimate="estimate",
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        expected_factor = n // 100_000  # 3
        assert result.spike_times.size == n // expected_factor + (1 if n % expected_factor else 0)

    def test_filter_amplitude_absolute(self, synthetic_ks4_output):
        """Absolute amplitude filter removes spikes outside bounds."""
        loader = DataLoader(synthetic_ks4_output)
        amplitudes = loader._spike_amplitudes
        low, high = np.percentile(amplitudes, (25, 75))

        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode="absolute",
            filter_amplitude_values=(low, high),
        )
        assert result.spike_times.size < loader._spike_times.size
        assert np.all(result.spike_amplitudes >= low)
        assert np.all(result.spike_amplitudes <= high)

    def test_filter_amplitude_percentile(self, synthetic_ks4_output):
        """Percentile amplitude filter removes spikes outside percentile bounds."""
        loader = DataLoader(synthetic_ks4_output)
        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode="percentile",
            filter_amplitude_values=(10, 90),
        )
        amplitudes = loader._spike_amplitudes
        low, high = np.percentile(amplitudes, (10, 90))

        assert result.spike_times.size < loader._spike_times.size
        assert np.all(result.spike_amplitudes >= low)
        assert np.all(result.spike_amplitudes <= high)

    def test_exclude_noise(self, synthetic_ks4_output):
        """exclude_noise with no noise labels should leave count unchanged."""
        loader = DataLoader(synthetic_ks4_output)
        result = loader.get_processed_data(
            exclude_noise=True, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size == loader._spike_times.size

    def test_combined_decimate_and_filter(self, synthetic_ks4_output):
        """Combining decimate + amplitude filter reduces spikes further."""
        loader = DataLoader(synthetic_ks4_output)
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

    def test_all_arrays_same_length(self, synthetic_ks4_output):
        """All output arrays must have the same length after processing."""
        loader = DataLoader(synthetic_ks4_output)
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

    NOISE_CLUSTER_IDS = [0]  # must match conftest.NOISE_CLUSTER_IDS

    @pytest.fixture()
    def loader(self, synthetic_ks4_output_with_noise):
        return DataLoader(synthetic_ks4_output_with_noise)

    def test_noise_spikes_removed(self, loader):
        """Spikes from noise clusters should be excluded."""
        result = loader.get_processed_data(
            exclude_noise=True, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size < loader._spike_times.size
        assert not np.any(np.isin(result.spike_clusters, self.NOISE_CLUSTER_IDS))

    def test_noise_off_keeps_all(self, loader):
        """With exclude_noise=False, noise spikes are kept."""
        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size == loader._spike_times.size

    def test_noise_spike_count_matches(self, loader):
        """Number of removed spikes should equal the noise spike count."""
        noise_count = np.isin(loader._spike_clusters.ravel(), self.NOISE_CLUSTER_IDS).sum()

        result = loader.get_processed_data(
            exclude_noise=True, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size == loader._spike_times.size - noise_count
