import numpy as np
import pytest

from driftplots.data_loader import DataLoader
from tests.test_unit.conftest import NOISE_CLUSTER_IDS

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
        np.testing.assert_array_equal(loader._spike_templates, synthetic_data["spike_templates"])
        np.testing.assert_array_almost_equal(loader._spike_amplitudes, synthetic_data["spike_amplitudes"])

        # Templates are loaded as-is from templates.npy (whitened for KS4)
        np.testing.assert_array_equal(loader.templates, synthetic_data["whitened_templates"])

        # Channel locations should match exactly
        np.testing.assert_array_equal(loader.channel_locations, synthetic_data["channel_locations"])

        # All arrays should be read-only
        for arr in (loader._spike_times, loader._spike_amplitudes,
            loader._spike_depths,
                loader._spike_templates,
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
        np.testing.assert_array_equal(result.spike_templates, loader._spike_templates)

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
        loader._spike_templates = rng.integers(0, 3, n)

        result = loader.get_processed_data(
            exclude_noise=False, decimate="estimate",
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        expected_factor = n // 100_000  # 3
        assert result.spike_times.size == len(loader._spike_times[::expected_factor])

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

    def test_combined_decimate_and_filter(self, synthetic_ks4_output):
        """Combined result equals filter-then-decimate (applied in that order)."""
        loader = DataLoader(synthetic_ks4_output)
        amps = loader._spike_amplitudes
        low, high = np.percentile(amps, (10, 90))

        result = loader.get_processed_data(
            exclude_noise=False, decimate=2,
            filter_amplitude_mode="absolute",
            filter_amplitude_values=(low, high),
        )
        only_filtered = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode="absolute",
            filter_amplitude_values=(low, high),
        )
        # Amplitude filter is applied before decimation, so the combined
        # result should exactly match slicing the filtered-only output.
        np.testing.assert_array_equal(result.spike_times, only_filtered.spike_times[::2])
        np.testing.assert_array_equal(result.spike_amplitudes, only_filtered.spike_amplitudes[::2])
        np.testing.assert_array_equal(result.spike_depths, only_filtered.spike_depths[::2])

        # Strictly fewer spikes than either operation alone
        only_decimated = loader.get_processed_data(
            exclude_noise=False, decimate=2,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size < only_decimated.spike_times.size
        assert result.spike_times.size < only_filtered.spike_times.size

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
        assert result.spike_templates.size == n


# ---------------------------------------------------------------------------
# Noise exclusion – synthetic data with known noise clusters
# ---------------------------------------------------------------------------
class TestExcludeNoise:
    """Test noise exclusion – template 0 is labelled noise in synthetic data."""

    def test_noise_spikes_removed(self, synthetic_ks4_output):
        """Spikes from noise clusters should be excluded."""
        loader = DataLoader(synthetic_ks4_output)
        result = loader.get_processed_data(
            exclude_noise=True, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        keep = ~np.isin(loader._spike_templates.ravel(), NOISE_CLUSTER_IDS)
        np.testing.assert_array_equal(result.spike_times, loader._spike_times[keep])
        np.testing.assert_array_equal(result.spike_amplitudes, loader._spike_amplitudes[keep])
        np.testing.assert_array_equal(result.spike_depths, loader._spike_depths[keep])
        np.testing.assert_array_equal(result.spike_templates, loader._spike_templates[keep])

    def test_noise_off_keeps_all(self, synthetic_ks4_output):
        """With exclude_noise=False, noise spikes are kept."""
        loader = DataLoader(synthetic_ks4_output)
        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert result.spike_times.size == loader._spike_times.size


# ---------------------------------------------------------------------------
# Template heatmap reconstruction
# ---------------------------------------------------------------------------
class TestTemplateHeatmap:
    """Verify get_template_heatmap reconstructs the correct depth-ordered template."""

    @pytest.mark.parametrize("template_id", range(3))
    def test_heatmap_matches_ground_truth(
        self, synthetic_ks4_output, synthetic_data, template_id,
    ):
        """Reconstructed heatmap should match the known ground truth template."""
        loader = DataLoader(synthetic_ks4_output)
        result = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )

        # Find the first spike assigned to this template
        spike_idx = int(np.where(result.spike_templates == template_id)[0][0])

        heatmap = result.get_template_heatmap(spike_idx, "heatmap")
        expected = synthetic_data["expected_heatmaps"][template_id]

        np.testing.assert_array_almost_equal(heatmap, expected)
